"""
Train pipeline using tsai PatchTST only.
Replace your existing train.py with this file.
This version trains and exports the model, saving test data separately for eval.py.
"""

import os
import logging
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

# tsai / fastai
from tsai.all import get_ts_dls, PatchTST, TSClassification, TSNormalize,  Learner, F1Score, accuracy
from fastai.callback.tracker import SaveModelCallback


# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
from pathlib import Path
import torch

CONFIG = {
    'file_path': Path('data/koi_lightcurves.parquet'),  
    'cols': ['kepid', 'time', 'flux', 'label'],
    'chunk_rows': 200_000,

    # Windowing / reduction
    'max_seq_len': 1024,        
    'window_step': 512,         
    'downsample_factor': 1,     

    # Training
    'batch_size': 256,
    'num_epochs': 50,           
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'test_size': 0.2,
    'random_state': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 0,
    'binary_classification': True,

    # PatchTST-specific params
    'model_arch': 'PatchTST',
    'model_params': {
        'seq_len': 1024,        
        'patch_len': 16,
        'd_model': 128,
        'n_layers': 3,
        'n_heads': 8,
        'd_ff': 256,
        'dropout': 0.1,
        'revin': False,         
        'pred_dim': None,       
        'subtract_last': False, 
    },

    'save_fname': 'best_patchtst',
}

# ----------------------------------------

class ClassifierWrapper(nn.Module):
    """
    Wrap a base model so that its forward returns logits of shape (B, n_classes)
    by pooling over the time dimension for sequence-level classification.
    """
    def __init__(self, base_model: nn.Module, n_classes: int, pool_type='mean'):
        super().__init__()
        self.base = base_model
        self.n_classes = n_classes
        self.head = None
        self.pool_type = pool_type

    def forward(self, x):
        out = self.base(x)
        if isinstance(out, (tuple, list)):
            out = out[0]  # take first output if tuple

        # handle 3D output (B, C, L) or (B, L, C)
        if out.dim() == 3:
            B, A, C = out.shape
            if A == self.n_classes:  # (B, C, L)
                if self.pool_type == 'mean':
                    logits = out.mean(dim=-1)
                else:  # max pooling
                    logits, _ = out.max(dim=-1)
            elif C == self.n_classes:  # (B, L, C)
                if self.pool_type == 'mean':
                    logits = out.mean(dim=1)
                else:
                    logits, _ = out.max(dim=1)
            else:
                # fallback: pool last dim
                if self.pool_type == 'mean':
                    pooled = out.mean(dim=-1)
                else:
                    pooled, _ = out.max(dim=-1)
                if pooled.shape[1] != self.n_classes:
                    if self.head is None:
                        self.head = nn.Linear(pooled.shape[1], self.n_classes).to(pooled.device)
                    logits = self.head(pooled)
                else:
                    logits = pooled
        elif out.dim() == 2:  # (B, D)
            if out.shape[1] != self.n_classes:
                if self.head is None:
                    self.head = nn.Linear(out.shape[1], self.n_classes).to(out.device)
                logits = self.head(out)
            else:
                logits = out
        else:
            # flatten everything else
            flat = out.view(out.size(0), -1)
            if self.head is None:
                self.head = nn.Linear(flat.shape[1], self.n_classes).to(flat.device)
            logits = self.head(flat)
        return logits

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds(CONFIG['random_state'])

# ---------------- Utilities ----------------
def _normalize(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return a
    m = np.nanmean(a)
    s = np.nanstd(a) + 1e-8
    return (a - m) / s

def sliding_windows(flux, window_len, step, downsample=1):
    if downsample > 1:
        flux = flux[::downsample]
    L = flux.shape[0]
    if L == 0:
        return []
    if L <= window_len:
        out = np.zeros(window_len, dtype=np.float32)
        out[:L] = flux
        return [out]
    windows = []
    for start in range(0, L - window_len + 1, step):
        windows.append(flux[start:start + window_len].astype(np.float32))
    if (L - window_len) % step != 0:
        windows.append(flux[-window_len:].astype(np.float32))
    return windows

def _ensure_divisible(seq_len, patch_len):
    if seq_len % patch_len == 0:
        return seq_len
    new_len = (seq_len // patch_len) * patch_len
    if new_len == 0:
        raise ValueError(f"max_seq_len {seq_len} too small for patch_len {patch_len}")
    logger.info(f"Adjusting max_seq_len {seq_len} -> {new_len} to be divisible by patch_len {patch_len}")
    return new_len

# ---------------- Data loading + windowing (group-aware) ----------------
def load_and_window_data(cfg):
    fp = cfg['file_path']
    if not fp.exists():
        raise FileNotFoundError(fp)

    logger.info("Streaming parquet and grouping by kepid...")
    pf = pq.ParquetFile(fp)
    seq_times = defaultdict(list)
    seq_flux = defaultdict(list)
    label_counts = defaultdict(lambda: defaultdict(int))

    for batch in tqdm(pf.iter_batches(batch_size=cfg['chunk_rows']), desc="Reading parquet"):
        df = batch.to_pandas()
        df = df[cfg['cols']]
        df['label'] = df['label'].fillna('unknown')
        df = df.dropna(subset=['kepid', 'time', 'flux'])
        for kepid, g in df.groupby('kepid'):
            seq_times[kepid].append(g['time'].to_numpy())
            seq_flux[kepid].append(g['flux'].to_numpy())
            if len(g) > 0:
                lab = g['label'].value_counts().idxmax()
                label_counts[kepid][lab] += len(g)

    # Resolve label per kepid by majority
    labels_dict = {}
    for kepid, counts in label_counts.items():
        if 'unknown' in counts and len(counts) > 1:
            del counts['unknown']
        if not counts:
            continue
        max_label = max(counts, key=counts.get)
        labels_dict[kepid] = max_label
        if len(counts) > 1:
            logger.warning(f"Label conflict for kepid {kepid}, resolved to {max_label}")

    logger.info("Merging and windowing sequences...")
    windows = []
    window_labels = []
    window_kepid = []

    max_seq_len = int(cfg['max_seq_len'])
    step = int(cfg['window_step'])
    down = int(cfg.get('downsample_factor', 1))

    # Ensure seq_len divisible by patch_len
    patch_len = int(cfg['model_params'].get('patch_len', 16))
    max_seq_len = _ensure_divisible(max_seq_len, patch_len)
    cfg['max_seq_len'] = max_seq_len
    cfg['model_params']['seq_len'] = max_seq_len

    for kepid in tqdm(seq_times.keys(), desc="Windowing KOIs"):
        if kepid not in labels_dict:
            continue
        times = np.concatenate(seq_times[kepid])
        fluxs = np.concatenate(seq_flux[kepid])
        order = np.argsort(times)
        fluxs = fluxs[order]
        fluxs = fluxs[np.isfinite(fluxs)]
        if fluxs.size == 0:
            continue
        fluxs = _normalize(fluxs)
        wnds = sliding_windows(fluxs, window_len=max_seq_len, step=step, downsample=down)
        for w in wnds:
            if np.allclose(w, 0.0):
                continue
            windows.append(w)
            window_labels.append(labels_dict[kepid])
            window_kepid.append(kepid)

    if not windows:
        raise ValueError("No windows created. Check data and config.")

    logger.info(f"Created {len(windows)} windows from {len(set(window_kepid))} KOIs")
    return windows, window_labels, window_kepid

# ---------------- Prepare DataLoaders (group split by kepid) ----------------
def prepare_dls_groupsplit(windows, labels, kepids, cfg):
    X = np.stack([np.asarray(w, dtype=np.float32) for w in windows])  # (N, L)
    N, L = X.shape
    X = X.reshape(N, 1, L)
    y_raw = np.array(labels)

    # Map to binary if requested (adapt mapping to your labels)
    if cfg['binary_classification']:
        def map_bin(li):
            try:
                li_i = int(li)
            except Exception:
                try:
                    li_i = int(float(li))
                except Exception:
                    return 0
            return 1 if li_i in [1, 2] else 0
        y = np.array([map_bin(li) for li in y_raw], dtype=int)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))

    # Group-aware split
    groups = np.array(kepids)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg['test_size'], random_state=cfg['random_state'])
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    splits = [list(train_idx), list(test_idx)]
    logger.info(f"Group split: train windows={len(train_idx)}, valid windows={len(test_idx)}; unique train kepid={len(set(groups[train_idx]))}, valid kepid={len(set(groups[test_idx]))}")

    tfms = [None, TSClassification()]
    batch_tfms = [TSNormalize()]

    dls = get_ts_dls(X, y=y, splits=splits, tfms=tfms, batch_tfms=batch_tfms,
                     bs=cfg['batch_size'], num_workers=cfg['num_workers'])
    return dls, X, y, train_idx, test_idx, y_raw


# ---------------- Train (PatchTST only) ----------------
def train_model(dls, cfg):
    pp = dict(cfg.get('model_params', {}))
    pp['seq_len'] = int(cfg['max_seq_len'])
    pp['patch_len'] = int(pp.get('patch_len', cfg['model_params'].get('patch_len', 16)))
    pp['seq_len'] = _ensure_divisible(pp['seq_len'], pp['patch_len'])

    logger.info(f"PatchTST params: seq_len={pp['seq_len']}, patch_len={pp['patch_len']}, d_model={pp.get('d_model')}, n_layers={pp.get('n_layers')}")

    base_model = PatchTST(dls.vars, dls.c, **pp)
    logger.info("Built PatchTST base model")

    # Wrap to guarantee (B, n_classes) logits
    model = ClassifierWrapper(base_model, n_classes=dls.c)
    logger.info("Wrapped PatchTST with ClassifierWrapper")

    # Loss function for classification
    loss_func = nn.CrossEntropyLoss()

    # Metrics for classification
    metrics = [accuracy, F1Score(average='macro')]

    save_cb = SaveModelCallback(monitor='valid_loss', fname=cfg.get('save_fname', 'best_patchtst'))
    
    learn = Learner(dls, model, loss_func=loss_func, metrics=metrics, cbs=[save_cb])
    learn.path = Path('models')         
    learn.model_dir = 'v2'   
    learn.to(cfg['device'])

    # Sanity check: one batch forward 
    xb, yb = next(iter(dls.train)) 
    with torch.no_grad(): 
        logits = model(xb.to(cfg['device'])) 
    logger.info(f"Sanity logits shape: {tuple(logits.shape)} (should be (B, n_classes))") 
    
    if logits.dim() != 2 or logits.shape[1] != dls.c: 
        logger.warning("Logits shape unexpected. Check ClassifierWrapper / model output.")

    logger.info("Starting training...")
    learn.fit_one_cycle(cfg['num_epochs'], cfg['lr'], wd=cfg.get('weight_decay', 0.0))
    logger.info("Training finished")

    # Load best weights if saved
    try:
        learn.load(cfg.get('save_fname', 'best_patchtst'))
        logger.info("Loaded best model weights")
    except Exception:
        logger.warning("Best model not found or load failed; using final model")

    return learn

# ---------------- Main ----------------
def main(cfg):
    logger.info(f"CONFIG: seq_len={cfg['max_seq_len']}, step={cfg['window_step']}, downsample={cfg['downsample_factor']}, device={cfg['device']}")
    windows, labels, kepids = load_and_window_data(cfg)
    dls, X, y, train_idx, test_idx, y_raw = prepare_dls_groupsplit(windows, labels, kepids, cfg)
    os.makedirs("models/v2",exist_ok=True)
    # Save test data for separate evaluation
    X_test = X[test_idx].astype(np.float32)
    y_test = y[test_idx].astype(int)
    np.save('models/v2/X_test.npy', X_test)
    np.save('models/v2/y_test.npy', y_test)
    logger.info(f"Saved test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    learn = train_model(dls, cfg)
    print("Training completed. Run model_loader.py for evaluation.")

if __name__ == "__main__":
    main(CONFIG)