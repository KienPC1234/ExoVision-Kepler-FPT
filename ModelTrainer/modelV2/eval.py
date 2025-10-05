# eval.py
"""
Separate evaluation script for the trained PatchTST model.
Run this after train.py to compute metrics on the test set.
Assumes X_test.npy, y_test.npy, and models/best_patchtst.pth are available.
"""

import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from pathlib import Path
from fastai.learner import Learner
from tsai.all import PatchTST, get_ts_dls, TSClassification, TSNormalize

# If your project has train.py with ClassifierWrapper and filter/_ensure helpers,
# you can import them; otherwise we re-define minimal ClassifierWrapper here.
# We'll try to import from train first.
try:
    from model_builder import ClassifierWrapper, filter_patchtst_params, _ensure_divisible, CONFIG
    print("Imported ClassifierWrapper and CONFIG from train.py")
except Exception as e:
    print("Could not import from train.py (will use local CONFIG and ClassifierWrapper). Reason:", e)
    # Minimal copy of ClassifierWrapper (matches your train.py behavior)
    class ClassifierWrapper(nn.Module):
        def __init__(self, base_model: nn.Module, n_classes: int, pool_type='mean'):
            super().__init__()
            self.base = base_model
            self.n_classes = n_classes
            self.head = None
            self.pool_type = pool_type

        def forward(self, x):
            out = self.base(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() == 3:
                # assume (B, L, C) typical -> pool dim=1
                if out.shape[-1] == self.n_classes:  # (B, L, C) with C==n_classes
                    if self.pool_type == 'mean':
                        logits = out.mean(dim=1)
                    else:
                        logits, _ = out.max(dim=1)
                else:
                    # pool last dim then map
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
            elif out.dim() == 2:
                if out.shape[1] != self.n_classes:
                    if self.head is None:
                        self.head = nn.Linear(out.shape[1], self.n_classes).to(out.device)
                    logits = self.head(out)
                else:
                    logits = out
            else:
                flat = out.view(out.size(0), -1)
                if self.head is None:
                    self.head = nn.Linear(flat.shape[1], self.n_classes).to(flat.device)
                logits = self.head(flat)
            return logits

    # Minimal helpers and config fallback
    def filter_patchtst_params(params):
        allowed = {'seq_len', 'd_model', 'n_layers', 'n_heads', 'd_ff', 'patch_len', 'dropout'}
        return {k: v for k, v in params.items() if k in allowed}

    def _ensure_divisible(seq_len, patch_len):
        if seq_len % patch_len == 0:
            return seq_len
        new_len = (seq_len // patch_len) * patch_len
        if new_len == 0:
            raise ValueError(f"max_seq_len {seq_len} too small for patch_len {patch_len}")
        print(f"Adjusting max_seq_len {seq_len} -> {new_len}")
        return new_len

    CONFIG = {
        'max_seq_len': 1024,
        'batch_size': 32,
        'model_params': {
            'seq_len': 1024,
            'patch_len': 16,
            'd_model': 128,
            'n_layers': 3,
            'n_heads': 8,
            'd_ff': 256,
            'dropout': 0.1,
        },
        'binary_classification': True,
        'save_fname': 'best_patchtst',
    }

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model():
    # Load test data
    X_test = np.load('models/v2/X_test.npy')
    y_test = np.load('models/v2/y_test.npy')
    logger.info(f"Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    # Paths
    CKPT_PATH = Path("models/v2") / f"{CONFIG.get('save_fname','best_patchtst')}.pth"

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = torch.device('cuda')

    # Load checkpoint (handle both raw state_dict or wrapped dict)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    # Extract state_dict if file is a dict with keys
    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model_state_dict' in ckpt):
        state_key = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
        state_dict = ckpt[state_key]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        # likely already a state_dict
        state_dict = ckpt
    else:
        # unknown structure, try to find tensor dict inside
        if isinstance(ckpt, dict):
            # pick the largest dict-like tensor mapping
            cand = None
            for k,v in ckpt.items():
                if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                    cand = v
                    break
            if cand is None:
                raise RuntimeError("Unrecognized checkpoint format; cannot locate state_dict inside.")
            state_dict = cand
        else:
            raise RuntimeError("Unrecognized checkpoint format; expected dict of tensors.")

    logger.info(f"Loaded checkpoint from {CKPT_PATH}. Keys in state_dict: {len(state_dict)}")

    # Infer n_classes
    if CONFIG.get('binary_classification', True):
        n_classes = 2
    else:
        # try infer from head.weight if present
        if 'head.weight' in state_dict:
            n_classes = state_dict['head.weight'].shape[0]
        else:
            n_classes = 2

    # Prepare PatchTST params
    arch_params = dict(CONFIG.get('model_params', {}))
    pp = filter_patchtst_params(arch_params)
    pp['seq_len'] = int(CONFIG.get('max_seq_len', pp.get('seq_len', 1024)))
    pp['patch_len'] = int(pp.get('patch_len', arch_params.get('patch_len', 16)))
    pp['seq_len'] = _ensure_divisible(pp['seq_len'], pp['patch_len'])

    # Create base model with vars=1 (single channel) and c=n_classes
    vars_ = 1
    base_model = PatchTST(vars_, n_classes, **pp)

    # Wrap with ClassifierWrapper
    model = ClassifierWrapper(base_model, n_classes=n_classes)

    # If checkpoint contains 'head.weight' but model.head is None, create head with correct in_features
    if 'head.weight' in state_dict:
        w = state_dict['head.weight']
        in_features = w.shape[1]
        out_features = w.shape[0]
        # create head only if missing or mismatched
        if getattr(model, "head", None) is None or getattr(model.head, "weight", None) is None or model.head.weight.shape[1] != in_features:
            logger.info(f"Creating head: in_features={in_features}, out_features={out_features}")
            model.head = nn.Linear(in_features, out_features)

    # Try to load state_dict (non-strict to ignore mismatches)
    load_res = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded state_dict with missing_keys={load_res.missing_keys}, unexpected_keys={load_res.unexpected_keys}")

    # Build DataLoaders from test data (no splits, full as train)
    bs = min(CONFIG.get('batch_size', 32), 64)  # smaller for eval if needed
    tfms = [None, TSClassification()]
    batch_tfms = [TSNormalize()]

    # Since binary, y_test should be 0,1
    dls = get_ts_dls(X_test, y=y_test, splits=None, tfms=tfms, batch_tfms=batch_tfms, bs=bs)

    # Create Learner and attach model
    learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss())

    logger.info("Reconstructed learner from checkpoint")

    # Predict using get_preds (returns raw logits, no decode issue)
    with learn.no_bar(), learn.no_logging():
        logits, targets = learn.get_preds(dl=dls.train)

    # Compute preds and probas
    preds = logits.argmax(dim=1).cpu().numpy()
    probas = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # positive class prob for binary

    logger.info(f"Predictions: logits shape {logits.shape}, targets shape {targets.shape}, preds shape {preds.shape}")

    # Prepare arrays
    targets_arr = targets.cpu().numpy()
    preds_arr = preds
    probas_arr = probas

    # Since binary 0,1, no need for LabelEncoder
    y_true = targets_arr
    y_pred = preds_arr

    # Compute metrics
    average_mode = 'binary'
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average_mode, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average_mode, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_mode, zero_division=0)
    roc = roc_auc_score(y_true, probas_arr)

    logger.info(f"Test metrics — acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, roc_auc: {roc:.4f}")

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc
    }
    return metrics

if __name__ == "__main__":
    metrics = evaluate_model()
    print("Final metrics:", metrics)