import logging, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tsai.all import PatchTST, get_ts_dls, TSClassification, TSNormalize
sys.path.append(str(Path(__file__).parent))
from model_builder import ClassifierWrapper, _ensure_divisible, CONFIG
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Utilities copied from model_builder.py
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

class SingletonModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info('Creating the SingletonModel instance')
            cls._instance = super(SingletonModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'model'):
            self.load_model()

    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = CONFIG
        self.max_seq_len = int(self.config['max_seq_len'])
        self.window_step = int(self.config.get('window_step', 512))
        self.downsample_factor = int(self.config.get('downsample_factor', 1))
        self.binary_classification = self.config.get('binary_classification', True)
        self.tfms = [None, TSClassification()]
        self.batch_tfms = [TSNormalize()]

        # Paths
        CKPT_PATH = Path("models/v2") / f"{self.config.get('save_fname', 'best_patchtst')}.pth"

        if not CKPT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

        # Load checkpoint
        ckpt = torch.load(CKPT_PATH, map_location=self.device)
        if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model_state_dict' in ckpt):
            state_key = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
            state_dict = ckpt[state_key]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            if isinstance(ckpt, dict):
                cand = None
                for k, v in ckpt.items():
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
        if self.binary_classification:
            self.n_classes = 2
        else:
            if 'head.weight' in state_dict:
                self.n_classes = state_dict['head.weight'].shape[0]
            else:
                self.n_classes = 2

        # Prepare PatchTST params
        pp = dict(self.config.get('model_params', {}))
        pp['seq_len'] = self.max_seq_len
        pp['patch_len'] = int(pp.get('patch_len', pp.get('patch_len', 16)))
        pp['seq_len'] = _ensure_divisible(pp['seq_len'], pp['patch_len'])
        self.patch_len = pp['patch_len']

        # Create base model
        base_model = PatchTST(1, self.n_classes, **pp)  # vars=1 (single channel)

        # Wrap with ClassifierWrapper
        self.model = ClassifierWrapper(base_model, n_classes=self.n_classes)

        # If checkpoint contains 'head.weight', ensure head matches
        if 'head.weight' in state_dict:
            w = state_dict['head.weight']
            in_features = w.shape[1]
            out_features = w.shape[0]
            if getattr(self.model, "head", None) is None or self.model.head.weight.shape[1] != in_features:
                logger.info(f"Creating head: in_features={in_features}, out_features={out_features}")
                self.model.head = nn.Linear(in_features, out_features)

        # Load state_dict
        load_res = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded state_dict with missing_keys={load_res.missing_keys}, unexpected_keys={load_res.unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded and set to eval mode")

    def predict(self, flux: list[float], id: any):
        """
        Predict on a single flux list and id.
        Returns: (predicted_class: int, probability: float) where probability is % for class 1.
        """
        flux_np = np.asarray(flux, dtype=np.float32)
        flux_np = flux_np[np.isfinite(flux_np)]
        if flux_np.size == 0:
            logger.warning(f"Empty flux for id {id}")
            return 0, 0.0

        flux_norm = _normalize(flux_np)
        windows = sliding_windows(flux_norm, window_len=self.max_seq_len, step=self.window_step, downsample=self.downsample_factor)
        if not windows:
            logger.warning(f"No windows for id {id}")
            return 0, 0.0

        X = np.stack(windows)  # (N, L)
        X = X[:, np.newaxis, :]  # (N, 1, L)

        # Create DataLoader (no y for prediction)
        dls = get_ts_dls(X, tfms=None, batch_tfms=self.batch_tfms, bs=32)
        dl = dls.train  # Full dataset as 'train' since no splits

        logits = []
        with torch.no_grad():
            for xb in dl:
                x = xb[0] if isinstance(xb, (tuple, list)) else xb
                log = self.model(x.to(self.device))
                logits.append(log.cpu())

        if not logits:
            return 0, 0.0

        logits = torch.cat(logits, dim=0)
        if self.binary_classification:
            probas = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            # For multi-class, return argmax and max prob; but assuming binary
            preds = logits.argmax(dim=1).numpy()
            probas = torch.softmax(logits, dim=1).max(dim=1)[0].numpy()

        # Handle potential NaN in probas
        if np.isnan(probas).any():
            logger.warning(f"NaN detected in probabilities for id {id}, defaulting to 0.0")
            return 0, 0.0

        mean_proba = np.mean(probas)
        pred_class = 1 if mean_proba > 0.5 else 0
        percent = mean_proba * 100

        logger.info(f"Prediction for id {id}: class={pred_class}, prob={percent:.2f}%")
        return pred_class, percent

    def evaluate(self):
        """
        Evaluate the model on the test set loaded from X_test.npy and y_test.npy.
        Returns a dict of metrics.
        """
        # Load test data
        X_test = np.load('models/v2/X_test.npy')
        y_test = np.load('models/v2/y_test.npy')
        logger.info(f"Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

        # Create DataLoaders (full as train)
        bs = 32
        tfms = [None, TSClassification()]
        batch_tfms = [TSNormalize()]
        dls = get_ts_dls(X_test, y=y_test, splits=None, tfms=tfms, batch_tfms=batch_tfms, bs=bs)

        # Predict logits
        logits = []
        targets = []
        with torch.no_grad():
            for xb, yb in dls.train:
                log = self.model(xb.to(self.device))
                logits.append(log.cpu())
                targets.append(yb.cpu())

        logits = torch.cat(logits, dim=0)
        targets = torch.cat(targets, dim=0)

        # Compute preds and probas
        preds = logits.argmax(dim=1).numpy()
        if self.binary_classification:
            probas = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            probas = torch.softmax(logits, dim=1).max(dim=1)[0].numpy()  # max prob for multi-class

        y_true = targets.numpy()
        y_pred = preds

        # Metrics
        average_mode = 'binary' if self.binary_classification else 'macro'
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average_mode, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average_mode, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_mode, zero_division=0)
        roc = roc_auc_score(y_true, probas) if self.binary_classification else None

        roc_str = f"{roc:.4f}" if roc is not None else 'N/A'

        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc
        }
        logger.info(f"Test metrics — acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, roc_auc: {roc_str}")
        return metrics


# Example usage for evaluate (comment out if not needed)
if __name__ == "__main__":
    model = SingletonModel()
    metrics = model.evaluate()
    print("Final metrics:", metrics)