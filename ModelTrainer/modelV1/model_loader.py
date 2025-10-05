import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel('ERROR')

# ------------------------- TFNNClassifier (unpickle-compatible, match training) -------------------------
class TFNNClassifier(BaseEstimator, ClassifierMixin):
    """TensorFlow Neural Network for tabular binary classification (recall-focused)."""
    def __init__(self, input_dim, epochs=30, batch_size=64, threshold=0.8):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = None
        self.classes_ = None  # Initialize classes_ attribute

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
        return model

    def fit(self, X, y):
        # Store unique classes for scikit-learn compatibility
        self.classes_ = np.unique(y)

        # Stratified split for validation to ensure both classes are represented
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self.model = self.build_model()
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))

        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_recall',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )

        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            class_weight=class_weight_dict,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        return self

    def predict(self, X):
        probs = self.model.predict(X, verbose=0).flatten()
        return (probs > self.threshold).astype(int)

    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - probs, probs])

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop('model', None)
        if model is not None:
            state['_tf_model_json'] = model.to_json()
            state['_tf_model_weights'] = model.get_weights()
        return state

    def __setstate__(self, state):
        tf_model_json = state.pop('_tf_model_json', None)
        tf_model_weights = state.pop('_tf_model_weights', None)
        self.__dict__.update(state)
        if tf_model_json is not None:
            model = tf.keras.models.model_from_json(tf_model_json)
            model.set_weights(tf_model_weights)
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
            self.model = model
        else:
            self.model = None


# ------------------------- ModelLoader (Singleton + Optimized) -------------------------
class ModelLoader:
    """Load stacking model plus preprocessors and provide a predict API. Singleton pattern for reuse (e.g., HTTP endpoints)."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_dir: str = 'models/v1'):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.num_imputer = None
        self.label_encoders = {}
        self.feature_list = None
        self._load_all()

    def _path(self, name: str) -> str:
        return os.path.join(self.model_dir, name)

    def _load_all(self):
        # Load stacking model WITHOUT mmap to avoid attribute access issues
        model_path = self._path('stacking_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.model = joblib.load(model_path)  # No mmap_mode
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_path}': {e}")

        # Load preprocessors (optional)
        try:
            self.scaler = joblib.load(self._path('global_scaler.pkl'))
        except Exception:
            self.scaler = None

        try:
            self.num_imputer = joblib.load(self._path('num_imputer.pkl'))
        except Exception:
            self.num_imputer = None

        try:
            self.label_encoders = joblib.load(self._path('label_encoders.pkl')) or {}
        except Exception:
            self.label_encoders = {}

        try:
            self.feature_list = joblib.load(self._path('feature_list.pkl'))
        except Exception:
            self.feature_list = None

        if self.feature_list is None:
            # Infer from model (now reliable without mmap)
            try:
                self.feature_list = list(self.model.feature_names_in_)
                if self.feature_list is None and hasattr(self.model, 'named_estimators_'):
                    for name, est in self.model.named_estimators_.items():
                        if hasattr(est, 'feature_names_in_'):
                            self.feature_list = list(est.feature_names_in_)
                            break
            except Exception:
                self.feature_list = None
            # Fallback: if still None, use whatever columns come in prepare_input

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        # Derived features used in training preprocessing
        if 'pl_radj' in df.columns:
            df['density_proxy'] = 1.0 / (df['pl_radj'].replace(0, np.nan) ** 3 + 1e-12)
            df['density_proxy'] = df['density_proxy'].fillna(0.0)
        if 'pl_orbper' in df.columns and 'st_teff' in df.columns:
            df['habitability_proxy'] = (df['pl_orbper'] * 0.7) / (df['st_teff'].replace(0, np.nan) + 1e-12)
            df['habitability_proxy'] = df['habitability_proxy'].fillna(0.0)
        if 'depth' in df.columns and 'pl_trandur' in df.columns:
            df['transit_shape_proxy'] = df['depth'] / (df['pl_trandur'].replace(0, np.nan) + 1e-12)
            df['transit_shape_proxy'] = df['transit_shape_proxy'].fillna(0.0)
        return df

    def prepare_input(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """Turn a dict of user inputs into a single-row DataFrame with all features in correct order."""
        df = pd.DataFrame([input_dict])

        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute derived
        df = self._compute_derived(df)

        # Add missing features as NaN if feature_list known
        if self.feature_list is not None:
            for feat in self.feature_list:
                if feat not in df.columns:
                    df[feat] = np.nan
            # Reindex to exact order
            df = df.reindex(columns=self.feature_list)

        # Impute numerics (use imputer or median fallback)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.num_imputer is not None and numeric_cols:
            try:
                df[numeric_cols] = self.num_imputer.transform(df[numeric_cols])
            except Exception:
                medians = df[numeric_cols].median()
                df[numeric_cols] = df[numeric_cols].fillna(medians)
        else:
            medians = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(medians)

        # Categorical: apply encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda v: le.transform([v])[0] if v in le.classes_ else len(le.classes_))

        # Scale if available
        if self.scaler is not None:
            try:
                scaled = self.scaler.transform(df)
                df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
            except Exception:
                pass

        # Final check: if feature_list, ensure columns match
        if self.feature_list is not None and list(df.columns) != self.feature_list:
            df = df.reindex(columns=self.feature_list).fillna(0)
            # print("DEBUG: Reindexed columns to:", list(df.columns))  # Uncomment for debug

        return df

    def predict(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """Return class (0/1) and probability for positive class. Use 0.5 threshold for consistency with training."""
        # Final reindex to model's expected features (extra safety)
        model_features = getattr(self.model, 'feature_names_in_', None)
        
        if model_features is not None:
            input_df = input_df.reindex(columns=model_features).fillna(0)
        
        X = input_df

        try:
            proba = self.model.predict_proba(X)
            pos_prob = float(proba[:, 1][0])
            pred = int(pos_prob > 0.5)  # Threshold for stacking meta-learner
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")

        return {"class": pred, "probability": pos_prob}


# ------------------------- Console UI -------------------------
REQUIRED_FIELDS = [
    ('pl_radj', 'Bán kính hành tinh (Earth radii). Ví dụ: 1.0'),
    ('pl_orbper', 'Chu kỳ quỹ đạo (days). Ví dụ: 365.25'),
    ('pl_trandur', 'Thời lượng transit (hours). Ví dụ: 10.5'),
    ('depth', 'Độ sâu transit (fraction, not ppm). Nhập fraction: 0.01 = 1%'),
    ('st_teff', 'Nhiệt độ hiệu dụng của sao (Kelvin). Ví dụ: 5778'),
    ('st_logg', 'log(g) (cgs). Ví dụ: 4.44'),
    ('st_rad', 'Bán kính sao (Solar radii). Ví dụ: 1.0'),
    ('koi_kepmag', 'Độ sáng (Kepler magnitude). Ví dụ: 12.5'),
]

OPTIONAL_FIELDS = [
    ('koi_impact', 'Impact parameter (optional)'),
    ('pl_insol', 'Insolation (optional)'),
    ('pl_eqt', 'Effective temperature (optional)'),
    ('st_dist', 'Stellar distance (optional)')
]


def prompt_for_inputs() -> Dict[str, Any]:
    print('Nhập các giá trị tối thiểu để model dự đoán. Bỏ trống để dùng giá trị mặc định/unknown.')
    data = {}
    for k, desc in REQUIRED_FIELDS:
        while True:
            v = input(f"{k} ({desc}): ").strip()
            if v == '':
                print("Trường bắt buộc, vui lòng nhập giá trị.")
                continue
            try:
                data[k] = float(v)
                break
            except ValueError:
                print('Sai định dạng số, thử lại (ví dụ: 1.0).')
    for k, desc in OPTIONAL_FIELDS:
        v = input(f"{k} (optional) ({desc}): ").strip()
        if v == '':
            data[k] = np.nan
        else:
            try:
                data[k] = float(v)
            except ValueError:
                data[k] = np.nan
    return data


def example_run():
    loader = ModelLoader()
    example_input = {
    'pl_radj': 1.168,  # pl_rade / 11.209
    'pl_orbper': 5.742625,
    'pl_trandur': 2.860293,
    'depth': 0.001106,  # pl_trandep / 1e6
    'st_teff': 5664.0,
    'st_logg': 3.26574,
    'st_rad': 3.86959,
    'koi_kepmag': 8.8849,  # st_tmag
    # Optional (NaN sẽ được handle)
    'koi_impact': np.nan,
    'pl_insol': np.nan,
    'pl_eqt': np.nan,
    'st_dist': 263.729
    }
    X = loader.prepare_input(example_input)
    out = loader.predict(X)
    print('\nExample input prediction:')
    print(f"Predicted class: {out['class']} (1 = candidate/confirmed), probability={out['probability']:.4f}")

#demo
if __name__ == '__main__':
    print('--- Model Loader Console ---')
    try:
        loader = ModelLoader()
    except Exception as e:
        print(f"Failed to initialize ModelLoader: {e}")
        sys.exit(1)

    use_example = input('Muốn chạy ví dụ nhanh? (y/n): ').strip().lower()
    if use_example == 'y':
        example_run()
        sys.exit(0)

    user_input = prompt_for_inputs()
    prepared = loader.prepare_input(user_input)
    result = loader.predict(prepared)

    print('\n--- Prediction result ---')
    print(f"Class (0=FP, 1=CAND/CONF): {result['class']}")
    print(f"Probability (positive class): {result['probability']:.4f}")
    print('\nIf you want to call this program from another script, import ModelLoader and use prepare_input/predict.')