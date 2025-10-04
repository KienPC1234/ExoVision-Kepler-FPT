"""
Contains:
- TFNNClassifier (compatible with the training code) to allow safe unpickling
- ModelLoader: high-performance loader that loads the stacking model plus preprocessors
  (scaler, imputers, label encoders, feature list)
- A simple console script that prompts the user for the minimal required fields,
  computes derived features used by the model, runs preprocessing and prints
  prediction + probability.
- WebSocket service: Loads model, warmup with example, exposes predict via WebSocket
  for other apps to call. Connect to ws://localhost:8765, send JSON {"input": {...}}

How to use console:
  $ python model_loader.py console

How to use WebSocket service:
  $ pip install websockets  # if needed
  $ python model_loader.py service

The service will warmup (load model + run example prediction), log output,
then start WebSocket server. Clients send JSON like:
{"input": {"pl_radj": 1.0, "pl_orbper": 365.25, ...}}
Receives back: {"result": {"class": 1, "probability": 0.85}}

Notes:
- Expects model files in models/v1/ produced by your training scripts:
  stacking_model.pkl, global_scaler.pkl, num_imputer.pkl
- If TensorFlow model weights are inside the pickled TFNNClassifier, this file
  includes the same class definition so unpickling will work.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import ipaddress
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

try:
    import tensorflow as tf
except Exception:
    tf = None

import asyncio
import websockets
import json

# ------------------------- TFNNClassifier (unpickle-compatible) -------------------------
# This matches the class used at training time so joblib.load can reconstruct it.
class TFNNClassifier(object):
    def __init__(self, input_dim=None, epochs=50, batch_size=16, threshold=0.4):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = None
        self.classes_ = None
        # pickled state will include _tf_model_json and _tf_model_weights when present

    def build_model(self):
        if tf is None:
            raise RuntimeError("TensorFlow is required to build the TF model but was not importable.")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop('model', None)
        if model is not None and tf is not None:
            state['_tf_model_json'] = model.to_json()
            state['_tf_model_weights'] = model.get_weights()
        return state

    def __setstate__(self, state):
        tf_model_json = state.pop('_tf_model_json', None)
        tf_model_weights = state.pop('_tf_model_weights', None)
        self.__dict__.update(state)
        if tf_model_json is not None:
            if tf is None:
                raise RuntimeError('Unpickling TFNNClassifier requires tensorflow to be importable')
            model = tf.keras.models.model_from_json(tf_model_json)
            # attempt to build using known input dim
            try:
                model.build((None, self.input_dim))
            except Exception:
                # some saved models may not need explicit build
                pass
            model.set_weights(tf_model_weights)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
            self.model = model
        else:
            self.model = None

    # prediction helpers in case user wants to call directly
    def predict(self, X):
        if self.model is None:
            raise RuntimeError('TF model not loaded inside TFNNClassifier')
        probs = self.model.predict(np.asarray(X), verbose=0, batch_size=self.batch_size).flatten()
        return (probs > self.threshold).astype(int)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError('TF model not loaded inside TFNNClassifier')
        probs = self.model.predict(np.asarray(X), verbose=0, batch_size=self.batch_size).flatten()
        return np.column_stack([1 - probs, probs])


# ------------------------- ModelLoader -------------------------
class ModelLoader:
    """Load stacking model plus preprocessors and provide a predict API.

    Expects directory structure:
      models/v1/stacking_model.pkl
      models/v1/global_scaler.pkl
      models/v1/num_imputer.pkl
      models/v1/label_encoders.pkl
      models/v1/feature_list.pkl
    """

    def __init__(self, model_dir: str = 'models/v1'):
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
        # Load stacking model (joblib) - use mmap_mode to reduce memory if data large
        model_path = self._path('stacking_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_path}': {e}")

        # Load preprocessors (optional if not present)
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
            # try to infer from model if possible
            try:
                self.feature_list = getattr(self.model, 'feature_names_in_', None)
                if self.feature_list is None and hasattr(self.model, 'named_estimators_'):
                    for name, est in self.model.named_estimators_.items():
                        if hasattr(est, 'feature_names_in_'):
                            self.feature_list = list(est.feature_names_in_)
                            break
            except Exception:
                self.feature_list = None

    def _compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        # Derived features used in training preprocessing
        if 'pl_radj' in df.columns:
            # density proxy approximation (avoid division by zero)
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
        """Turn a dict of user inputs into a single-row DataFrame with all features in correct order.
        Missing numerical values are imputed; missing optional features are set to NaN and then imputed.
        """
        # Start from dict -> df
        df = pd.DataFrame([input_dict])

        # Ensure expected numeric types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

        # Compute derived features
        df = self._compute_derived(df)

        # If the training feature_list exists, ensure all features are present.
        if self.feature_list is not None:
            for feat in self.feature_list:
                if feat not in df.columns:
                    df[feat] = np.nan
            # Keep only the columns in feature_list order after prep
        else:
            # If no feature list, we will let scaler handle present numeric columns
            pass

        # Apply imputers for numerical and categorical if available
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.num_imputer is not None and numeric_cols:
            try:
                df[numeric_cols] = self.num_imputer.transform(df[numeric_cols])
            except Exception:
                # num_imputer might be fitted on many columns; simple fallback: fillna with median(0)
                df[numeric_cols] = df[numeric_cols].fillna(0)
        else:
            df[numeric_cols] = df[numeric_cols].fillna(0)

        # Categorical columns: apply saved label encoders where possible
        for col, le in (self.label_encoders or {}).items():
            if col in df.columns:
                # If user provided a value outside training classes, map to a new index (len classes)
                try:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].map(lambda v: le.transform([v])[0] if v in le.classes_ else len(le.classes_))
                except Exception:
                    # fallback: set to 0
                    df[col] = 0

        # Final scaling
        final_cols = self.feature_list if self.feature_list is not None else df.columns.tolist()
        # Reorder
        df = df.reindex(columns=final_cols)

        if self.scaler is not None:
            try:
                scaled = self.scaler.transform(df)
                df_scaled = pd.DataFrame(scaled, columns=final_cols, index=df.index)
                return df_scaled
            except Exception:
                # If scaler fails, fallback to filling NaNs and returning raw
                df = df.fillna(0)
                return df
        else:
            return df.fillna(0)
        

    def predict(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """Return class (0/1) and probability for positive class."""
        input_df = input_df.reindex(columns=self.model.feature_names_in_)
        X = input_df  # giữ DataFrame để không mất tên cột
    
        try:
            proba = self.model.predict_proba(X)
            # assume positive class là column 1
            pos_prob = float(proba[:, 1][0])
            pred = int((pos_prob > 0.7))
        except Exception:
            try:
                pred = int(self.model.predict(X)[0])
                pos_prob = float(np.nan)
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


def example_run(loader):
    # Example: show how to call loader programmatically
    example = {
        'koi_kepmag': 12.5, 'pl_radj': 1.0, 'koi_impact': 0.5, 'pl_trandur': 10.5, 'depth': 0.01, 'pl_orbper': 365.25,
        'st_teff': 5778, 'st_logg': 4.44, 'st_rad': 1.0, 'pl_insol': 1.0, 'pl_eqt': 288, 'st_dist': 100
    }
    X = loader.prepare_input(example)
    out = loader.predict(X)
    print('\nExample input prediction:')
    print(f"Predicted class: {out['class']} (1 = candidate/confirmed), probability={out['probability']:.4f}")
    return out


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'console'
    
    if mode == 'console':
        print('--- Model Loader Console ---')
        try:
            loader = ModelLoader()
        except Exception as e:
            print(f"Failed to initialize ModelLoader: {e}")
            sys.exit(1)

        use_example = input('Muốn chạy ví dụ nhanh? (y/n): ').strip().lower()
        if use_example == 'y':
            example_run(loader)
            sys.exit(0)

        user_input = prompt_for_inputs()
        prepared = loader.prepare_input(user_input)
        result = loader.predict(prepared)

        print('\n--- Prediction result ---')
        print(f"Class (0=FP, 1=CAND/CONF): {result['class']}")
        print(f"Probability (positive class): {result['probability']:.4f}")
        print('\nIf you want to call this program from another script, import ModelLoader and use prepare_input/predict.')
    
    elif mode == 'service':
        print('--- Model Loader WebSocket Service ---')
        try:
            loader = ModelLoader()
        except Exception as e:
            print(f"Failed to initialize ModelLoader: {e}")
            sys.exit(1)

        # Warmup: run example and log
        print("Warmup: Loading model and running example prediction...")
        example_result = example_run(loader)
        print(f"Warmup complete. Example logged above. Server ready.")
        
        async def client_handler(reader, writer):
            addr = writer.get_extra_info('peername')
            ip, port = addr
            print(f"Client connected from {ip}:{port}")

            try:
                while True:
                    data = await reader.readline()
                    if not data:
                        break

                    print(f"[DEBUG] Received message from {ip}:{port}")

                    try:
                        data = json.loads(data)
                        input_dict = data.get("input", {})
                        if not input_dict:
                            response = {"error": "No input was provided"}
                        else:
                            prepared = loader.prepare_input(input_dict)
                            result = loader.predict(prepared)
                            response = {
                                'result': result,
                                'prepared_features': prepared.to_dict('records')[0]
                            }
                    except json.JSONDecodeError:
                        response = {'error': 'Invalid JSON'}
                    except Exception as e:
                        response = {'error': f"Prediction error: {e}"}

                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    print(f"[DEBUG] Sent response to {ip}:{port}")
            except Exception as e:
                print(f"Handler error: {e}")
            finally:
                print(f"Client disconnected from {ip}:{port}")
                writer.close()
                await writer.wait_closed()

        async def server_main():
            server = await asyncio.start_server(client_handler, "127.0.0.1", 8765)
            addr = server.sockets[0].getsockname()
            print(f"TCP server started on {addr} (loopback only)")
            print("Press Ctrl+C to stop.")

            async with server:
                await server.serve_forever()

        asyncio.run(server_main())
    else:
        print("Invalid mode. Use 'console' or 'service'.")
        sys.exit(1)