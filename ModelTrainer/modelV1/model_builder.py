import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, recall_score, PrecisionRecallDisplay, RocCurveDisplay, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.layers import Input #type:ignore
import joblib
import time
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

# Compatibility for different TF versions
try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="RecallClass0")
class RecallClass0(tf.keras.metrics.Metric):
    """Custom metric for recall of class 0 (specificity)."""
    def __init__(self, name='recall_class0', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Class 0: TN = (1 - y_true) * (1 - y_pred), FP = (1 - y_true) * y_pred
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())

    def reset_states(self):
        self.true_negatives.assign(0.)
        self.false_positives.assign(0.)

class EpochLogger(tf.keras.callbacks.Callback):
    """Custom callback to log metrics for each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: "
              f"loss: {logs['loss']:.4f}, "
              f"accuracy: {logs['accuracy']:.4f}, "
              f"recall_0: {logs.get('recall_class0', 0):.4f}, "
              f"val_loss: {logs['val_loss']:.4f}, "
              f"val_accuracy: {logs['val_accuracy']:.4f}, "
              f"val_recall_0: {logs.get('val_recall_class0', 0):.4f}")

class TFNNClassifier(BaseEstimator, ClassifierMixin):
    """TensorFlow Neural Network for tabular binary classification (recall_0-focused)."""
    def __init__(self, input_dim, epochs=50, batch_size=64, threshold=0.85):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = None
        self.classes_ = None  # Initialize classes_ attribute

    def build_model(self):
        model = tf.keras.Sequential([
            Input(shape=(self.input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', RecallClass0()])
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
        # Boost weight for class 0 to focus more
        class_weight_dict[0] *= 2.0

        # Early stopping focused on recall_0
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_recall_class0',
            patience=10,
            restore_best_weights=True,
            mode='max'
        )

        # Add epoch logger callback
        epoch_logger = EpochLogger()

        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,  # Set to 0 to avoid default Keras logs, use custom logger instead
            class_weight=class_weight_dict,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, epoch_logger]
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
                          metrics=['accuracy', RecallClass0()])
            self.model = model
        else:
            self.model = None

class StackingBuilder:
    """Builder for Stacking Ensemble."""
    def __init__(self, input_dim, X_train, y_train):
        self.input_dim = input_dim
        self.X_train = X_train
        self.y_train = y_train
        self.model = None
        self.history = []
        self.feature_importances = {}

    def tune_base_estimator(self, name, estimator, param_grid):
        scorer = make_scorer(recall_score, pos_label=0)
        grid = GridSearchCV(estimator, param_grid, cv=StratifiedKFold(n_splits=3), scoring=scorer, n_jobs=1)
        grid.fit(self.X_train, self.y_train)
        print(f"Best params for {name}: {grid.best_params_}")
        return grid.best_estimator_

    def build_base_estimators(self):
        # Tune TFNN - but since no params to tune, just fit
        tf_nn = TFNNClassifier(self.input_dim)
        tf_nn.fit(self.X_train, self.y_train)

        # Tune LGBM with higher weight for class 0
        lgb_estimator = lgb.LGBMClassifier(learning_rate=0.1, random_state=42, verbose=-1, class_weight={0: 4.0, 1: 1.0})
        lgb_param_grid = {'n_estimators': [100]}
        lgb_model = self.tune_base_estimator('lgb', lgb_estimator, lgb_param_grid)

        # Tune RF with higher weight for class 0
        rf_estimator = RandomForestClassifier(max_depth=None, criterion='entropy', random_state=42, class_weight={0: 4.0, 1: 1.0})
        rf_param_grid = {'n_estimators': [100]}
        rf_model = self.tune_base_estimator('rf', rf_estimator, rf_param_grid)

        # Tune XGB with lower scale_pos_weight to reduce bias to positive
        xgb_estimator = xgb.XGBClassifier(learning_rate=0.1, random_state=42, scale_pos_weight=0.25)
        xgb_param_grid = {'n_estimators': [100]}
        xgb_model = self.tune_base_estimator('xgb', xgb_estimator, xgb_param_grid)

        return [
            ('tf_nn', tf_nn),
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('xgb', xgb_model)
        ]

    def build_stacking_model(self):
        estimators = self.build_base_estimators()
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, class_weight={0: 2.5, 1: 1.0}),
            cv=StratifiedKFold(n_splits=5),
            stack_method='predict_proba'
        )

        # Tune final estimator
        param_grid = {'final_estimator__C': [1.0]}
        scorer = make_scorer(recall_score, pos_label=0)
        grid = GridSearchCV(stacking, param_grid, cv=StratifiedKFold(n_splits=3), scoring=scorer, n_jobs=1, return_train_score=True)
        grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_
        self.history = grid.cv_results_['mean_test_score']
        return grid.best_score_

    def extract_feature_importances(self, feature_names):
        for name, est in self.model.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                self.feature_importances[name] = dict(zip(feature_names, est.feature_importances_))
        # For TFNN, skip or implement permutation importance if needed
        return self.feature_importances

    def plot_scores(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, marker='o')
        plt.title('Cross-Validation Recall_0 Scores for Final Estimator')
        plt.xlabel('Parameter Combination Index')
        plt.ylabel('Mean Recall_0 Score')
        plt.grid(True)
        plt.savefig('models/v1/cv_recall_0_scores.png')
        plt.close()

def plot_curves(y_test, y_pred_proba):
    # Precision-Recall Curve
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba[:, 1], ax=ax)
    plt.title('Precision-Recall Curve')
    plt.savefig('models/v1/precision_recall_curve.png')
    plt.close()

    # ROC Curve
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_pred_proba[:, 1], ax=ax)
    plt.title('ROC Curve')
    plt.savefig('models/v1/roc_curve.png')
    plt.close()

def train_v1():
    os.makedirs('models/v1', exist_ok=True)
    current_epoch = int(time.time())
    print(f"Current Epoch Time: {current_epoch}")

    if not os.path.exists('data/merged_processed.csv'):
        raise FileNotFoundError("Run data_preprocess.py first to generate merged_processed.csv")

    df = pd.read_csv('data/merged_processed.csv')
    X = df.drop('disposition', axis=1)
    y = df['disposition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Apply SMOTE to train set only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Check feature importance and select non-zero
    model = lgb.LGBMClassifier(random_state=42, class_weight={0: 3.0, 1: 1.0})
    model.fit(X_train_res, y_train_res)
    importance = pd.Series(model.feature_importances_, index=X_train_res.columns).sort_values(ascending=False)
    print("Top 20 feature importances:")
    print(importance.head(20))

    selected_features = importance[importance > 0].index.tolist()
    X_train_res = X_train_res[selected_features]
    X_test = X_test[selected_features]

    input_dim = X_train_res.shape[1]
    print(f"Input dim after feature selection: {input_dim} (Enough for model!)")

    builder = StackingBuilder(input_dim, X_train_res, y_train_res)
    best_recall_0 = builder.build_stacking_model()
    print(f"Tuned Recall_0: {best_recall_0:.2f}")

    builder.plot_scores()
    print("Cross-validation scores plot saved as 'models/v1/cv_recall_0_scores.png'")

    model = builder.model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    plot_curves(y_test, y_pred_proba)
    print("Precision-Recall and ROC curves saved in 'models/v1/'")

    feature_importances = builder.extract_feature_importances(X_train_res.columns)
    for name, imp in feature_importances.items():
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        print(f"Feature importances for {name}:")
        for feat, score in sorted_imp:
            print(f"{feat}: {score}")

    # Save importances to file
    with open('models/v1/feature_importances.txt', 'w') as f:
        for name, imp in feature_importances.items():
            f.write(f"Feature importances for {name}:\n")
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            for feat, score in sorted_imp:
                f.write(f"{feat}: {score}\n")
            f.write("\n")

    print("Feature importances saved to 'models/v1/feature_importances.txt'")

    
    joblib.dump(model, 'models/v1/stacking_model.pkl')
    print("v1 trained & saved!")

if __name__ == "__main__":
    train_v1()