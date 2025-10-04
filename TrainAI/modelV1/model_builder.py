import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, recall_score, PrecisionRecallDisplay, RocCurveDisplay
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
from tensorflow.keras.callbacks import TensorBoard #type:ignore
import joblib
import time
import matplotlib.pyplot as plt
from datetime import datetime

tf.get_logger().setLevel('ERROR')

class EpochLogger(tf.keras.callbacks.Callback):
    """Custom callback to log metrics for each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: "
              f"loss: {logs['loss']:.4f}, "
              f"accuracy: {logs['accuracy']:.4f}, "
              f"recall: {logs['recall']:.4f}, "
              f"val_loss: {logs['val_loss']:.4f}, "
              f"val_accuracy: {logs['val_accuracy']:.4f}, "
              f"val_recall: {logs['val_recall']:.4f}")

class TFNNClassifier(BaseEstimator, ClassifierMixin):
    """TensorFlow Neural Network for tabular binary classification (recall-focused)."""
    def __init__(self, input_dim, epochs=50, batch_size=32, threshold=0.7):
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
                      metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])  # Explicitly name the recall metric
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
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=10,
            restore_best_weights=True,
            mode='max'
        )

        # TensorBoard logging
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Add epoch logger callback
        epoch_logger = EpochLogger()

        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,  # Set to 0 to avoid default Keras logs, use custom logger instead
            class_weight=class_weight_dict,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, epoch_logger, tensorboard_callback]
        )
        return self

    def predict(self, X):
        probs = self.model.predict(X, verbose=0).flatten()
        return (probs > self.threshold).astype(int)

    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - probs, probs])

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
        grid = GridSearchCV(estimator, param_grid, cv=StratifiedKFold(n_splits=5), scoring='recall', n_jobs=1)
        grid.fit(self.X_train, self.y_train)
        print(f"Best params for {name}: {grid.best_params_}")
        return grid.best_estimator_

    def build_base_estimators(self):
        # Tune TFNN - but since no params to tune, just fit
        tf_nn = TFNNClassifier(self.input_dim)
        tf_nn.fit(self.X_train, self.y_train)

        # Tune LGBM
        lgb_estimator = lgb.LGBMClassifier(learning_rate=0.1, random_state=42, verbose=-1, class_weight='balanced')
        lgb_param_grid = {'n_estimators': [100, 200]}
        lgb_model = self.tune_base_estimator('lgb', lgb_estimator, lgb_param_grid)

        # Tune RF
        rf_estimator = RandomForestClassifier(max_depth=None, criterion='entropy', random_state=42, class_weight='balanced')
        rf_param_grid = {'n_estimators': [100, 200]}
        rf_model = self.tune_base_estimator('rf', rf_estimator, rf_param_grid)

        # Tune XGB - add tuning for consistency
        xgb_estimator = xgb.XGBClassifier(learning_rate=0.1, random_state=42, scale_pos_weight=2)
        xgb_param_grid = {'n_estimators': [100, 200]}
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
            final_estimator=LogisticRegression(random_state=42, class_weight='balanced'),
            cv=StratifiedKFold(n_splits=10),
            stack_method='predict_proba'
        )

        # Tune final estimator
        param_grid = {'final_estimator__C': [0.1, 1.0]}
        grid = GridSearchCV(stacking, param_grid, cv=StratifiedKFold(n_splits=5), scoring='recall', n_jobs=1, return_train_score=True)
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
        plt.title('Cross-Validation Recall Scores for Final Estimator')
        plt.xlabel('Parameter Combination Index')
        plt.ylabel('Mean Recall Score')
        plt.grid(True)
        plt.savefig('models/v1/cv_recall_scores.png')
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
    model = lgb.LGBMClassifier(random_state=42)
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
    best_recall = builder.build_stacking_model()
    print(f"Tuned Recall: {best_recall:.2f}")

    builder.plot_scores()
    print("Cross-validation scores plot saved as 'models/v1/cv_recall_scores.png'")

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

    os.makedirs('models/v1', exist_ok=True)
    joblib.dump(model, 'models/v1/stacking_model.pkl')
    print("v1 trained & saved!")

if __name__ == "__main__":
    train_v1()