import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from scipy import sparse

from tabular_training import (
    prepare_tabular_features,
    split_training_data,
    compute_balanced_weights,
)
from training_data_pipeline import build_training_dataframe


MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

def ensure_output_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0),
    }

    return metrics

def save_model_bundle(model_name, model, metrics, label_encoder, feature_columns):
    joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")
    joblib.dump(label_encoder, MODELS_DIR / f"{model_name}_label_encoder.pkl")
    joblib.dump(feature_columns, MODELS_DIR / f"{model_name}_feature_columns.pkl")

    with open(METRICS_DIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", random_state=42,)
    model.fit(X_train, y_train)
    return model


def train_sgd_classifier(X_train, y_train):
    model = SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", random_state=42, tol=1e-3)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(to_dense_if_needed(X_train), y_train)
    return model

def train_xgboost(X_train, y_train, sample_weights):
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.2, random_state=42, eval_metric="mlogloss")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.1, random_state=42, class_weight="balanced", verbose=-1)
    model.fit(X_train, y_train)
    return model

def train_catboost(X_train, y_train):
    model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, loss_function="MultiClass", verbose=0)
    model.fit(to_dense_if_needed(X_train), y_train)
    return model

def train_calibrated_xgboost(X_train, y_train, sample_weights):
    base_model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.2, random_state=42, eval_metric="mlogloss")
    base_model.fit(X_train, y_train, sample_weight=sample_weights)

    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def to_dense_if_needed(X):
    if sparse.issparse(X):
        return X.toarray()
    return X

def main():
    ensure_output_dirs()

    training_df = build_training_dataframe(seasons=[2023, 2024, 2025])
    X, y, label_encoder, preprocessor = prepare_tabular_features(training_df)

    X_train, X_test, y_train, y_test = split_training_data(X, y)
    sample_weights = compute_balanced_weights(y_train)

    trainers = [
        ("logistic_regression", lambda: train_logistic_regression(X_train, y_train)),
        ("sgd_classifier", lambda: train_sgd_classifier(X_train, y_train)),
        ("random_forest", lambda: train_random_forest(X_train, y_train)),
        ("xgboost", lambda: train_xgboost(X_train, y_train, sample_weights)),
        ("lightgbm", lambda: train_lightgbm(X_train, y_train)),
        ("catboost", lambda: train_catboost(X_train, y_train)),
        ("calibrated_xgboost", lambda: train_calibrated_xgboost(X_train, y_train, sample_weights)),
    ]

    for model_name, trainer in trainers:
        print(f"\nTraining {model_name}...")
        model = trainer()

        evaluation_X = to_dense_if_needed(X_test) if model_name in {"random_forest", "catboost"} else X_test
        metrics = evaluate_model(model, evaluation_X, y_test, label_encoder)
        print(f"{model_name} accuracy: {metrics['accuracy']:.4f}")

        save_model_bundle(
            model_name=model_name,
            model=model,
            metrics=metrics,
            label_encoder=label_encoder,
            feature_columns=preprocessor,
        )

if __name__ == "__main__":
    main()
