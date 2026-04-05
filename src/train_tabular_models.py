import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from model_artifacts import save_bundle_metadata
from tabular_training import (
    compute_balanced_weights,
    prepare_tabular_features,
    split_training_data,
)
from training_data_pipeline import build_training_dataframe


MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")


def ensure_output_dirs(models_dir=MODELS_DIR, metrics_dir=METRICS_DIR):
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_model_bundle(
    model_name,
    model,
    metrics,
    label_encoder,
    feature_columns,
    models_dir=MODELS_DIR,
    metrics_dir=METRICS_DIR,
):
    model_filename = f"{model_name}.pkl"
    label_encoder_filename = f"{model_name}_label_encoder.pkl"
    feature_columns_filename = f"{model_name}_feature_columns.pkl"
    metrics_filename = f"{model_name}_metrics.json"

    joblib.dump(model, models_dir / model_filename)
    joblib.dump(label_encoder, models_dir / label_encoder_filename)
    joblib.dump(feature_columns, models_dir / feature_columns_filename)

    with open(metrics_dir / metrics_filename, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    save_bundle_metadata(
        model_name=model_name,
        model_type="tabular",
        model_filename=model_filename,
        label_encoder_filename=label_encoder_filename,
        feature_columns_filename=feature_columns_filename,
        metrics_filename=metrics_filename,
        output_dir=models_dir,
    )


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=3000,
        solver="saga",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_sgd_classifier(X_train, y_train):
    model = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        tol=1e-3,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(to_dense_if_needed(X_train), y_train)
    return model


def train_xgboost(X_train, y_train, sample_weights):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.2,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=-1,
        learning_rate=0.1,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        loss_function="MultiClass",
        verbose=0,
    )
    model.fit(to_dense_if_needed(X_train), y_train)
    return model


def train_calibrated_xgboost(X_train, y_train, sample_weights):
    base_model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.2,
        random_state=42,
        eval_metric="mlogloss",
    )
    base_model.fit(X_train, y_train, sample_weight=sample_weights)

    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def to_dense_if_needed(X):
    if sparse.issparse(X):
        return X.toarray()
    return X


def train_tabular_model_suite(
    seasons=(2023, 2024, 2025),
    models_dir=MODELS_DIR,
    metrics_dir=METRICS_DIR,
):
    ensure_output_dirs(models_dir=models_dir, metrics_dir=metrics_dir)

    training_df = build_training_dataframe(seasons=list(seasons))
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

    model_summaries = []
    best_model_name = None
    best_accuracy = float("-inf")

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
            models_dir=models_dir,
            metrics_dir=metrics_dir,
        )

        model_summaries.append({"name": model_name, "accuracy": metrics["accuracy"]})
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_model_name = model_name

    return {
        "success": True,
        "training_rows": int(training_df.shape[0]),
        "models_trained": model_summaries,
        "best_model": best_model_name,
        "best_accuracy": best_accuracy,
        "models_dir": str(models_dir.as_posix()),
        "metrics_dir": str(metrics_dir.as_posix()),
    }


def main():
    summary = train_tabular_model_suite()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
