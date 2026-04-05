import json
from pathlib import Path


MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

MODEL_DESCRIPTIONS = {
    "logistic_regression": "Balanced logistic regression baseline for tabular pitch prediction.",
    "sgd_classifier": "Online-friendly linear baseline using SGD with log-loss.",
    "random_forest": "Random forest classifier for tabular pitch prediction.",
    "xgboost": "Primary gradient-boosted tree model for tabular pitch prediction.",
    "lightgbm": "LightGBM gradient-boosted tree model for tabular pitch prediction.",
    "catboost": "CatBoost gradient-boosted tree model for tabular pitch prediction.",
    "calibrated_xgboost": "Probability-calibrated XGBoost model for better confidence estimates.",
}


def load_metrics(model_name):
    metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_active_model(model_name, metrics):
    return metrics is not None


def build_registry(source="local"):
    registry = []

    for model_path in sorted(MODELS_DIR.glob("*.pkl")):
        if model_path.name.endswith("_label_encoder.pkl") or model_path.name.endswith("_feature_columns.pkl"):
            continue

        model_name = model_path.stem
        label_encoder_path = MODELS_DIR / f"{model_name}_label_encoder.pkl"
        feature_columns_path = MODELS_DIR / f"{model_name}_feature_columns.pkl"
        metrics = load_metrics(model_name)

        if not label_encoder_path.exists() or not feature_columns_path.exists():
            continue

        entry = {
            "name": model_name,
            "description": MODEL_DESCRIPTIONS.get(model_name, ""),
            "model_filename": model_path.name,
            "label_encoder_filename": label_encoder_path.name,
            "feature_columns_filename": feature_columns_path.name,
            "metrics_filename": f"{model_name}_metrics.json",
            "accuracy": metrics.get("accuracy") if metrics else None,
            "active": is_active_model(model_name, metrics),
            "source": source,
        }

        if source == "local":
            entry.update(
                {
                    "model_path": str(model_path.as_posix()),
                    "label_encoder_path": str(label_encoder_path.as_posix()),
                    "feature_columns_path": str(feature_columns_path.as_posix()),
                    "metrics_path": str((METRICS_DIR / f"{model_name}_metrics.json").as_posix()),
                }
            )

        registry.append(entry)

    return registry


def main():
    registry = build_registry(source="local")

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Saved model registry to {REGISTRY_PATH}")
    print(f"Registered {len(registry)} models.")


if __name__ == "__main__":
    main()
