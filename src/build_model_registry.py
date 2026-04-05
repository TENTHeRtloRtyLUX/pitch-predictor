import json
from pathlib import Path

from model_artifacts import load_bundle_metadata


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
    "lstm_sequence": "LSTM sequence model using prior pitches in the current at-bat plus static context.",
    "transformer_sequence": "Transformer sequence model using recent at-bat pitch history plus static context.",
}


def load_metrics(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_registry(source="local", models_dir=MODELS_DIR, metrics_dir=METRICS_DIR):
    registry = []
    registered_names = set()

    for bundle_path in sorted(models_dir.glob("*_bundle.json")):
        bundle = load_bundle_metadata(bundle_path)
        model_filename = bundle["model_filename"]
        label_encoder_filename = bundle["label_encoder_filename"]
        metrics_filename = bundle["metrics_filename"]
        feature_columns_filename = bundle.get("feature_columns_filename")
        preprocessor_filename = bundle.get("preprocessor_filename")

        model_path = models_dir / model_filename
        label_encoder_path = models_dir / label_encoder_filename
        metrics_path = metrics_dir / metrics_filename
        feature_columns_path = models_dir / feature_columns_filename if feature_columns_filename else None
        preprocessor_path = models_dir / preprocessor_filename if preprocessor_filename else None

        required_paths = [model_path, label_encoder_path]
        if feature_columns_path:
            required_paths.append(feature_columns_path)
        if preprocessor_path:
            required_paths.append(preprocessor_path)

        if not all(path.exists() for path in required_paths):
            continue

        metrics = load_metrics(metrics_path)
        entry = {
            "name": bundle["name"],
            "description": MODEL_DESCRIPTIONS.get(bundle["name"], bundle.get("description", "")),
            "model_type": bundle.get("model_type", "tabular"),
            "model_filename": model_filename,
            "label_encoder_filename": label_encoder_filename,
            "metrics_filename": metrics_filename,
            "feature_columns_filename": feature_columns_filename,
            "preprocessor_filename": preprocessor_filename,
            "accuracy": metrics.get("accuracy") if metrics else None,
            "active": metrics is not None,
            "source": source,
        }

        if source == "local":
            entry["model_path"] = str(model_path.as_posix())
            entry["label_encoder_path"] = str(label_encoder_path.as_posix())
            entry["metrics_path"] = str(metrics_path.as_posix())
            if feature_columns_path:
                entry["feature_columns_path"] = str(feature_columns_path.as_posix())
            if preprocessor_path:
                entry["preprocessor_path"] = str(preprocessor_path.as_posix())

        registry.append(entry)
        registered_names.add(entry["name"])

    for model_path in sorted(models_dir.glob("*.pkl")):
        if model_path.name.endswith("_label_encoder.pkl") or model_path.name.endswith("_feature_columns.pkl"):
            continue

        model_name = model_path.stem
        if model_name in registered_names:
            continue

        label_encoder_path = models_dir / f"{model_name}_label_encoder.pkl"
        feature_columns_path = models_dir / f"{model_name}_feature_columns.pkl"
        metrics_path = metrics_dir / f"{model_name}_metrics.json"
        if not label_encoder_path.exists() or not feature_columns_path.exists():
            continue

        metrics = load_metrics(metrics_path)
        entry = {
            "name": model_name,
            "description": MODEL_DESCRIPTIONS.get(model_name, ""),
            "model_type": "tabular",
            "model_filename": model_path.name,
            "label_encoder_filename": label_encoder_path.name,
            "metrics_filename": metrics_path.name,
            "feature_columns_filename": feature_columns_path.name,
            "preprocessor_filename": None,
            "accuracy": metrics.get("accuracy") if metrics else None,
            "active": metrics is not None,
            "source": source,
        }

        if source == "local":
            entry["model_path"] = str(model_path.as_posix())
            entry["label_encoder_path"] = str(label_encoder_path.as_posix())
            entry["feature_columns_path"] = str(feature_columns_path.as_posix())
            entry["metrics_path"] = str(metrics_path.as_posix())

        registry.append(entry)

    return registry


def build_and_save_registry(
    source="local",
    models_dir=MODELS_DIR,
    metrics_dir=METRICS_DIR,
    registry_path=REGISTRY_PATH,
):
    registry = build_registry(source=source, models_dir=models_dir, metrics_dir=metrics_dir)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as file_obj:
        json.dump(registry, file_obj, indent=2)

    summary = {
        "success": True,
        "registry_path": str(registry_path.as_posix()),
        "registered_models": len(registry),
    }
    print(f"Saved model registry to {registry_path}")
    print(f"Registered {len(registry)} models.")
    return summary


def main():
    summary = build_and_save_registry(source="local")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
