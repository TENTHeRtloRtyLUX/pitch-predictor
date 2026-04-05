import json
from pathlib import Path


def save_bundle_metadata(
    model_name,
    model_type,
    model_filename,
    label_encoder_filename,
    metrics_filename,
    preprocessor_filename=None,
    feature_columns_filename=None,
    output_dir="models",
    extra_metadata=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    payload = {
        "name": model_name,
        "model_type": model_type,
        "model_filename": model_filename,
        "label_encoder_filename": label_encoder_filename,
        "metrics_filename": metrics_filename,
    }
    if preprocessor_filename:
        payload["preprocessor_filename"] = preprocessor_filename
    if feature_columns_filename:
        payload["feature_columns_filename"] = feature_columns_filename
    if extra_metadata:
        payload.update(extra_metadata)

    bundle_path = output_path / f"{model_name}_bundle.json"
    with open(bundle_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    return bundle_path


def load_bundle_metadata(bundle_path):
    with open(bundle_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
