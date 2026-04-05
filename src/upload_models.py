import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from build_model_registry import build_and_save_registry, build_registry
from training_data_pipeline import TENDENCY_FILENAMES


load_dotenv()

MODEL_REPO_ID = "rkhosla/pitch-predictor"
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
DATA_DIR = Path("data")
HF_REGISTRY_PATH = MODELS_DIR / "model_registry_hf.json"


def require_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Missing HF_TOKEN in environment.")
    login(token=token)


def build_hf_registry(models_dir=MODELS_DIR, metrics_dir=METRICS_DIR, registry_path=HF_REGISTRY_PATH):
    summary = build_and_save_registry(
        source="huggingface",
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        registry_path=registry_path,
    )
    return summary


def iter_upload_files(registry, models_dir=MODELS_DIR, metrics_dir=METRICS_DIR, data_dir=DATA_DIR):
    files = []

    for entry in registry:
        files.append(models_dir / entry["model_filename"])
        files.append(models_dir / entry["label_encoder_filename"])
        files.append(metrics_dir / entry["metrics_filename"])

        if entry.get("feature_columns_filename"):
            files.append(models_dir / entry["feature_columns_filename"])
        if entry.get("preprocessor_filename"):
            files.append(models_dir / entry["preprocessor_filename"])

    files.append(HF_REGISTRY_PATH)

    for filename in TENDENCY_FILENAMES.values():
        local_path = Path(data_dir) / filename
        if local_path.exists():
            files.append(local_path)

    seen = set()
    for file_path in files:
        if file_path.exists() and file_path not in seen:
            seen.add(file_path)
            yield file_path


def upload_files(
    models_dir=MODELS_DIR,
    metrics_dir=METRICS_DIR,
    data_dir=DATA_DIR,
    allow_upload=True,
):
    registry = build_registry(source="huggingface", models_dir=models_dir, metrics_dir=metrics_dir)
    with open(HF_REGISTRY_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(registry, file_obj, indent=2)

    upload_files_list = list(iter_upload_files(registry, models_dir=models_dir, metrics_dir=metrics_dir, data_dir=data_dir))
    summary = {
        "success": True,
        "repo_id": MODEL_REPO_ID,
        "files_prepared": [str(path.as_posix()) for path in upload_files_list],
        "upload_enabled": allow_upload,
    }

    if not allow_upload:
        print(json.dumps(summary, indent=2))
        return summary

    require_token()

    api = HfApi()
    api.create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)

    for file_path in upload_files_list:
        path_in_repo = "model_registry.json" if file_path == HF_REGISTRY_PATH else file_path.name
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )
        print(f"Uploaded {file_path}")

    summary["files_uploaded"] = len(upload_files_list)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    upload_files()


if __name__ == "__main__":
    main()
