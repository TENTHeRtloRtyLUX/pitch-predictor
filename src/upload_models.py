import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from build_model_registry import build_registry


load_dotenv()

MODEL_REPO_ID = "rkhosla/pitch-predictor"
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
HF_REGISTRY_PATH = MODELS_DIR / "model_registry_hf.json"
TENDENCY_FILES = [
    "pitcher_overall_tendencies.csv",
    "pitcher_hand_tendencies.csv",
    "pitcher_count_tendencies.csv",
]


def require_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Missing HF_TOKEN in environment.")
    login(token=token)


def build_hf_registry():
    registry = build_registry(source="huggingface")
    with open(HF_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    return registry


def iter_upload_files(registry):
    files = []

    for entry in registry:
        files.extend(
            [
                MODELS_DIR / entry["model_filename"],
                MODELS_DIR / entry["label_encoder_filename"],
                MODELS_DIR / entry["feature_columns_filename"],
                METRICS_DIR / entry["metrics_filename"],
            ]
        )

    files.append(HF_REGISTRY_PATH)

    for filename in TENDENCY_FILES:
        local_path = Path("data") / filename
        if local_path.exists():
            files.append(local_path)

    seen = set()
    for file_path in files:
        if file_path.exists() and file_path not in seen:
            seen.add(file_path)
            yield file_path


def upload_files():
    require_token()

    api = HfApi()
    api.create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)

    registry = build_hf_registry()
    upload_files_list = list(iter_upload_files(registry))

    for file_path in upload_files_list:
        path_in_repo = file_path.name
        if file_path == HF_REGISTRY_PATH:
            path_in_repo = "model_registry.json"

        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=MODEL_REPO_ID,
            repo_type="model",
        )
        print(f"Uploaded {file_path}")

    print(f"Uploaded {len(upload_files_list)} files to {MODEL_REPO_ID}.")


if __name__ == "__main__":
    upload_files()
