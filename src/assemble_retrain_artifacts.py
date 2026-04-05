import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from build_model_registry import build_and_save_registry
from upload_models import upload_files


OUTPUT_DIR = Path("output")
ASSEMBLY_SUMMARY_PATH = OUTPUT_DIR / "assemble_retrain_summary.json"
RUN_STATE_PATH = OUTPUT_DIR / "weekly_retrain_state.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def get_previous_best_accuracy(registry_path):
    registry = read_json(registry_path, default=[]) or []
    accuracies = [entry.get("accuracy") for entry in registry if entry.get("accuracy") is not None]
    return max(accuracies) if accuracies else None


def copy_tree_contents(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.glob("*"):
        shutil.copy2(path, target_dir / path.name)


def update_run_state(summary):
    state = read_json(RUN_STATE_PATH, default={}) or {}
    if summary.get("success"):
        state["last_successful_run"] = utc_now()
    else:
        state["last_failed_run"] = utc_now()
    state["retrain_summary"] = summary
    write_json(RUN_STATE_PATH, state)


def assemble_retrain_artifacts(
    prep_dir,
    trained_models_dir,
    trained_metrics_dir,
    prod_models_dir="models",
    prod_metrics_dir="metrics",
    prod_data_dir="data",
    allow_upload=False,
    accuracy_drop_threshold=0.02,
    previous_registry_path="models/model_registry.json",
):
    prep_dir = Path(prep_dir)
    trained_models_dir = Path(trained_models_dir)
    trained_metrics_dir = Path(trained_metrics_dir)
    prod_models_dir = Path(prod_models_dir)
    prod_metrics_dir = Path(prod_metrics_dir)
    prod_data_dir = Path(prod_data_dir)
    previous_registry_path = Path(previous_registry_path)

    previous_best_accuracy = get_previous_best_accuracy(previous_registry_path)

    copy_tree_contents(prep_dir / "data", prod_data_dir)
    copy_tree_contents(trained_models_dir, prod_models_dir)
    copy_tree_contents(trained_metrics_dir, prod_metrics_dir)

    registry_summary = build_and_save_registry(
        source="local",
        models_dir=prod_models_dir,
        metrics_dir=prod_metrics_dir,
        registry_path=prod_models_dir / "model_registry.json",
    )
    registry = read_json(prod_models_dir / "model_registry.json", default=[]) or []
    new_accuracies = [entry.get("accuracy") for entry in registry if entry.get("accuracy") is not None]
    new_best_accuracy = max(new_accuracies) if new_accuracies else None

    if (
        previous_best_accuracy is not None
        and new_best_accuracy is not None
        and new_best_accuracy < previous_best_accuracy - accuracy_drop_threshold
    ):
        summary = {
            "success": False,
            "guardrail_blocked_upload": True,
            "previous_best_accuracy": previous_best_accuracy,
            "new_best_accuracy": new_best_accuracy,
            "accuracy_drop_threshold": accuracy_drop_threshold,
            "registry_summary": registry_summary,
            "upload_enabled": allow_upload,
        }
        write_json(ASSEMBLY_SUMMARY_PATH, summary)
        update_run_state(summary)
        return summary

    upload_summary = upload_files(
        models_dir=prod_models_dir,
        metrics_dir=prod_metrics_dir,
        data_dir=prod_data_dir,
        allow_upload=allow_upload,
    )

    summary = {
        "success": True,
        "previous_best_accuracy": previous_best_accuracy,
        "new_best_accuracy": new_best_accuracy,
        "accuracy_drop_threshold": accuracy_drop_threshold,
        "registry_summary": registry_summary,
        "upload_summary": upload_summary,
        "upload_enabled": allow_upload,
    }
    write_json(ASSEMBLY_SUMMARY_PATH, summary)
    update_run_state(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Assemble parallel retraining artifacts and optionally upload.")
    parser.add_argument("--prep-dir", default="output/retrain_shared")
    parser.add_argument("--trained-models-dir", default="output/collected_models/models")
    parser.add_argument("--trained-metrics-dir", default="output/collected_models/metrics")
    parser.add_argument("--prod-models-dir", default="models")
    parser.add_argument("--prod-metrics-dir", default="metrics")
    parser.add_argument("--prod-data-dir", default="data")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--accuracy-drop-threshold", type=float, default=0.02)
    args = parser.parse_args()

    summary = assemble_retrain_artifacts(
        prep_dir=args.prep_dir,
        trained_models_dir=args.trained_models_dir,
        trained_metrics_dir=args.trained_metrics_dir,
        prod_models_dir=args.prod_models_dir,
        prod_metrics_dir=args.prod_metrics_dir,
        prod_data_dir=args.prod_data_dir,
        allow_upload=args.upload,
        accuracy_drop_threshold=args.accuracy_drop_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
