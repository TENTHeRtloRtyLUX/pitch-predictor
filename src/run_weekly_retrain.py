import argparse
import json
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from build_model_registry import build_and_save_registry
from load_to_supabase import run_incremental_ingest
from train_tabular_models import train_tabular_model_suite
from training_data_pipeline import refresh_tendency_files
from upload_models import upload_files


OUTPUT_DIR = Path("output")
STAGING_DIR = OUTPUT_DIR / "retrain_staging"
RUN_SUMMARY_PATH = OUTPUT_DIR / "weekly_retrain_summary.json"
RUN_STATE_PATH = OUTPUT_DIR / "weekly_retrain_state.json"
LOCK_PATH = OUTPUT_DIR / "weekly_retrain.lock"
PROD_MODELS_DIR = Path("models")
PROD_METRICS_DIR = Path("metrics")
PROD_DATA_DIR = Path("data")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    STAGING_DIR.mkdir(exist_ok=True)


def read_json(path, default=None):
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


@contextmanager
def fail_fast_lock():
    ensure_output_dirs()
    if LOCK_PATH.exists():
        raise RuntimeError(f"Another weekly retrain run appears active because {LOCK_PATH} already exists.")
    write_json(LOCK_PATH, {"acquired_utc": utc_now()})
    try:
        yield
    finally:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()


def run_step(step_name, fn, timeout_seconds, global_started_at, global_timeout_seconds, **kwargs):
    if time.monotonic() - global_started_at > global_timeout_seconds:
        raise TimeoutError(f"Global timeout exceeded before starting step '{step_name}'.")

    started_at = time.monotonic()
    result = fn(**kwargs)
    elapsed = time.monotonic() - started_at

    if elapsed > timeout_seconds:
        raise TimeoutError(
            f"Step '{step_name}' exceeded timeout budget of {timeout_seconds}s with elapsed {elapsed:.1f}s."
        )
    if time.monotonic() - global_started_at > global_timeout_seconds:
        raise TimeoutError(f"Global timeout exceeded after step '{step_name}'.")

    return {
        "step": step_name,
        "success": True,
        "elapsed_seconds": round(elapsed, 2),
        "result": result,
    }


def get_previous_best_accuracy(registry_path=PROD_MODELS_DIR / "model_registry.json"):
    registry = read_json(registry_path, default=[]) or []
    accuracies = [entry.get("accuracy") for entry in registry if entry.get("accuracy") is not None]
    return max(accuracies) if accuracies else None


def promote_staging_artifacts(staging_models_dir, staging_metrics_dir, staging_data_dir):
    PROD_MODELS_DIR.mkdir(exist_ok=True)
    PROD_METRICS_DIR.mkdir(exist_ok=True)
    PROD_DATA_DIR.mkdir(exist_ok=True)

    for file_path in Path(staging_models_dir).glob("*"):
        shutil.copy2(file_path, PROD_MODELS_DIR / file_path.name)
    for file_path in Path(staging_metrics_dir).glob("*"):
        shutil.copy2(file_path, PROD_METRICS_DIR / file_path.name)
    for file_path in Path(staging_data_dir).glob("*"):
        shutil.copy2(file_path, PROD_DATA_DIR / file_path.name)


def update_run_state(success, summary):
    current_state = read_json(RUN_STATE_PATH, default={}) or {}
    if success:
        current_state["last_successful_run"] = utc_now()
    else:
        current_state["last_failed_run"] = utc_now()
    current_state["retrain_summary"] = summary
    write_json(RUN_STATE_PATH, current_state)


def run_weekly_retrain(
    seasons=(2023, 2024, 2025),
    allow_upload=True,
    accuracy_drop_threshold=0.02,
    global_timeout_seconds=4 * 60 * 60,
):
    ensure_output_dirs()
    global_started_at = time.monotonic()

    staging_models_dir = STAGING_DIR / "models"
    staging_metrics_dir = STAGING_DIR / "metrics"
    staging_data_dir = STAGING_DIR / "data"
    staging_models_dir.mkdir(parents=True, exist_ok=True)
    staging_metrics_dir.mkdir(parents=True, exist_ok=True)
    staging_data_dir.mkdir(parents=True, exist_ok=True)

    previous_best_accuracy = get_previous_best_accuracy()
    step_summaries = []

    def get_step_result(step_name):
        for step_summary in step_summaries:
            if step_summary.get("step") == step_name:
                return step_summary["result"]
        raise KeyError(f"Step '{step_name}' was not recorded in the weekly retrain summary.")

    try:
        with fail_fast_lock():
            step_summaries.append(
                run_step(
                    "incremental_ingest",
                    run_incremental_ingest,
                    timeout_seconds=45 * 60,
                    global_started_at=global_started_at,
                    global_timeout_seconds=global_timeout_seconds,
                )
            )
            step_summaries.append(
                run_step(
                    "tendency_refresh",
                    refresh_tendency_files,
                    timeout_seconds=20 * 60,
                    global_started_at=global_started_at,
                    global_timeout_seconds=global_timeout_seconds,
                    seasons=list(seasons),
                    output_dir=staging_data_dir,
                )
            )
            step_summaries.append(
                run_step(
                    "tabular_training",
                    train_tabular_model_suite,
                    timeout_seconds=2 * 60 * 60,
                    global_started_at=global_started_at,
                    global_timeout_seconds=global_timeout_seconds,
                    seasons=list(seasons),
                    models_dir=staging_models_dir,
                    metrics_dir=staging_metrics_dir,
                )
            )
            step_summaries.append(
                run_step(
                    "registry_rebuild",
                    build_and_save_registry,
                    timeout_seconds=5 * 60,
                    global_started_at=global_started_at,
                    global_timeout_seconds=global_timeout_seconds,
                    source="local",
                    models_dir=staging_models_dir,
                    metrics_dir=staging_metrics_dir,
                    registry_path=staging_models_dir / "model_registry.json",
                )
            )

            new_best_accuracy = get_step_result("tabular_training")["best_accuracy"]
            if (
                previous_best_accuracy is not None
                and new_best_accuracy < previous_best_accuracy - accuracy_drop_threshold
            ):
                summary = {
                    "success": False,
                    "guardrail_blocked_upload": True,
                    "previous_best_accuracy": previous_best_accuracy,
                    "new_best_accuracy": new_best_accuracy,
                    "accuracy_drop_threshold": accuracy_drop_threshold,
                    "steps": step_summaries,
                }
                write_json(RUN_SUMMARY_PATH, summary)
                update_run_state(False, summary)
                return summary

            promote_staging_artifacts(staging_models_dir, staging_metrics_dir, staging_data_dir)
            build_and_save_registry(source="local")

            step_summaries.append(
                run_step(
                    "upload",
                    upload_files,
                    timeout_seconds=30 * 60,
                    global_started_at=global_started_at,
                    global_timeout_seconds=global_timeout_seconds,
                    allow_upload=allow_upload,
                )
            )

        summary = {
            "success": True,
            "previous_best_accuracy": previous_best_accuracy,
            "new_best_accuracy": get_step_result("tabular_training")["best_accuracy"],
            "upload_enabled": allow_upload,
            "steps": step_summaries,
        }
        write_json(RUN_SUMMARY_PATH, summary)
        update_run_state(True, summary)
        return summary
    except Exception as exc:
        summary = {
            "success": False,
            "error": str(exc),
            "previous_best_accuracy": previous_best_accuracy,
            "steps": step_summaries,
        }
        write_json(RUN_SUMMARY_PATH, summary)
        update_run_state(False, summary)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run the weekly pitch predictor retraining pipeline.")
    parser.add_argument("--no-upload", action="store_true", help="Build everything but skip Hugging Face upload.")
    parser.add_argument("--accuracy-drop-threshold", type=float, default=0.02)
    parser.add_argument("--global-timeout-seconds", type=int, default=4 * 60 * 60)
    args = parser.parse_args()

    summary = run_weekly_retrain(
        allow_upload=not args.no_upload,
        accuracy_drop_threshold=args.accuracy_drop_threshold,
        global_timeout_seconds=args.global_timeout_seconds,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
