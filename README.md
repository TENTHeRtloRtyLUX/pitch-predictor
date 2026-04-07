# MLB Pitch Predictor

A Streamlit app and retraining pipeline for predicting the next MLB pitch from game state, pitcher tendencies, and within-at-bat sequence context.

Live app: [pitch-predictor.streamlit.app](https://pitch-predictor.streamlit.app)

## Overview

The project now supports:

- incremental MLB ingest into Supabase with a persisted overlap watermark
- a weekly retraining workflow split into prep, parallel training, and assembly stages
- side-by-side tabular, LSTM, and transformer model bundles
- runtime model routing in the app based on `model_type`

## Tech Stack

- Frontend: Streamlit
- Data ingestion: MLB Stats API
- Data storage: Supabase
- Model hosting: Hugging Face
- ML libraries: scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow

## Project Structure

### Runtime
| File | Description |
|------|-------------|
| `app.py` | Streamlit application with tabular and sequence model inference routing |
| `src/mlb_api.py` | MLB Stats API client for schedule, live state, and pitch event retrieval |

### Data Pipeline
| File | Description |
|------|-------------|
| `src/load_to_supabase.py` | Incremental overlap ingest with watermark state, locking, dry-run mode, and run summaries |
| `src/supabase_data_loader.py` | Loads retained pitch rows from Supabase for training |
| `src/training_data_pipeline.py` | Shared tabular and sequence training-data builders plus tendency file refresh |
| `src/prepare_full_data.py` | Reusable pitch-cleaning helpers |

### Training and Packaging
| File | Description |
|------|-------------|
| `src/train_tabular_models.py` | Trains the tabular model suite and writes bundle metadata |
| `src/train_sequence_models.py` | Trains LSTM or transformer sequence models |
| `src/prepare_retraining_artifacts.py` | Builds the shared retraining dataframe and sequence dataset used by the workflow |
| `src/assemble_retrain_artifacts.py` | Promotes trained artifacts into production folders, rebuilds the registry, and optionally uploads |
| `src/build_model_registry.py` | Builds the registry from saved bundle metadata and metrics |
| `src/upload_models.py` | Uploads model bundles, metrics, registry, and tendency files to Hugging Face |
| `src/run_weekly_retrain.py` | Local orchestration entrypoint with sequential staging, guardrails, and persistent run state |

### Legacy
| File | Description |
|------|-------------|
| `src/legacy/fetch_data.py` | Superseded pitch export script retained for reference |
| `src/legacy/train_full_xgb_v2.py` | Superseded one-off XGBoost training script retained for reference |
| `src/legacy/train_full_xgb.py` | Earlier XGBoost training script retained for reference |
| `src/legacy/train_rf.py` | Earlier random forest training script retained for reference |
| `src/legacy/train_hybrid.py` | Older hybrid training workflow retained for reference |
| `src/legacy/train_lstm.py` | Older sequence training workflow retained for reference |
| `src/legacy/train_model.py` | Older generic training script retained for reference |
| `src/legacy/train_xgb.py` | Older XGBoost training script retained for reference |
| `src/legacy/tune_model.py` | Older tuning script retained for reference |

## Ingest Workflow

The primary ingest command is now incremental:

```bash
python src/load_to_supabase.py --dry-run
python src/load_to_supabase.py
```

Behavior:

- reads the last successful watermark
- computes `start_date = max(last_ingested_game_date - overlap_days, season_start)`
- fetches final games in that window
- upserts pitches by `pitch_uid`
- commits the watermark only after the full run succeeds
- writes the latest summary to `output/pitch_ingest_latest_summary.json`

If the `pipeline_state` table is unavailable in Supabase, the script falls back to `output/pitch_ingest_state.json`.

Recommended Supabase setup:

- unique constraint on `pitches.pitch_uid`
- optional `pipeline_state` table keyed by `pipeline_name`

## Retraining Workflow

Manual local workflow:

```bash
python src/load_to_supabase.py
python src/prepare_retraining_artifacts.py --seasons 2023,2024,2025 --output-dir output/retrain_shared --batch-size 250
python src/train_tabular_models.py --prepared-training-path output/retrain_shared/training_dataframe.joblib --models-dir output/collected_models/models --metrics-dir output/collected_models/metrics
python src/train_sequence_models.py --model-type lstm --prepared-sequence-path output/retrain_shared/sequence_dataset.joblib --models-dir output/collected_models/models --metrics-dir output/collected_models/metrics
python src/train_sequence_models.py --model-type transformer --prepared-sequence-path output/retrain_shared/sequence_dataset.joblib --models-dir output/collected_models/models --metrics-dir output/collected_models/metrics
python src/assemble_retrain_artifacts.py --prep-dir output/retrain_shared --trained-models-dir output/collected_models/models --trained-metrics-dir output/collected_models/metrics
```

Weekly orchestration:

```bash
python src/run_weekly_retrain.py --no-upload
python src/run_weekly_retrain.py
```

Guardrails in the weekly flow:

- fail-fast lock to prevent overlapping runs
- per-step and global timeout budgets
- staging directories so previous production bundles remain untouched until checks pass
- regression guard that blocks upload if the new best tabular accuracy drops past the threshold
- persistent run state in `output/weekly_retrain_state.json`
- latest run summary in `output/weekly_retrain_summary.json`

The GitHub Actions workflow uses the staged pipeline directly instead of the sequential local wrapper:

1. `src/load_to_supabase.py`
2. `src/prepare_retraining_artifacts.py`
3. parallel tabular training with `src/train_tabular_models.py`
4. parallel sequence training with `src/train_sequence_models.py`
5. `src/assemble_retrain_artifacts.py`

## Model Types

The registry now supports:

- `tabular`
- `lstm`
- `transformer`

Tabular models use the saved sparse feature preprocessor.

Sequence models use:

- prior pitches in the same at-bat as sequence input
- current pitch context as static features
- game-level train/test splitting

## GitHub Actions

The repo includes a weekly scheduled workflow and manual dispatch entrypoint in `.github/workflows/weekly_retrain.yml`.

Required secrets:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `HF_TOKEN`

Artifacts uploaded on every run:

- weekly retrain summary
- ingest summary
- failure logs when present

The workflow trains tabular and sequence models in parallel matrix jobs, then downloads the outputs into an assembly job that rebuilds the registry and optionally uploads the bundle set to Hugging Face.

## Notes

- Local and deployed registries can diverge if uploads are skipped.
- Sequence models depend on TensorFlow being installed in the environment.
- The app still falls back to a single legacy XGBoost bundle if no registry is available.
