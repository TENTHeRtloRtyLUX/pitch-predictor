# MLB Pitch Predictor

A Streamlit app and retraining pipeline for predicting the next MLB pitch from game state, pitcher tendencies, and within-at-bat sequence context.

Live app: [pitch-predictor.streamlit.app](https://pitch-predictor.streamlit.app)

## Overview

The project now supports:

- incremental MLB ingest into Supabase with a persisted overlap watermark
- weekly retraining orchestration with guardrails and staged promotion
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
| `src/build_model_registry.py` | Builds the registry from saved bundle metadata and metrics |
| `src/upload_models.py` | Uploads model bundles, metrics, registry, and tendency files to Hugging Face |
| `src/run_weekly_retrain.py` | Weekly orchestration entrypoint with staging, guardrails, and persistent run state |

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
python src/training_data_pipeline.py
python src/train_tabular_models.py
python src/train_sequence_models.py --model-type lstm
python src/train_sequence_models.py --model-type transformer
python src/build_model_registry.py
python src/upload_models.py
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

## Notes

- Local and deployed registries can diverge if uploads are skipped.
- Sequence models depend on TensorFlow being installed in the environment.
- The app still falls back to a single legacy XGBoost bundle if no registry is available.
