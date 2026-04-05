# MLB Pitch Predictor

A Streamlit app that predicts what pitch an MLB pitcher will throw next based on game state, count, pitcher tendencies, and matchup context.

Live app: [pitch-predictor.streamlit.app](https://pitch-predictor.streamlit.app)

## Overview

The app now supports multiple trained models side by side. At the top of the app, a model leaderboard shows the saved evaluation accuracy for every model currently available in the registry. Users can then select any subset of those models and compare their predictions in both live-game and manual modes.

The project has also moved away from a local CSV-first workflow and toward a Supabase-backed training pipeline:

- MLB Stats API is the source for historical and live pitch data
- Supabase stores the compact retained pitch-level training dataset
- feature preparation and tendency generation now run in memory
- the app can load either local model bundles or uploaded Hugging Face artifacts

## Features

- Live game mode for current MLB matchups
- Manual setup mode for custom situations
- Multi-model pitch prediction comparison
- Model accuracy leaderboard shown directly in the app
- Pitch-type probability outputs, not just a single class prediction
- Pitcher tendency features by overall usage, batter handedness, and count
- Supabase-backed historical pitch data for centralized retraining

## Tech Stack

- Frontend: Streamlit
- Data ingestion: MLB Stats API
- Data storage: Supabase
- Model hosting: Hugging Face
- ML libraries: scikit-learn, XGBoost, LightGBM, CatBoost

## Project Structure

### Active Runtime
| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application with live and manual prediction flows |
| `src/mlb_api.py` | Fetches schedules, live game state, and historical pitch logs from the MLB Stats API |

### Active Data Pipeline
| File | Description |
|------|-------------|
| `src/load_to_supabase.py` | Loads compact pitch-level training data into Supabase |
| `src/supabase_data_loader.py` | Reads retained training data back from Supabase for model training |
| `src/prepare_full_data.py` | Reusable in-memory pitch preparation functions |
| `src/build_pitcher_tendencies.py` | Builds pitcher tendency feature tables in memory |
| `src/merge_tendencies.py` | Merges tendency features into the prepared pitch dataset |
| `src/training_data_pipeline.py` | Shared training-data entrypoint for all tabular models |

### Active Training and Packaging
| File | Description |
|------|-------------|
| `src/tabular_training.py` | Shared tabular feature preparation and train/test helpers |
| `src/train_tabular_models.py` | Trains the active tabular model suite |
| `src/build_model_registry.py` | Builds a registry of trained models and saved metrics |
| `src/upload_models.py` | Uploads model bundles, metrics, tendencies, and registry files to Hugging Face |

### Legacy

Older experiments and superseded scripts live under `src/legacy/`.

## Canonical Training Schema

The retained `pitches` dataset is intentionally compact. The active training pipeline expects:

- `id`
- `game_id`
- `season`
- `pitcher_name`
- `batter_name`
- `p_throws`
- `stand`
- `pitch_type`
- `balls`
- `strikes`
- `outs`
- `inning`
- `on_1b`
- `on_2b`
- `on_3b`
- `score_diff`
- `at_bat_number`
- `pitch_number`
- `prev_pitch`
- `count`

## Local Development

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` on Windows or `source venv/bin/activate` on macOS/Linux
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file for training scripts with:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `HF_TOKEN` when you want to upload artifacts
5. Create `.streamlit/secrets.toml` for the Streamlit app with:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
6. Run the app locally: `streamlit run app.py`

## Training Workflow

The current retraining flow is:

1. Update Supabase from the MLB Stats API with `python src/load_to_supabase.py`
2. Train the tabular models with `python src/train_tabular_models.py`
3. Rebuild the registry with `python src/build_model_registry.py`
4. Upload selected artifacts with `python src/upload_models.py`

The app will prefer:

- local `models/model_registry.json` and local model bundles when present
- otherwise a remote `model_registry.json` from Hugging Face
- otherwise a single fallback XGBoost v2 model

For tendency tables, the app prefers local `data/*.csv` files when available and falls back to Hugging Face otherwise.

## Active Models

The active tabular training pipeline supports:

- Logistic Regression
- SGDClassifier
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Calibrated XGBoost

Any trained model with a complete bundle and saved metrics can appear in the app and in the model leaderboard.

## Notes

- Accuracy shown in the app is the most recent saved evaluation accuracy from the latest training run.
- Local and deployed apps can show different model sets if the local registry and the remote Hugging Face registry are out of sync.
- Sequence models like LSTM and transformers are planned next once the shared data pipeline and comparison flow are stable.

## Roadmap

- [x] Live MLB game integration
- [x] Supabase-backed runtime data
- [x] Shared in-memory tabular training pipeline
- [x] Multi-model tabular training support
- [x] Model comparison UI in Streamlit
- [ ] Scheduled retraining and tendency refresh workflow
- [ ] Sequence modeling with LSTM
- [ ] Transformer-based pitch sequence modeling
