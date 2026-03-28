# ⚾ MLB Pitch Predictor

A machine learning app that predicts what pitch an MLB pitcher will throw next based on game situation, count, pitcher tendencies, and batter matchup.

**🔴 Live App:** [pitch-predictor.streamlit.app](https://pitch-predictor.streamlit.app)

## Features

- **Live Game Mode** - Select any MLB game and get real-time pitch predictions
- **Manual Setup Mode** - Configure any game situation manually to explore predictions
- **XGBoost Model** - Trained on 478,000+ pitches from the 2023 MLB season
- **9 Pitch Types** - Predicts fastballs, sliders, curveballs, changeups, and more

## Tech Stack

- **Frontend:** Streamlit
- **ML Model:** XGBoost (~39% accuracy across 9 pitch types)
- **Data:** MLB Stats API (live games), Supabase (pitcher data)
- **Hosting:** Streamlit Cloud, HuggingFace (model files)

## Project Structure

### Active (Runtime)
| File | Description |
|------|-------------|
| `app.py` | Main Streamlit web application with live game and manual prediction modes |
| `src/mlb_api.py` | Fetches live game data from MLB Stats API |
| `src/load_to_supabase.py` | Loads pitcher and pitch data into Supabase database |

### Pipeline (Data & Training)
| File | Description |
|------|-------------|
| `src/fetch_data.py` | Downloads raw Statcast pitch data from Baseball Savant |
| `src/prepare_full_data.py` | Full data preparation including tendency features |
| `src/build_pitcher_tendencies.py` | Generates pitcher tendency CSV files from pitch data |
| `src/merge_tendencies.py` | Merges tendency data with pitch records |
| `src/train_full_xgb_v2.py` | Trains the XGBoost v2 model (current deployed model) |
| `src/upload_models.py` | Uploads trained models to HuggingFace Hub |

### Legacy (Experimental/Outdated)
| File | Description |
|------|-------------|
| `src/prepare_data.py` | *[Legacy]* Basic data cleaning, replaced by prepare_full_data.py |
| `src/train_model.py` | *[Legacy]* Original logistic regression model |
| `src/train_rf.py` | *[Legacy]* Random Forest experiment |
| `src/train_xgb.py` | *[Legacy]* Basic XGBoost experiment |
| `src/train_full_xgb.py` | *[Legacy]* XGBoost v1, replaced by v2 |
| `src/train_lstm.py` | *[Legacy]* LSTM sequence model experiment |
| `src/train_hybrid.py` | *[Legacy]* Hybrid XGBoost + LSTM experiment |
| `src/tune_model.py` | *[Legacy]* Hyperparameter tuning script |
| `src/explore_data.py` | *[Legacy]* Data exploration script |

## Local Development

1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file with your Supabase credentials
6. Run: `streamlit run app.py`

## Model Training

To retrain the model with new data:
1. `python src/fetch_data.py` — pulls Statcast data
2. `python src/prepare_full_data.py` — cleans and prepares data with tendencies
3. `python src/train_full_xgb_v2.py` — trains the XGBoost model
4. `python src/upload_models.py` — uploads model to HuggingFace

## Roadmap

- [x] XGBoost model with pitcher tendencies
- [x] Live MLB game integration
- [x] Streamlit Cloud deployment
- [x] Supabase database integration
- [ ] Auto-update pipeline (GitHub Actions)
- [ ] Improve model accuracy (more features, more data)
- [ ] LSTM sequence modeling for pitch sequences