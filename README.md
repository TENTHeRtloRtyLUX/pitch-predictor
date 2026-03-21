# Pitch Predictor

## What is this?
A machine learning project that predicts the type of pitch an MLB pitcher 
will throw next, based on game situation, batter/pitcher matchup, and more.

## Setup
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install pybaseball pandas scikit-learn`

## How to Run
Run scripts in this order from the project root:
1. `python src/fetch_data.py` — pulls raw Statcast data from Baseball Savant
2. `python src/prepare_data.py` — cleans and encodes the data
3. `python src/train_model.py` — trains and evaluates the model

## Current Status
Phase 1: Basic pipeline with logistic regression baseline (~38% accuracy, 9 pitch types). 
Added Random Forest w/ prev pitch and count features

## Roadmap
- Phase 2: Add batter vs pitcher matchup features, upgrade to XGBoost
- Phase 3: Add LSTM sequence modeling for within at-bat pitch history
- Phase 4: Live MLB Stats API and weather integration
- Phase 5: Deploy to GitHub Pages with a FastAPI backend