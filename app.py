import streamlit as st
import pandas as pd
import joblib
import sys
sys.path.append("src")
from mlb_api import get_games_on_date, get_pitches_from_game
from datetime import date

model = joblib.load("models/full_xgb_model.pkl")
le_pitch = joblib.load("models/full_label_encoder.pkl")
feature_columns = joblib.load("models/full_feature_columns.pkl")

st.title("MLB Pitch Predictor")
st.write("Enter the current game situation to predict the next pitch.")

pitcher_df = pd.read_csv("data/clean_full_pitches.csv")
pitcher_list = sorted(pitcher_df["pitcher_name"].unique().tolist())

pitcher = st.selectbox("Pitcher", pitcher_list)

pitcher_handedness = pitcher_df.groupby("pitcher_name")["p_throws"].first().to_dict()

p_throws = pitcher_handedness.get(pitcher, "R")
st.write(f"Pitcher throws: **{p_throws}**")

stand = st.selectbox("Batter Stands", ["R", "L"])
balls = st.slider("Balls", 0, 3, 0)
strikes = st.slider("Strikes", 0, 2, 0)
outs = st.slider("Outs", 0, 2, 0)
innings = st.slider("Inning", 1, 9, 1)
on_1b = st.checkbox("Runner on 1st")
on_2b = st.checkbox("Runner on 2nd")
on_3b = st.checkbox("Runner on 3rd")
prev_pitch = st.selectbox("Previous Pitch", ["FF", "SL", "CU", "CH", "FC", "SI", "FS", "KC", "ST"])


if st.button("Predict Next Pitch"):
    input_dict = {
        "balls": balls,
        "strikes": strikes,
        "outs_when_up": outs,
        "inning": innings,
        "on_1b": int(on_1b),
        "on_2b": int(on_2b),
        "on_3b": int(on_3b),
        "p_throws": p_throws,
        "stand": stand,
        "pitcher_name": pitcher,
        "prev_pitch": prev_pitch,
        "count": f"{balls}-{strikes}",
    }

    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    pitch_name = le_pitch.inverse_transform([pred])[0]
    confidence = round(max(proba) * 100, 1)

    st.success(f"Predicted Pitch: **{pitch_name}**")
    st.write(f"Confidence: {confidence}%")

    proba_df = pd.DataFrame({
        "Pitch Type": le_pitch.classes_,
        "Probability": [round(p * 100, 1) for p in proba]
    }).sort_values("Probability", ascending=False)

    st.dataframe(proba_df)

    st.divider()
st.header("🔴 Live Game Mode")

selected_date = st.date_input("Select a date", value=date.today())

if st.button("Load Games"):
    games = get_games_on_date(str(selected_date))
    
    if not games:
        st.warning("No games found for this date.")
    else:
        st.session_state["games"] = games

if "games" in st.session_state:
    games = st.session_state["games"]
    game_labels = [f"{g['away_team']} @ {g['home_team']} ({g['status']})" for g in games]
    selected_game = st.selectbox("Select a game", game_labels)
    
    game_idx = game_labels.index(selected_game)
    game_id = games[game_idx]["game_id"]
    
    if st.button("Load Pitches"):
        pitches = get_pitches_from_game(game_id)
        st.session_state["pitches"] = pitches
        st.success(f"Loaded {len(pitches)} pitches")
        st.dataframe(pitches)