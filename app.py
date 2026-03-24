import streamlit as st
import pandas as pd
import joblib
import sys
import time
from datetime import date

sys.path.append("src")
from mlb_api import get_games_on_date, get_live_game_state, PITCH_NAMES

model = joblib.load("models/full_xgb_model.pkl")
le_pitch = joblib.load("models/full_label_encoder.pkl")
feature_columns = joblib.load("models/full_feature_columns.pkl")

# Load pitcher data
pitcher_df = pd.read_csv("data/clean_full_pitches.csv")
pitcher_list = sorted(pitcher_df["pitcher_name"].unique().tolist())
pitcher_handedness = pitcher_df.groupby("pitcher_name")["p_throws"].first().to_dict()

def predict_pitch(pitcher, p_throws, stand, balls, strikes, outs, inning, on_1b, on_2b, on_3b, prev_pitch):
    count = f"{balls}-{strikes}"
    
    input_dict = {
        "balls": balls,
        "strikes": strikes,
        "outs_when_up": outs,
        "inning": inning,
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "p_throws": p_throws,
        "stand": stand,
        "pitcher_name": pitcher,
        "prev_pitch": prev_pitch,
        "count": count,
    }
    
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    pitch_name = le_pitch.inverse_transform([pred])[0]
    
    proba_df = pd.DataFrame({
    "Pitch Type": [PITCH_NAMES.get(c, c) for c in le_pitch.classes_],
    "Probability": [round(p * 100, 1) for p in proba]
}).sort_values("Probability", ascending=False)
    
    return pitch_name, proba_df

# ---- MAIN UI ----
st.title("⚾ MLB Pitch Predictor")

tab1, tab2 = st.tabs(["🔴 Live Games", "🎮 Manual Setup"])

# ---- TAB 1: LIVE GAMES ----
with tab1:
    st.header("Live Game Predictions")
    
    today = str(date.today())
    games = get_games_on_date(today)
    
    if not games:
        st.warning("No games found today.")
    else:
        game_labels = [f"{g['away_team']} @ {g['home_team']} ({g['status']})" for g in games]
        selected_game = st.selectbox("Select a game", game_labels)
        game_idx = game_labels.index(selected_game)
        game_id = games[game_idx]["game_id"]
        game_status = games[game_idx]["status"]
        
        auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=False)
        
        if st.button("Load Game") or auto_refresh:
            state = get_live_game_state(game_id)
            
            if not state:
                st.warning("No live data available for this game yet.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Pitcher", state["pitcher"])
                col2.metric("Batter", state["batter"])
                col3.metric("Inning", f"{'Top' if state['inning_half'] == 'top' else 'Bot'} {state['inning']}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Count", f"{state['balls']}-{state['strikes']}")
                col5.metric("Outs", state["outs"])
                col6.metric("Bases", f"{'1' if state['on_1b'] else '-'} {'2' if state['on_2b'] else '-'} {'3' if state['on_3b'] else '-'}")
                
                if state["recent_pitches"]:
                    st.subheader("Current At-Bat")
                    recent_df = pd.DataFrame(state["recent_pitches"])
                    st.dataframe(recent_df, use_container_width=True)
                
                st.subheader("Next Pitch Prediction")
                pitch_name, proba_df = predict_pitch(
                    state["pitcher"],
                    state["p_throws"],
                    state["stand"],
                    state["balls"],
                    state["strikes"],
                    state["outs"],
                    state["inning"],
                    state["on_1b"],
                    state["on_2b"],
                    state["on_3b"],
                    state["prev_pitch"]
                )
                
                st.success(f"Predicted Next Pitch: **{PITCH_NAMES.get(pitch_name, pitch_name)}**")
                st.dataframe(proba_df, use_container_width=True)
                
                if auto_refresh:
                    time.sleep(30)
                    st.rerun()

# ---- TAB 2: MANUAL SETUP ----
with tab2:
    st.header("Manual Situation Setup")
    
    pitcher = st.selectbox("Pitcher", pitcher_list)
    p_throws = pitcher_handedness.get(pitcher, "R")
    st.write(f"Pitcher throws: **{p_throws}**")
    
    stand = st.selectbox("Batter Stands", ["R", "L"])
    balls = st.slider("Balls", 0, 3, 0)
    strikes = st.slider("Strikes", 0, 2, 0)
    outs = st.slider("Outs", 0, 2, 0)
    inning = st.slider("Inning", 1, 9, 1)
    on_1b = st.checkbox("Runner on 1st")
    on_2b = st.checkbox("Runner on 2nd")
    on_3b = st.checkbox("Runner on 3rd")
    prev_pitch_options = {v: k for k, v in PITCH_NAMES.items()}
    prev_pitch_label = st.selectbox("Previous Pitch", list(prev_pitch_options.keys()))
    prev_pitch = prev_pitch_options[prev_pitch_label]
    
    if st.button("Predict Next Pitch"):
        pitch_name, proba_df = predict_pitch(
            pitcher, p_throws, stand, balls, strikes,
            outs, inning, int(on_1b), int(on_2b), int(on_3b), prev_pitch
        )
        st.success(f"Predicted Next Pitch: **{PITCH_NAMES.get(pitch_name, pitch_name)}**")
        st.dataframe(proba_df, use_container_width=True)