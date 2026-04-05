import json
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from supabase import create_client

sys.path.append("src")
from mlb_api import PITCH_NAMES, get_games_on_date, get_live_game_state


MODEL_REPO_ID = "rkhosla/pitch-predictor"
LOCAL_REGISTRY_PATH = Path("models/model_registry.json")
LOCAL_DATA_DIR = Path("data")

supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_resource
def load_model_registry():
    if LOCAL_REGISTRY_PATH.exists():
        with open(LOCAL_REGISTRY_PATH, "r", encoding="utf-8") as f:
            local_registry = json.load(f)

        local_registry_is_usable = True
        for entry in local_registry:
            if entry.get("source") == "huggingface":
                continue

            required_paths = [
                entry.get("model_path"),
                entry.get("label_encoder_path"),
                entry.get("feature_columns_path"),
            ]
            if not all(required_paths) or not all(Path(path).exists() for path in required_paths):
                local_registry_is_usable = False
                break

        if local_registry_is_usable:
            return local_registry

    try:
        registry_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="model_registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass

    return [
        {
            "name": "xgboost_v2",
            "description": "Fallback deployed XGBoost model.",
            "source": "huggingface",
            "model_filename": "full_xgb_v2_model.pkl",
            "label_encoder_filename": "full_v2_label_encoder.pkl",
            "feature_columns_filename": "full_v2_feature_columns.pkl",
            "active": True,
        }
    ]


@st.cache_resource
def load_model_artifacts(model_name, model_path, label_encoder_path, feature_columns_path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    feature_artifact = joblib.load(feature_columns_path)
    return model, label_encoder, feature_artifact


@st.cache_resource
def load_registry_models():
    registry = [entry for entry in load_model_registry() if entry.get("active", True)]
    loaded_models = {}

    for entry in registry:
        if entry.get("source") == "huggingface":
            model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=entry["model_filename"])
            label_encoder_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=entry["label_encoder_filename"])
            feature_columns_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=entry["feature_columns_filename"])
        else:
            model_path = entry["model_path"]
            label_encoder_path = entry["label_encoder_path"]
            feature_columns_path = entry["feature_columns_path"]

        model, label_encoder, feature_artifact = load_model_artifacts(
            entry["name"],
            model_path,
            label_encoder_path,
            feature_columns_path,
        )

        loaded_models[entry["name"]] = {
            "meta": entry,
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": feature_artifact,
        }

    return loaded_models


MODELS = load_registry_models()


def load_tendency_table(filename):
    local_path = LOCAL_DATA_DIR / filename
    if local_path.exists():
        return pd.read_csv(local_path)

    return pd.read_csv(hf_hub_download(repo_id=MODEL_REPO_ID, filename=filename))


overall_tendencies = load_tendency_table("pitcher_overall_tendencies.csv")
hand_tendencies = load_tendency_table("pitcher_hand_tendencies.csv")
count_tendencies = load_tendency_table("pitcher_count_tendencies.csv")


@st.cache_resource
def load_pitcher_data():
    all_pitchers = []
    page = 0
    page_size = 1000

    while True:
        response = supabase.table("pitchers").select("name, throws").range(
            page * page_size, (page + 1) * page_size - 1
        ).execute()

        if not response.data:
            break

        all_pitchers.extend(response.data)
        page += 1

    if not all_pitchers:
        return [], {}

    pitcher_df = pd.DataFrame(all_pitchers)
    if "name" not in pitcher_df.columns or "throws" not in pitcher_df.columns:
        return [], {}

    pitcher_list = sorted(pitcher_df["name"].tolist())
    pitcher_handedness = dict(zip(pitcher_df["name"], pitcher_df["throws"]))
    return pitcher_list, pitcher_handedness


pitcher_list, pitcher_handedness = load_pitcher_data()


MODEL_LABELS = {model_name: model_name.replace("_", " ").title() for model_name in MODELS}
MODEL_OPTIONS = list(MODELS.keys())
DEFAULT_MODELS = MODEL_OPTIONS[: min(3, len(MODEL_OPTIONS))]


def build_model_accuracy_table():
    rows = []
    for model_name, model_bundle in MODELS.items():
        rows.append(
            {
                "Model": MODEL_LABELS[model_name],
                "Accuracy": model_bundle["meta"].get("accuracy"),
                "Description": model_bundle["meta"].get("description", ""),
            }
        )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard

    return leaderboard.sort_values("Accuracy", ascending=False, na_position="last").reset_index(drop=True)


def normalize_prev_pitch(prev_pitch):
    if prev_pitch in PITCH_NAMES:
        return prev_pitch

    reverse_names = {name: code for code, name in PITCH_NAMES.items()}
    return reverse_names.get(prev_pitch, "FF")


def build_prediction_features(
    pitcher,
    p_throws,
    stand,
    balls,
    strikes,
    outs,
    inning,
    on_1b,
    on_2b,
    on_3b,
    prev_pitch,
    score_diff,
):
    count = f"{balls}-{strikes}"

    input_dict = {
        "balls": balls,
        "strikes": strikes,
        "outs": outs,
        "inning": inning,
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "p_throws": p_throws,
        "stand": stand,
        "pitcher_name": pitcher,
        "prev_pitch": normalize_prev_pitch(prev_pitch),
        "count": count,
        "score_diff": score_diff,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.merge(overall_tendencies, on="pitcher_name", how="left")
    input_df = input_df.merge(hand_tendencies, on=["pitcher_name", "stand"], how="left")
    input_df = input_df.merge(count_tendencies, on=["pitcher_name", "count"], how="left")

    tendency_cols = [c for c in input_df.columns if "_pct" in c]
    input_df[tendency_cols] = input_df[tendency_cols].fillna(0)
    return input_df


def predict_with_model(model_name, features_df):
    model_bundle = MODELS[model_name]
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    feature_artifact = model_bundle["feature_columns"]

    if hasattr(feature_artifact, "transform"):
        model_input = feature_artifact.transform(features_df)
        if model_name in {"random_forest", "catboost"}:
            model_input = model_input.toarray()
    else:
        input_df = pd.get_dummies(features_df)
        model_input = input_df.reindex(columns=feature_artifact, fill_value=0)

    pred = model.predict(model_input)[0]
    pitch_code = label_encoder.inverse_transform([pred])[0]

    proba_df = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(model_input)[0]
        proba_df = pd.DataFrame(
            {
                "Pitch Type": [PITCH_NAMES.get(c, c) for c in label_encoder.classes_],
                "Probability": [round(p * 100, 1) for p in proba],
            }
        ).sort_values("Probability", ascending=False)

    return pitch_code, proba_df


def predict_pitch(selected_models, **kwargs):
    features_df = build_prediction_features(**kwargs)
    results = {}

    for model_name in selected_models:
        pitch_code, proba_df = predict_with_model(model_name, features_df)
        results[model_name] = {
            "pitch_code": pitch_code,
            "probabilities": proba_df,
        }

    return results


def render_prediction_results(results):
    if len(results) == 1:
        model_name, result = next(iter(results.items()))
        st.success(
            f"{MODEL_LABELS[model_name]}: **{PITCH_NAMES.get(result['pitch_code'], result['pitch_code'])}**"
        )
        if result["probabilities"] is not None:
            st.dataframe(result["probabilities"], use_container_width=True)
        return

    summary_rows = []
    for model_name, result in results.items():
        summary_rows.append(
            {
                "Model": MODEL_LABELS[model_name],
                "Accuracy": MODELS[model_name]["meta"].get("accuracy"),
                "Predicted Pitch": PITCH_NAMES.get(result["pitch_code"], result["pitch_code"]),
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    for model_name, result in results.items():
        if result["probabilities"] is not None:
            st.subheader(f"{MODEL_LABELS[model_name]} probabilities")
            st.dataframe(result["probabilities"], use_container_width=True)


st.title("MLB Pitch Predictor")
if not MODELS:
    st.error("No models are available. Build the model registry or upload model artifacts first.")
    st.stop()

st.caption("Most recent saved evaluation accuracy for each available model.")
accuracy_table = build_model_accuracy_table()
if not accuracy_table.empty:
    st.dataframe(
        accuracy_table,
        use_container_width=True,
        hide_index=True,
        column_config={"Accuracy": st.column_config.NumberColumn(format="%.3f")},
    )

tab1, tab2 = st.tabs(["Live Games", "Manual Setup"])

with tab1:
    st.header("Live Game Predictions")

    live_selected_models = st.multiselect(
        "Models to run",
        MODEL_OPTIONS,
        default=DEFAULT_MODELS,
        key="live_models",
    )

    today = st.date_input("Game date", value=date.today()).strftime("%Y-%m-%d")
    games = get_games_on_date(today)

    if not games:
        st.warning("No games found today.")
    elif not live_selected_models:
        st.info("Select at least one model to generate predictions.")
    else:
        game_labels = [f"{g['away_team']} @ {g['home_team']} ({g['status']})" for g in games]
        selected_game = st.selectbox("Select a game", game_labels)
        game_idx = game_labels.index(selected_game)
        game_id = games[game_idx]["game_id"]

        auto_refresh = st.toggle("Auto-refresh every 10 seconds", value=False)

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

                col7, col8, col9 = st.columns(3)
                col7.metric("Home Score", state["home_score"])
                col8.metric("Away Score", state["away_score"])
                col9.metric("Score Diff", f"{'+' if state['score_diff'] > 0 else ''}{state['score_diff']}")

                if state["recent_pitches"]:
                    st.subheader("Current At-Bat")
                    recent_df = pd.DataFrame(state["recent_pitches"])
                    st.dataframe(recent_df, use_container_width=True)

                st.subheader("Next Pitch Prediction")
                results = predict_pitch(
                    live_selected_models,
                    pitcher=state["pitcher"],
                    p_throws=state["p_throws"],
                    stand=state["stand"],
                    balls=state["balls"],
                    strikes=state["strikes"],
                    outs=state["outs"],
                    inning=state["inning"],
                    on_1b=state["on_1b"],
                    on_2b=state["on_2b"],
                    on_3b=state["on_3b"],
                    prev_pitch=state["prev_pitch"],
                    score_diff=state["score_diff"],
                )
                render_prediction_results(results)

                if auto_refresh:
                    time.sleep(10)
                    st.rerun()

with tab2:
    st.header("Manual Situation Setup")

    manual_selected_models = st.multiselect(
        "Models to run",
        MODEL_OPTIONS,
        default=DEFAULT_MODELS,
        key="manual_models",
    )

    if not pitcher_list:
        st.warning("No pitcher data available from Supabase.")
    else:
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
        prev_pitch_options = {PITCH_NAMES.get(code, code): code for code in PITCH_NAMES}
        prev_pitch_label = st.selectbox("Previous Pitch", list(prev_pitch_options.keys()))
        prev_pitch = prev_pitch_options[prev_pitch_label]
        score_diff = st.slider("Score Differential (your team)", -10, 10, 0)

        if st.button("Predict Next Pitch"):
            if not manual_selected_models:
                st.info("Select at least one model to generate predictions.")
            else:
                results = predict_pitch(
                    manual_selected_models,
                    pitcher=pitcher,
                    p_throws=p_throws,
                    stand=stand,
                    balls=balls,
                    strikes=strikes,
                    outs=outs,
                    inning=inning,
                    on_1b=int(on_1b),
                    on_2b=int(on_2b),
                    on_3b=int(on_3b),
                    prev_pitch=prev_pitch,
                    score_diff=score_diff,
                )
                render_prediction_results(results)
