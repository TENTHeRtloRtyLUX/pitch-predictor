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
from tensorflow.keras.models import load_model

sys.path.append("src")
from mlb_api import PITCH_NAMES, get_games_on_date, get_live_game_state
from training_data_pipeline import build_sequence_inference_inputs


MODEL_REPO_ID = "rkhosla/pitch-predictor"
LOCAL_REGISTRY_PATH = Path("models/model_registry.json")
LOCAL_DATA_DIR = Path("data")

def get_app_supabase_client():
    """Get Supabase client for Streamlit app using publishable key (respects RLS)."""
    url = st.secrets["SUPABASE_URL"]
    # Try modern key first, then fall back to legacy keys
    key = (
        st.secrets.get("SUPABASE_PUBLISHABLE_KEY") or
        st.secrets.get("SUPABASE_ANON_KEY") or
        st.secrets.get("SUPABASE_KEY")
    )
    if not key:
        raise KeyError(
            "Missing SUPABASE_PUBLISHABLE_KEY in Streamlit secrets. "
            "(Or legacy SUPABASE_ANON_KEY / SUPABASE_KEY for backward compatibility.)"
        )
    return create_client(url, key)


supabase = get_app_supabase_client()


@st.cache_resource
def load_model_registry():
    if LOCAL_REGISTRY_PATH.exists():
        with open(LOCAL_REGISTRY_PATH, "r", encoding="utf-8") as file_obj:
            local_registry = json.load(file_obj)

        local_registry_is_usable = True
        for entry in local_registry:
            if entry.get("source") == "huggingface":
                continue

            required_paths = [
                entry.get("model_path"),
                entry.get("label_encoder_path"),
            ]
            if entry.get("model_type") == "tabular":
                required_paths.append(entry.get("feature_columns_path"))
            else:
                required_paths.append(entry.get("preprocessor_path"))

            if not all(required_paths) or not all(Path(path).exists() for path in required_paths):
                local_registry_is_usable = False
                break

        if local_registry_is_usable:
            return local_registry

    try:
        registry_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="model_registry.json")
        with open(registry_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except Exception:
        pass

    return [
        {
            "name": "xgboost_v2",
            "description": "Fallback deployed XGBoost model.",
            "source": "huggingface",
            "model_type": "tabular",
            "model_filename": "full_xgb_v2_model.pkl",
            "label_encoder_filename": "full_v2_label_encoder.pkl",
            "feature_columns_filename": "full_v2_feature_columns.pkl",
            "active": True,
        }
    ]


def load_artifact(entry, filename_key, path_key):
    if entry.get("source") == "huggingface":
        return hf_hub_download(repo_id=MODEL_REPO_ID, filename=entry[filename_key])
    return entry[path_key]


@st.cache_resource
def load_model_artifacts(model_name, model_type, model_path, label_encoder_path, feature_path=None, preprocessor_path=None):
    if model_type == "tabular":
        model = joblib.load(model_path)
    else:
        model = load_model(model_path)
    label_encoder = joblib.load(label_encoder_path)
    feature_artifact = joblib.load(feature_path) if feature_path else None
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path else None
    return model, label_encoder, feature_artifact, preprocessor


@st.cache_resource
def load_registry_models():
    registry = [entry for entry in load_model_registry() if entry.get("active", True)]
    loaded_models = {}
    failed_models = []
    
    # Log file for diagnostics (helpful for Streamlit Cloud)
    log_file = Path("model_loading_log.txt")
    log_messages = []

    for entry in registry:
        try:
            model_path = load_artifact(entry, "model_filename", "model_path")
            label_encoder_path = load_artifact(entry, "label_encoder_filename", "label_encoder_path")
            feature_path = None
            preprocessor_path = None

            if entry.get("feature_columns_filename"):
                feature_path = load_artifact(entry, "feature_columns_filename", "feature_columns_path")
            if entry.get("preprocessor_filename"):
                preprocessor_path = load_artifact(entry, "preprocessor_filename", "preprocessor_path")

            model, label_encoder, feature_artifact, preprocessor = load_model_artifacts(
                entry["name"],
                entry.get("model_type", "tabular"),
                model_path,
                label_encoder_path,
                feature_path,
                preprocessor_path,
            )

            loaded_models[entry["name"]] = {
                "meta": entry,
                "model": model,
                "label_encoder": label_encoder,
                "feature_columns": feature_artifact,
                "preprocessor": preprocessor,
            }
            log_messages.append(f"✅ Loaded {entry['name']} ({entry.get('model_type', 'tabular')})")
        except Exception as e:
            failed_models.append((entry["name"], str(e)))
            log_messages.append(f"❌ Failed to load {entry['name']}: {type(e).__name__}: {str(e)[:100]}")

    # Write log file
    with open(log_file, "w") as f:
        f.write("\n".join(log_messages))
    
    # Print to console (visible in local terminal)
    print("\n".join(log_messages))
    
    if failed_models:
        print(f"\n⚠️  Failed to load {len(failed_models)} models (see model_loading_log.txt)")

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
            page * page_size,
            (page + 1) * page_size - 1,
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
                "Type": model_bundle["meta"].get("model_type", "tabular"),
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
        "season": float(date.today().year),
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.merge(overall_tendencies, on="pitcher_name", how="left")
    input_df = input_df.merge(hand_tendencies, on=["pitcher_name", "stand"], how="left")
    input_df = input_df.merge(count_tendencies, on=["pitcher_name", "count"], how="left")

    tendency_cols = [column for column in input_df.columns if "_pct" in column]
    input_df[tendency_cols] = input_df[tendency_cols].fillna(0)
    return input_df


def build_sequence_history_df(recent_pitches, fallback_state):
    rows = []
    for index, pitch in enumerate(recent_pitches or []):
        rows.append(
            {
                "play_event_index": pitch.get("play_event_index", index),
                "pitch_type": normalize_prev_pitch(pitch.get("pitch_code") or pitch.get("pitch_type", "FF")),
                "balls": pitch.get("balls", fallback_state["balls"]),
                "strikes": pitch.get("strikes", fallback_state["strikes"]),
                "outs": pitch.get("outs", fallback_state["outs"]),
                "on_1b": fallback_state["on_1b"],
                "on_2b": fallback_state["on_2b"],
                "on_3b": fallback_state["on_3b"],
                "score_diff": fallback_state["score_diff"],
            }
        )
    return pd.DataFrame(rows)


def predict_with_tabular_model(model_bundle, model_name, features_df):
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    feature_artifact = model_bundle["feature_columns"]

    if hasattr(feature_artifact, "transform"):
        model_input = feature_artifact.transform(features_df)
        if model_name == "catboost":
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
                "Pitch Type": [PITCH_NAMES.get(code, code) for code in label_encoder.classes_],
                "Probability": [round(probability * 100, 1) for probability in proba],
            }
        ).sort_values("Probability", ascending=False)

    return pitch_code, proba_df


def predict_with_sequence_model(model_bundle, features_df, recent_pitches, fallback_state):
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    preprocessor = model_bundle["preprocessor"]

    history_df = build_sequence_history_df(recent_pitches, fallback_state)
    model_inputs = build_sequence_inference_inputs(history_df, features_df, preprocessor)
    predictions = model.predict(
        [
            model_inputs["sequence_tokens"],
            model_inputs["sequence_numeric"],
            model_inputs["static_features"],
        ],
        verbose=0,
    )[0]

    pred_index = int(predictions.argmax())
    pitch_code = label_encoder.inverse_transform([pred_index])[0]
    proba_df = pd.DataFrame(
        {
            "Pitch Type": [PITCH_NAMES.get(code, code) for code in label_encoder.classes_],
            "Probability": [round(probability * 100, 1) for probability in predictions],
        }
    ).sort_values("Probability", ascending=False)

    return pitch_code, proba_df


def predict_with_model(model_name, features_df, recent_pitches=None, fallback_state=None):
    model_bundle = MODELS[model_name]
    model_type = model_bundle["meta"].get("model_type", "tabular")

    if model_type == "tabular":
        return predict_with_tabular_model(model_bundle, model_name, features_df)

    return predict_with_sequence_model(
        model_bundle,
        features_df,
        recent_pitches=recent_pitches or [],
        fallback_state=fallback_state or {},
    )


def predict_pitch(selected_models, recent_pitches=None, **kwargs):
    features_df = build_prediction_features(**kwargs)
    results = {}

    fallback_state = {
        "balls": kwargs["balls"],
        "strikes": kwargs["strikes"],
        "outs": kwargs["outs"],
        "on_1b": kwargs["on_1b"],
        "on_2b": kwargs["on_2b"],
        "on_3b": kwargs["on_3b"],
        "score_diff": kwargs["score_diff"],
    }

    for model_name in selected_models:
        pitch_code, proba_df = predict_with_model(
            model_name,
            features_df,
            recent_pitches=recent_pitches,
            fallback_state=fallback_state,
        )
        results[model_name] = {
            "pitch_code": pitch_code,
            "probabilities": proba_df,
        }

    return results


def render_prediction_results(results):
    if len(results) == 1:
        model_name, result = next(iter(results.items()))
        st.success(f"{MODEL_LABELS[model_name]}: **{PITCH_NAMES.get(result['pitch_code'], result['pitch_code'])}**")
        if result["probabilities"] is not None:
            st.dataframe(result["probabilities"], use_container_width=True)
        return

    summary_rows = []
    for model_name, result in results.items():
        predicted_pitch_name = PITCH_NAMES.get(result["pitch_code"], result["pitch_code"])
        
        # Get confidence for the predicted pitch
        confidence = None
        if result["probabilities"] is not None and not result["probabilities"].empty:
            prob_row = result["probabilities"][result["probabilities"]["Pitch Type"] == predicted_pitch_name]
            if not prob_row.empty:
                confidence = prob_row.iloc[0]["Probability"]
        
        summary_rows.append(
            {
                "Model": MODEL_LABELS[model_name],
                "Type": MODELS[model_name]["meta"].get("model_type", "tabular"),
                "Confidence": confidence,
                "Predicted Pitch": predicted_pitch_name,
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
        game_labels = [f"{game['away_team']} @ {game['home_team']} ({game['status']})" for game in games]
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
                col6.metric(
                    "Bases",
                    f"{'1' if state['on_1b'] else '-'} {'2' if state['on_2b'] else '-'} {'3' if state['on_3b'] else '-'}",
                )

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
                    recent_pitches=state["recent_pitches"],
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
                manual_history = [
                    {
                        "pitch_code": prev_pitch,
                        "pitch_type": prev_pitch,
                        "balls": balls,
                        "strikes": strikes,
                        "outs": outs,
                        "play_event_index": 0,
                    }
                ]
                results = predict_pitch(
                    manual_selected_models,
                    recent_pitches=manual_history,
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
