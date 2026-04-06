import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from build_pitcher_tendencies import build_all_tendencies_from_counts
from merge_tendencies import merge_tendency_features
from prepare_full_data import prepare_pitch_data
from supabase_data_loader import iter_pitches_from_supabase

DEFAULT_SEQUENCE_LENGTH = 12
DROP_SEQUENCE_COLUMNS = {
    "id",
    "pitch_uid",
    "game_id",
    "at_bat_index",
    "at_bat_number",
    "play_event_index",
    "pitch_number",
    "batter_name",
    "pitch_type",
}
SEQUENCE_NUMERIC_COLUMNS = [
    "balls",
    "strikes",
    "outs",
    "on_1b",
    "on_2b",
    "on_3b",
    "score_diff",
]
STATIC_CATEGORICAL_COLUMNS = ["p_throws", "stand", "prev_pitch", "count"]
TENDENCY_FILENAMES = {
    "overall": "pitcher_overall_tendencies.csv",
    "hand": "pitcher_hand_tendencies.csv",
    "count": "pitcher_count_tendencies.csv",
}
TRAINING_DF_FILENAME = "training_dataframe.joblib"
SEQUENCE_DATASET_FILENAME = "sequence_dataset.joblib"
PREP_SUMMARY_FILENAME = "prep_summary.json"


def _empty_series():
    return pd.Series(dtype="int64")


def _add_group_counts(existing_counts, new_counts):
    if new_counts is None or new_counts.empty:
        return existing_counts
    if existing_counts is None or existing_counts.empty:
        return new_counts.astype("int64")
    return existing_counts.add(new_counts, fill_value=0).astype("int64")


def _normalize_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_training_dataframe(seasons, batch_size=1000):
    prepared_batches = []
    overall_counts = _empty_series()
    hand_counts = _empty_series()
    count_counts = _empty_series()

    for raw_batch in iter_pitches_from_supabase(seasons=seasons, batch_size=batch_size):
        if raw_batch.empty:
            continue

        prepared_batch = prepare_pitch_data(raw_batch)
        if prepared_batch.empty:
            continue

        prepared_batches.append(prepared_batch)
        overall_counts = _add_group_counts(
            overall_counts,
            prepared_batch.groupby(["pitcher_name", "pitch_type"]).size(),
        )
        hand_counts = _add_group_counts(
            hand_counts,
            prepared_batch.groupby(["pitcher_name", "stand", "pitch_type"]).size(),
        )
        count_counts = _add_group_counts(
            count_counts,
            prepared_batch.groupby(["pitcher_name", "count", "pitch_type"]).size(),
        )

    if not prepared_batches:
        print("No data loaded for the specified seasons.")
        return pd.DataFrame()

    prepared_df = pd.concat(prepared_batches, ignore_index=True)
    overall_tendencies, hand_tendencies, count_tendencies = build_all_tendencies_from_counts(
        overall_counts,
        hand_counts,
        count_counts,
    )

    training_df = merge_tendency_features(
        prepared_df,
        overall_tendencies,
        hand_tendencies,
        count_tendencies,
    )
    return training_df


def refresh_tendency_files(seasons, output_dir="data", batch_size=1000):
    training_df = build_training_dataframe(seasons=seasons, batch_size=batch_size)
    return save_tendency_files_from_training_dataframe(training_df, output_dir=output_dir)


def save_tendency_files_from_training_dataframe(training_df, output_dir="data"):
    if training_df.empty:
        raise ValueError("Cannot build tendency files because the training dataframe is empty.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    overall_tendencies = training_df[
        ["pitcher_name"] + [col for col in training_df.columns if col.startswith("pitcher_pct_")]
    ].drop_duplicates()
    hand_tendencies = training_df[
        ["pitcher_name", "stand"] + [col for col in training_df.columns if col.startswith("pitcher_vs_")]
    ].drop_duplicates()
    count_tendencies = training_df[
        ["pitcher_name", "count"] + [col for col in training_df.columns if col.startswith("pitcher_count_")]
    ].drop_duplicates()

    overall_path = output_path / TENDENCY_FILENAMES["overall"]
    hand_path = output_path / TENDENCY_FILENAMES["hand"]
    count_path = output_path / TENDENCY_FILENAMES["count"]

    overall_tendencies.to_csv(overall_path, index=False)
    hand_tendencies.to_csv(hand_path, index=False)
    count_tendencies.to_csv(count_path, index=False)

    return {
        "success": True,
        "training_rows": int(len(training_df)),
        "files": {
            "overall": str(overall_path.as_posix()),
            "hand": str(hand_path.as_posix()),
            "count": str(count_path.as_posix()),
        },
    }


def _split_games(training_df, test_size=0.2, random_state=42):
    unique_game_ids = sorted(training_df["game_id"].dropna().unique().tolist())
    if len(unique_game_ids) < 2:
        raise ValueError("Need at least two games to split sequence data by game_id.")

    train_games, test_games = train_test_split(
        unique_game_ids,
        test_size=test_size,
        random_state=random_state,
    )
    return set(train_games), set(test_games)


def _build_static_feature_frame(training_df):
    drop_columns = [column for column in DROP_SEQUENCE_COLUMNS if column in training_df.columns]
    feature_df = training_df.drop(columns=drop_columns).copy()
    if "season" in feature_df.columns:
        feature_df["season"] = feature_df["season"].astype(float)
    return feature_df


def fit_sequence_preprocessor(train_df, max_sequence_length=DEFAULT_SEQUENCE_LENGTH):
    static_feature_df = _build_static_feature_frame(train_df)
    if "pitch_type" in static_feature_df.columns:
        static_feature_df = static_feature_df.drop(columns=["pitch_type"])

    static_numeric_columns = [
        column
        for column in static_feature_df.columns
        if column not in STATIC_CATEGORICAL_COLUMNS and pd.api.types.is_numeric_dtype(static_feature_df[column])
    ]
    static_encoder = _normalize_one_hot_encoder()
    static_scaler = StandardScaler()
    pitch_encoder = LabelEncoder()

    if static_numeric_columns:
        static_scaler.fit(static_feature_df[static_numeric_columns].fillna(0))
    if STATIC_CATEGORICAL_COLUMNS:
        static_encoder.fit(static_feature_df.reindex(columns=STATIC_CATEGORICAL_COLUMNS, fill_value="__missing__"))

    pitch_encoder.fit(train_df["pitch_type"])
    pitch_to_id = {"__PAD__": 0}
    for index, pitch_code in enumerate(pitch_encoder.classes_, start=1):
        pitch_to_id[pitch_code] = index

    return {
        "max_sequence_length": max_sequence_length,
        "sequence_numeric_columns": SEQUENCE_NUMERIC_COLUMNS,
        "static_numeric_columns": static_numeric_columns,
        "static_categorical_columns": STATIC_CATEGORICAL_COLUMNS,
        "static_scaler": static_scaler,
        "static_encoder": static_encoder,
        "pitch_to_id": pitch_to_id,
    }


def _transform_static_row(row_df, preprocessor):
    numeric_columns = preprocessor["static_numeric_columns"]
    categorical_columns = preprocessor["static_categorical_columns"]
    parts = []

    if numeric_columns:
        numeric_values = preprocessor["static_scaler"].transform(
            row_df.reindex(columns=numeric_columns, fill_value=0)
        )
        parts.append(numeric_values.astype("float32"))

    if categorical_columns:
        categorical_values = preprocessor["static_encoder"].transform(
            row_df.reindex(columns=categorical_columns, fill_value="__missing__").astype(str)
        )
        parts.append(categorical_values.astype("float32"))

    if not parts:
        return np.zeros((len(row_df), 0), dtype="float32")

    return np.concatenate(parts, axis=1)


def build_sequence_examples(df, preprocessor):
    if df.empty:
        return {
            "sequence_tokens": np.empty((0, preprocessor["max_sequence_length"]), dtype="int32"),
            "sequence_numeric": np.empty(
                (0, preprocessor["max_sequence_length"], len(preprocessor["sequence_numeric_columns"])),
                dtype="float32",
            ),
            "static_features": np.empty((0, 0), dtype="float32"),
            "targets": np.empty((0,), dtype="int32"),
            "group_ids": [],
        }

    sequence_tokens = []
    sequence_numeric = []
    static_features = []
    targets = []
    group_ids = []
    pitch_to_id = preprocessor["pitch_to_id"]
    max_length = preprocessor["max_sequence_length"]
    sequence_numeric_columns = preprocessor["sequence_numeric_columns"]

    for (_, _), at_bat_df in df.groupby(["game_id", "at_bat_index"], sort=False):
        ordered = at_bat_df.sort_values("play_event_index").reset_index(drop=True)
        for idx in range(1, len(ordered)):
            history = ordered.iloc[:idx]
            current_row = ordered.iloc[[idx]]

            token_history = [pitch_to_id.get(pitch, 0) for pitch in history["pitch_type"].tolist()]
            numeric_history = history.reindex(columns=sequence_numeric_columns, fill_value=0).to_numpy(
                dtype="float32"
            )

            token_array = np.zeros(max_length, dtype="int32")
            numeric_array = np.zeros((max_length, len(sequence_numeric_columns)), dtype="float32")

            trimmed_tokens = token_history[-max_length:]
            trimmed_numeric = numeric_history[-max_length:]

            token_array[-len(trimmed_tokens) :] = trimmed_tokens
            if len(trimmed_numeric):
                numeric_array[-len(trimmed_numeric) :, :] = trimmed_numeric

            sequence_tokens.append(token_array)
            sequence_numeric.append(numeric_array)

            static_row = _build_static_feature_frame(current_row)
            if "pitch_type" in static_row.columns:
                static_row = static_row.drop(columns=["pitch_type"])
            static_features.append(_transform_static_row(static_row, preprocessor)[0])
            targets.append(pitch_to_id[current_row["pitch_type"].iloc[0]] - 1)
            group_ids.append(current_row["game_id"].iloc[0])

    if not targets:
        raise ValueError("No sequence examples were created from the training dataframe.")

    return {
        "sequence_tokens": np.asarray(sequence_tokens, dtype="int32"),
        "sequence_numeric": np.asarray(sequence_numeric, dtype="float32"),
        "static_features": np.asarray(static_features, dtype="float32"),
        "targets": np.asarray(targets, dtype="int32"),
        "group_ids": group_ids,
    }


def build_sequence_training_dataset(
    seasons,
    batch_size=1000,
    max_sequence_length=DEFAULT_SEQUENCE_LENGTH,
    test_size=0.2,
    random_state=42,
    training_df=None,
):
    training_df = training_df if training_df is not None else build_training_dataframe(seasons=seasons, batch_size=batch_size)
    if training_df.empty:
        raise ValueError("Training dataframe is empty; cannot build sequence dataset.")

    train_games, test_games = _split_games(training_df, test_size=test_size, random_state=random_state)
    train_df = training_df[training_df["game_id"].isin(train_games)].copy()
    test_df = training_df[training_df["game_id"].isin(test_games)].copy()

    preprocessor = fit_sequence_preprocessor(train_df, max_sequence_length=max_sequence_length)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["pitch_type"])

    train_examples = build_sequence_examples(train_df, preprocessor)
    test_examples = build_sequence_examples(test_df, preprocessor)

    return {
        "training_df": training_df,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "train_game_ids": sorted(train_games),
        "test_game_ids": sorted(test_games),
    }


def save_training_dataframe(training_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_df, output_path)
    return output_path


def load_training_dataframe(input_path):
    return joblib.load(input_path)


def save_sequence_training_dataset(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, output_path)
    return output_path


def load_sequence_training_dataset(input_path):
    input_path = Path(input_path)
    candidate_paths = [
        input_path,
        Path("retrain_shared") / input_path.name,
        Path("output") / input_path.name,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return joblib.load(candidate)
    raise FileNotFoundError(
        "Prepared sequence dataset not found. Checked: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def save_sequence_preprocessor(preprocessor, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)


def save_sequence_split_manifest(dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_game_ids": dataset["train_game_ids"],
        "test_game_ids": dataset["test_game_ids"],
        "train_examples": int(len(dataset["train_examples"]["targets"])),
        "test_examples": int(len(dataset["test_examples"]["targets"])),
    }
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def build_sequence_inference_inputs(history_df, current_feature_df, preprocessor):
    history_df = history_df.sort_values("play_event_index").reset_index(drop=True)
    token_array = np.zeros((1, preprocessor["max_sequence_length"]), dtype="int32")
    numeric_array = np.zeros(
        (1, preprocessor["max_sequence_length"], len(preprocessor["sequence_numeric_columns"])),
        dtype="float32",
    )

    if not history_df.empty:
        token_history = [preprocessor["pitch_to_id"].get(pitch, 0) for pitch in history_df["pitch_type"].tolist()]
        numeric_history = history_df.reindex(
            columns=preprocessor["sequence_numeric_columns"], fill_value=0
        ).to_numpy(dtype="float32")

        trimmed_tokens = token_history[-preprocessor["max_sequence_length"] :]
        trimmed_numeric = numeric_history[-preprocessor["max_sequence_length"] :]
        token_array[0, -len(trimmed_tokens) :] = trimmed_tokens
        if len(trimmed_numeric):
            numeric_array[0, -len(trimmed_numeric) :, :] = trimmed_numeric

    static_features = _transform_static_row(current_feature_df, preprocessor)
    return {
        "sequence_tokens": token_array,
        "sequence_numeric": numeric_array,
        "static_features": static_features.astype("float32"),
    }


def prepare_retraining_artifacts(
    seasons,
    output_dir,
    batch_size=1000,
    max_sequence_length=DEFAULT_SEQUENCE_LENGTH,
    test_size=0.2,
    random_state=42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_df = build_training_dataframe(seasons=seasons, batch_size=batch_size)
    if training_df.empty:
        raise ValueError("Training dataframe is empty; cannot prepare retraining artifacts.")

    tendency_summary = save_tendency_files_from_training_dataframe(training_df, output_dir=output_path / "data")
    training_df_path = save_training_dataframe(training_df, output_path / TRAINING_DF_FILENAME)
    sequence_dataset = build_sequence_training_dataset(
        seasons=seasons,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        test_size=test_size,
        random_state=random_state,
        training_df=training_df,
    )
    sequence_dataset_path = save_sequence_training_dataset(sequence_dataset, output_path / SEQUENCE_DATASET_FILENAME)

    summary = {
        "success": True,
        "seasons": list(seasons),
        "training_rows": int(len(training_df)),
        "training_dataframe_path": str(training_df_path.as_posix()),
        "sequence_dataset_path": str(sequence_dataset_path.as_posix()),
        "tendency_files": tendency_summary["files"],
        "train_examples": int(len(sequence_dataset["train_examples"]["targets"])),
        "test_examples": int(len(sequence_dataset["test_examples"]["targets"])),
    }
    with open(output_path / PREP_SUMMARY_FILENAME, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


if __name__ == "__main__":
    df = build_training_dataframe(seasons=[2023, 2024, 2025])
    print(df.head())
    print(df.shape)
