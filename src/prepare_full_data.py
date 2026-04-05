import pandas as pd

DEFAULT_VALID_PITCHES = ["FF", "SL", "SI", "CH", "FC", "CU", "ST", "FS", "KC"]

def filter_valid_pitches(df, valid_pitches=None):
    valid_pitches = valid_pitches or DEFAULT_VALID_PITCHES
    return df[df["pitch_type"].isin(valid_pitches)].copy()


def filter_valid_counts(df):
    df = df.copy()
    return df[
        df["balls"].between(0, 3)
        & df["strikes"].between(0, 2)
        & df["outs"].between(0, 2)
    ].copy()

def add_count_feature(df):
    df = df.copy()
    df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    return df

def add_at_bat_and_pitch_features(df):
    df = df.copy()

    required_temporal_columns = {"game_id", "at_bat_index", "play_event_index"}
    if required_temporal_columns.issubset(df.columns):
        df = df.sort_values(["game_id", "at_bat_index", "play_event_index"]).reset_index(drop=True)
        if "at_bat_number" not in df.columns:
            df["at_bat_number"] = df["at_bat_index"] + 1
        if "pitch_number" not in df.columns:
            df["pitch_number"] = df.groupby(["game_id", "at_bat_index"]).cumcount() + 1
        group_columns = ["game_id", "at_bat_index"]
    else:
        raise ValueError(
            "Pitch data is missing true temporal identifiers. Expected "
            "'game_id', 'at_bat_index', and 'play_event_index'."
        )

    if "prev_pitch" not in df.columns:
        df["prev_pitch"] = df.groupby(group_columns)["pitch_type"].shift(1)

    return df

def prepare_pitch_data(df, valid_pitches=None, drop_first_pitch=True):
    df = df.copy()

    if "outs_when_up" in df.columns and "outs" not in df.columns:
        df = df.rename(columns={"outs_when_up": "outs"})

    df = filter_valid_pitches(df, valid_pitches)
    df = filter_valid_counts(df)
    df = add_count_feature(df)
    df = add_at_bat_and_pitch_features(df)

    if drop_first_pitch:
        df = df.dropna(subset=["prev_pitch", "pitch_type"]).reset_index(drop=True)

    return df

if __name__ == "__main__":
    raise SystemExit(
        "This module now provides reusable functions. Import and call prepare_pitch_data(df)."
    )
