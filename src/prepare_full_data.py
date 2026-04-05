import pandas as pd

DEFAULT_VALID_PITCHES = ["FF", "SL", "SI", "CH", "FC", "CU", "ST", "FS", "KC"]

def filter_valid_pitches(df, valid_pitches=None):
    valid_pitches = valid_pitches or DEFAULT_VALID_PITCHES
    return df[df["pitch_type"].isin(valid_pitches)].copy()

def add_count_feature(df):
    df = df.copy()
    df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    return df

def add_at_bat_and_pitch_features(df):
    df = df.copy()

    df = df.sort_values(["game_id", "pitcher_name", "batter_name"]).reset_index(drop=True)

    df["at_bat_number"] = (df.groupby("game_id")["batter_name"].transform(lambda x: (x != x.shift()).cumsum()))

    df["pitch_number"] = df.groupby(["game_id", "at_bat_number"]).cumcount() + 1

    df = df.sort_values(["game_id", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    df["prev_pitch"] = df.groupby(["game_id", "at_bat_number"])["pitch_type"].shift(1)

    return df

def prepare_pitch_data(df, valid_pitches=None, drop_first_pitch=True):
    df = df.copy()

    if "outs_when_up" in df.columns and "outs" not in df.columns:
        df = df.rename(columns={"outs_when_up": "outs"})

    df = filter_valid_pitches(df, valid_pitches)
    df = add_count_feature(df)
    df = add_at_bat_and_pitch_features(df)

    if drop_first_pitch:
        df = df.dropna(subset=["prev_pitch", "pitch_type"]).reset_index(drop=True)

    return df

if __name__ == "__main__":
    raise SystemExit(
        "This module now provides reusable functions. Import and call prepare_pitch_data(df)."
    )