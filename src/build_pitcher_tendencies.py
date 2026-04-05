import pandas as pd

def build_overall_tendencies(df):
    pitch_mix = df.groupby(["pitcher_name", "pitch_type"]).size().unstack(fill_value=0)
    pitch_mix = pitch_mix.div(pitch_mix.sum(axis=1), axis=0)
    pitch_mix.columns = [f"pitcher_pct_{c}" for c in pitch_mix.columns]
    return pitch_mix.reset_index()


def build_hand_tendencies(df):
    hand_mix = df.groupby(["pitcher_name", "stand", "pitch_type"]).size().unstack(fill_value=0)
    hand_mix = hand_mix.div(hand_mix.sum(axis=1), axis=0)
    hand_mix.columns = [f"pitcher_vs_{c}_pct" for c in hand_mix.columns]
    return hand_mix.reset_index()

def build_count_tendencies(df):
    count_mix = df.groupby(["pitcher_name", "count", "pitch_type"]).size().unstack(fill_value=0)
    count_mix = count_mix.div(count_mix.sum(axis=1), axis=0)
    count_mix.columns = [f"pitcher_count_{c}_pct" for c in count_mix.columns]
    return count_mix.reset_index()

def build_all_tendencies(df):
    overall = build_overall_tendencies(df)
    hand = build_hand_tendencies(df)
    count = build_count_tendencies(df)

    return overall, hand, count

if __name__ == "__main__":
    raise SystemExit(
        "This module now provides reusable functions. Import and call build_all_tendencies(df)."
    )