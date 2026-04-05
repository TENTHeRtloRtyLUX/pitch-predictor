import pandas as pd


def counts_to_rates(counts, group_levels):
    if counts.empty:
        index_names = counts.index.names[:group_levels]
        empty_index = pd.MultiIndex.from_arrays([[] for _ in range(group_levels)], names=index_names)
        return pd.DataFrame(index=empty_index).reset_index()

    mix = counts.unstack(fill_value=0).sort_index(axis=1)
    mix = mix.div(mix.sum(axis=1), axis=0)
    return mix.reset_index()


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


def build_all_tendencies_from_counts(overall_counts, hand_counts, count_counts):
    overall = counts_to_rates(overall_counts, group_levels=1)
    hand = counts_to_rates(hand_counts, group_levels=2)
    count = counts_to_rates(count_counts, group_levels=2)

    if "pitcher_name" in overall.columns:
        overall.columns = [
            col if col == "pitcher_name" else f"pitcher_pct_{col}"
            for col in overall.columns
        ]

    if {"pitcher_name", "stand"}.issubset(hand.columns):
        hand.columns = [
            col if col in {"pitcher_name", "stand"} else f"pitcher_vs_{col}_pct"
            for col in hand.columns
        ]

    if {"pitcher_name", "count"}.issubset(count.columns):
        count.columns = [
            col if col in {"pitcher_name", "count"} else f"pitcher_count_{col}_pct"
            for col in count.columns
        ]

    return overall, hand, count

if __name__ == "__main__":
    raise SystemExit(
        "This module now provides reusable functions. Import and call build_all_tendencies(df)."
    )
