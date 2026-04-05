import pandas as pd

from prepare_full_data import prepare_pitch_data
from build_pitcher_tendencies import build_all_tendencies
from merge_tendencies import merge_tendency_features
from supabase_data_loader import load_pitches_from_supabase


def build_training_dataframe(seasons, batch_size=5000):
    raw_df = load_pitches_from_supabase(seasons=seasons, batch_size=batch_size)

    if raw_df.empty:
        print("No data loaded for the specified seasons.")
        return raw_df

    prepared_df = prepare_pitch_data(raw_df)
    overall_tendencies, hand_tendencies, count_tendencies = build_all_tendencies(prepared_df)

    training_df = merge_tendency_features(
        prepared_df,
        overall_tendencies,
        hand_tendencies,
        count_tendencies,
    )
    return training_df


if __name__ == "__main__":
    df = build_training_dataframe(seasons=[2023, 2024, 2025])
    print(df.head())
    print(df.shape)
