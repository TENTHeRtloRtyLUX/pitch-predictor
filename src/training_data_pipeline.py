import pandas as pd

from prepare_full_data import prepare_pitch_data
from build_pitcher_tendencies import build_all_tendencies_from_counts
from merge_tendencies import merge_tendency_features
from supabase_data_loader import iter_pitches_from_supabase


def build_training_dataframe(seasons, batch_size=1000):
    prepared_batches = []
    overall_counts = pd.Series(dtype="int64")
    hand_counts = pd.Series(dtype="int64")
    count_counts = pd.Series(dtype="int64")

    for raw_batch in iter_pitches_from_supabase(seasons=seasons, batch_size=batch_size):
        if raw_batch.empty:
            continue

        prepared_batch = prepare_pitch_data(raw_batch)
        if prepared_batch.empty:
            continue

        prepared_batches.append(prepared_batch)
        overall_counts = overall_counts.add(
            prepared_batch.groupby(["pitcher_name", "pitch_type"]).size(),
            fill_value=0,
        )
        hand_counts = hand_counts.add(
            prepared_batch.groupby(["pitcher_name", "stand", "pitch_type"]).size(),
            fill_value=0,
        )
        count_counts = count_counts.add(
            prepared_batch.groupby(["pitcher_name", "count", "pitch_type"]).size(),
            fill_value=0,
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


if __name__ == "__main__":
    df = build_training_dataframe(seasons=[2023, 2024, 2025])
    print(df.head())
    print(df.shape)
