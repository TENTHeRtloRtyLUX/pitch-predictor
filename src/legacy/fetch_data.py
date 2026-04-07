from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from supabase_data_loader import iter_pitches_from_supabase


def fetch_all_pitches(seasons, page_size=1000):
    frames = []

    for i, batch_df in enumerate(
        iter_pitches_from_supabase(seasons=seasons, batch_size=page_size),
        start=1,
    ):
        if batch_df.empty:
            continue

        frames.append(batch_df)

        if i % 100 == 0:
            print(f"Fetched {sum(len(frame) for frame in frames)} pitches so far...")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"Total pitches fetched: {len(df)}")
    return df


if __name__ == "__main__":
    df = fetch_all_pitches(seasons=[2023, 2024, 2025])
    df.to_csv("data/training_data.csv", index=False)
    print("Saved to data/training_data.csv")