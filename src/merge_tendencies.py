import pandas as pd

def merge_tendencies(df, overall, hand, count):
    merged = df.merge(overall, on="pitcher_name", how="left")
    merged = merged.merge(hand, on=["pitcher_name", "stand"], how="left")
    merged = merged.merge(count, on=["pitcher_name", "count"], how="left")

    tendency_cols = [c for c in merged.columns if "_pct" in c]
    merged[tendency_cols] = merged[tendency_cols].fillna(0)

    return merged

if __name__ == "__main__":
    raise SystemExit(
        "This module now provides reusable functions. Import and call merge_tendency_features(df, overall, hand, count)."
    )