import pandas as pd

df = pd.read_csv("data/clean_full_pitches.csv")
overall = pd.read_csv("data/pitcher_overall_tendencies.csv")
hand = pd.read_csv("data/pitcher_hand_tendencies.csv")
count = pd.read_csv("data/pitcher_count_tendencies.csv")

df = df.merge(overall, on="pitcher_name", how="left")

df = df.merge(hand, on=["pitcher_name", "stand"], how="left")

df = df.merge(count, on=["pitcher_name", "count"], how="left")

tendency_cols = [c for c in df.columns if "_pct" in c]
df[tendency_cols] = df[tendency_cols].fillna(0)

print("Shape after merging tendencies:", df.shape)
print("New tendency columns:", len(tendency_cols))
print("\nSample tendency data:")
print(df[["pitcher_name"] + tendency_cols[:5]].head())

df.to_csv("data/clean_full_pitches_with_tendencies.csv", index=False)
print("\nSaved to data/clean_full_pitches_with_tendencies.csv")