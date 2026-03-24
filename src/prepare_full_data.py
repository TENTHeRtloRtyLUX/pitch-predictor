import pandas as pd

df = pd.read_csv("data/season_2023_pitches.csv")

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nPitch types:\n", df["pitch_type"].value_counts())
print("\nSample:\n", df.head())

df = df.rename(columns={"outs": "outs_when_up"})

min_samples = 5000
pitch_counts = df["pitch_type"].value_counts()
valid_pitches = pitch_counts[pitch_counts >= min_samples].index
df = df[df["pitch_type"].isin(valid_pitches)]
print(f"\nAfter filtering rare pitches: {len(df)} rows")
print("Remaining pitch types:\n", df["pitch_type"].value_counts())

df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

df = df.sort_values(["game_id", "pitcher_name", "batter_name"])

df["at_bat_number"] = (
    df.groupby("game_id")["batter_name"]
    .transform(lambda x: (x != x.shift()).cumsum())
)

df["pitch_number"] = df.groupby(
    ["game_id", "at_bat_number"]
).cumcount() + 1

df = df.sort_values(["game_id", "at_bat_number", "pitch_number"])
df["prev_pitch"] = df.groupby(
    ["game_id", "at_bat_number"]
)["pitch_type"].shift(1)

df = df.dropna(subset=["prev_pitch", "pitch_type"])

print("\nFinal shape:", df.shape)
print("\nSample:\n", df.head())

df.to_csv("data/clean_full_pitches.csv", index=False)
print("\nSaved to data/clean_full_pitches.csv")