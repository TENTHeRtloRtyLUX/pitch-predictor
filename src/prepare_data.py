import pandas as pd

df = pd.read_csv("data/raw_pitches.csv")

df = df.dropna(subset=["pitch_type"])

df["on_1b"] = df["on_1b"].notnull().astype(int)
df["on_2b"] = df["on_2b"].notnull().astype(int)
df["on_3b"] = df["on_3b"].notnull().astype(int)

df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

df = df.sort_values(["pitcher_name", "game_date", "at_bat_number", "pitch_number"])
df["prev_pitch"] = df.groupby(["pitcher_name", "game_date"])["pitch_type"].shift(1)
df = df.dropna(subset=["prev_pitch"])

cols_to_keep = [
    "pitcher_name",
    "pitch_type",
    "p_throws",
    "stand",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "prev_pitch",
    "count",
]

df = df[cols_to_keep]

print("Clean dataset shape: ", df.shape)
print("\nPitch Types: \n", df["pitch_type"].value_counts())
print("\nSample data:\n", df.head())

df.to_csv("data/clean_pitches.csv", index=False)

