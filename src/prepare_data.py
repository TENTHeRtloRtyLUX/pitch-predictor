import pandas as pd

df = pd.read_csv("data/raw_pitches.csv")

df = df.dropna(subset=["pitch_type"])

df["on_1b"] = df["on_1b"].notnull().astype(int)
df["on_2b"] = df["on_2b"].notnull().astype(int)
df["on_3b"] = df["on_3b"].notnull().astype(int)

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
]

df = df[cols_to_keep]

print("Clean dataset shape: ", df.shape)
print("\nPitch Types: \n", df["pitch_type"].value_counts())
print("\nSample data:\n", df.head())

df.to_csv("data/clean_pitches.csv", index=False)

