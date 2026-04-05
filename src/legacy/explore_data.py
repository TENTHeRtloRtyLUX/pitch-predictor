import pandas as pd

df = pd.read_csv("data/raw_pitches.csv")

print("Shape: ", df.shape)
print("\nColumns: ", df.columns.tolist())
print("\nPitch Types: \n", df["pitch_type"].value_counts())
print("\nSample data:\n", df.iloc[0])

cols_we_want = [
    "pitch_type", "p_throws", "stand",
    "balls", "strikes", "outs_when_up",
    "inning", "on_1b", "on_2b", "on_3b"
]

print("\nMissing vals in key cols:\n", df[cols_we_want].isnull().sum())

print("\nPitcher hand (p_throws):\n", df["p_throws"].value_counts())
print("\nBatter stance:\n", df["stand"].value_counts())
