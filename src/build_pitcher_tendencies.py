import pandas as pd

df = pd.read_csv("data/clean_full_pitches.csv")

pitch_mix = df.groupby(["pitcher_name", "pitch_type"]).size().unstack(fill_value=0)
pitch_mix = pitch_mix.div(pitch_mix.sum(axis=1), axis=0)
pitch_mix.columns = [f"pitcher_pct_{c}" for c in pitch_mix.columns]
pitch_mix = pitch_mix.reset_index()

hand_mix = df.groupby(["pitcher_name", "stand", "pitch_type"]).size().unstack(fill_value=0)
hand_mix = hand_mix.div(hand_mix.sum(axis=1), axis=0)
hand_mix.columns = [f"pitcher_vs_{c}_pct" for c in hand_mix.columns]
hand_mix = hand_mix.reset_index()

count_mix = df.groupby(["pitcher_name", "count", "pitch_type"]).size().unstack(fill_value=0)
count_mix = count_mix.div(count_mix.sum(axis=1), axis=0)
count_mix.columns = [f"pitcher_count_{c}_pct" for c in count_mix.columns]
count_mix = count_mix.reset_index()

pitch_mix.to_csv("data/pitcher_overall_tendencies.csv", index=False)
hand_mix.to_csv("data/pitcher_hand_tendencies.csv", index=False)
count_mix.to_csv("data/pitcher_count_tendencies.csv", index=False)

print("Pitcher tendencies saved.")
print("Overall:", pitch_mix.shape)
print("By handedness:", hand_mix.shape)
print("By count:", count_mix.shape)