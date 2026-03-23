from pybaseball import statcast_pitcher
from pybaseball import statcast
import pandas as pd
import os

from pybaseball import cache
cache.enable()

PITCHERS = {
    "Gerrit Cole": 543037,
    "Corbin Burnes": 669203,
    "Framber Valdez": 664285,
    "Kevin Gausman": 592332,
}

START_DATE = "2023-03-30"
END_DATE = "2023-10-01"

def fetch_all_pitchers():
    all_data = []

    for name, pid in PITCHERS.items():
        print(f"Data for {name}")
        df = statcast_pitcher(START_DATE, END_DATE, pid)
        df["pitcher_name"] = name
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined

def fetch_batter_stats():
    df = statcast("2022-04-07", "2022-10-05")

    batter_stats = df.groupby("batter").apply(lambda x: pd.Series({
        "k_rate": (x["events"] == "strikeout").sum() / max(x["events"].notna().sum(), 1),
        "bb_rate": (x["events"] == "walk").sum() / max(x["events"].notna().sum(), 1),
    })).reset_index()

    batter_stats.to_csv("data/batter_stats.csv", index=False)


if __name__ == "__main__":
    df = fetch_all_pitchers()

    out_path = os.path.join("data", "raw_pitches.csv")
    df.to_csv(out_path, index=False)

    print(f"Data saved to {out_path}")
    print(df["pitch_type"].value_counts())

    fetch_batter_stats()


