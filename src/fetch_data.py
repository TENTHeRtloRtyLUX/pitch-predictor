from pybaseball import statcast_pitcher
import pandas as pd
import os

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

if __name__ == "__main__":
    df = fetch_all_pitchers()

    out_path = os.path.join("data", "raw_pitches.csv")
    df.to_csv(out_path, index=False)

    print(f"Data saved to {out_path}")
    print(df["pitch_type"].value_counts())


