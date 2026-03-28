import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_pitchers():
    pitcher_df = pd.read_csv("data/clean_full_pitches.csv")
    pitcher_df = pitcher_df[["pitcher_name", "p_throws"]].drop_duplicates()
    pitcher_df = pitcher_df.rename(columns={"pitcher_name": "name", "p_throws": "throws"})
    data = pitcher_df.to_dict(orient="records")
    supabase.table("pitchers").insert(data).execute()

def load_pitches():
    pitch_df = pd.read_csv("data/clean_full_pitches.csv")
    pitch_df = pitch_df[[
        "game_id", "pitcher_name", "batter_name", "p_throws", "stand",
        "pitch_type", "balls", "strikes", "outs_when_up", "inning",
        "on_1b", "on_2b", "on_3b", "score_diff"
    ]]
    pitch_df = pitch_df.rename(columns={"outs_when_up": "outs"})
    data = pitch_df.to_dict(orient="records")
    
    # Upload in batches of 5000 to avoid timeout
    batch_size = 5000
    total = len(data)
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        supabase.table("pitches").insert(batch).execute()
        print(f"Uploaded {min(i + batch_size, total)} / {total}")


if __name__ == "__main__":
    # load_pitchers()
    # print("Pitchers loaded!")
    load_pitches()
    print("Pitches loaded!")