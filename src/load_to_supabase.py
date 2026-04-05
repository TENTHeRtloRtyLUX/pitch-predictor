import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv
import sys
sys.path.append("src")
from mlb_api import get_all_games_for_season, get_pitches_from_game
import time

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

VALID_PITCHES = ["FF", "SL", "SI", "CH", "FC", "CU", "ST", "FS", "KC"]

def clean_season_data(df, season):
    df = df.rename(columns={"outs": "outs_when_up"})
    df = df[df["pitch_type"].isin(VALID_PITCHES)]
    df["season"] = season
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
    return df

def get_done_games(season):
    all_games = []
    page = 0
    page_size = 1000
    
    while True:
        response = supabase.table("uploaded_games").select("game_id").eq("season", season).range(
            page * page_size, (page + 1) * page_size - 1
        ).execute()
        
        if not response.data:
            break
            
        all_games.extend(response.data)
        page += 1
    
    return set(r["game_id"] for r in all_games)

def mark_games_done(game_ids, season):
    records = [{"game_id": gid, "season": season} for gid in game_ids]
    supabase.table("uploaded_games").upsert(records, on_conflict="game_id").execute()

def upsert_pitchers(df):
    pitcher_df = df[["pitcher_name", "p_throws"]].drop_duplicates()
    pitcher_df = pitcher_df.rename(columns={"pitcher_name": "name", "p_throws": "throws"})
    records = pitcher_df.to_dict(orient="records")
    supabase.table("pitchers").upsert(records, on_conflict="name").execute()

def upload_season(season, batch_size=5000):
    print(f"\nProcessing season {season}...")

    done_games = get_done_games(season)
    print(f"Already uploaded: {len(done_games)} games")

    game_ids = get_all_games_for_season(season)
    remaining = [g for g in game_ids if g not in done_games]
    print(f"Remaining: {len(remaining)} games")

    for i in range(0, len(remaining), 50):
        batch_games = remaining[i:i+50]
        batch_data = []

        for game_id in batch_games:
            try:
                pitches = get_pitches_from_game(game_id)
                pitches["game_id"] = game_id
                batch_data.append(pitches)
            except Exception as e:
                print(f"  Failed game {game_id}: {e}")
                continue

        if not batch_data:
            continue

        batch_df = pd.concat(batch_data, ignore_index=True)
        batch_df = clean_season_data(batch_df, season)
        batch_df = batch_df.rename(columns={"outs_when_up": "outs"})

        cols = ["game_id", "pitcher_name", "batter_name", "p_throws", "stand",
                "pitch_type", "balls", "strikes", "outs", "inning",
                "on_1b", "on_2b", "on_3b", "score_diff", "season",
                "prev_pitch", "count", "pitch_number", "at_bat_number"]
        batch_df = batch_df[cols]

        records = batch_df.to_dict(orient="records")
        for j in range(0, len(records), batch_size):
            chunk = records[j:j+batch_size]
            supabase.table("pitches").upsert(
                chunk, on_conflict="game_id,at_bat_number,pitch_number"
            ).execute()

        upsert_pitchers(batch_df)

        mark_games_done(list(set(batch_games)), season)

        print(f"  Batch {i//50 + 1} — {min(i+50, len(remaining))}/{len(remaining)} games")
        time.sleep(1)

    print(f"Season {season} complete.")

if __name__ == "__main__":
    for season in [2023, 2024, 2025, 2026]:
        upload_season(season)
    print("\nAll done!")