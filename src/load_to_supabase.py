import pandas as pd
from supabase import create_client
import os
from dotenv import load_dotenv
import sys
sys.path.append("src")
from mlb_api import get_all_games_for_season, get_pitches_from_game
import time
from prepare_full_data import prepare_pitch_data

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

VALID_PITCHES = ["FF", "SL", "SI", "CH", "FC", "CU", "ST", "FS", "KC"]
MAX_RETRIES = 5
BACKOFF_SECONDS = 2


def execute_with_retry(query_factory, operation_name):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return query_factory().execute()
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            sleep_seconds = BACKOFF_SECONDS ** attempt
            print(f"{operation_name} failed on attempt {attempt}/{MAX_RETRIES}: {exc}. Retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"{operation_name} failed after {MAX_RETRIES} attempts") from last_error


def clean_season_data(df, season):
    if df.empty:
        return df

    prepared_df = prepare_pitch_data(df, valid_pitches=VALID_PITCHES)
    prepared_df["season"] = season
    return prepared_df

def get_done_games(season):
    all_games = []
    page = 0
    page_size = 1000
    
    while True:
        response = execute_with_retry(
            lambda: supabase.table("uploaded_games").select("game_id").eq("season", season).range(
                page * page_size, (page + 1) * page_size - 1
            ),
            operation_name=f"load uploaded games for season {season}",
        )
        
        if not response.data:
            break
            
        all_games.extend(response.data)
        page += 1
    
    return set(r["game_id"] for r in all_games)

def mark_games_done(game_ids, season):
    records = [{"game_id": gid, "season": season} for gid in game_ids]
    execute_with_retry(
        lambda: supabase.table("uploaded_games").upsert(records, on_conflict="game_id"),
        operation_name=f"mark uploaded games for season {season}",
    )

def upsert_pitchers(df):
    pitcher_df = df[["pitcher_name", "p_throws"]].drop_duplicates()
    pitcher_df = pitcher_df.rename(columns={"pitcher_name": "name", "p_throws": "throws"})
    records = pitcher_df.to_dict(orient="records")
    execute_with_retry(
        lambda: supabase.table("pitchers").upsert(records, on_conflict="name"),
        operation_name="upsert pitchers",
    )

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

        if batch_df.empty:
            continue

        cols = [
            "pitch_uid",
            "game_id",
            "season",
            "at_bat_index",
            "at_bat_number",
            "play_event_index",
            "pitch_number",
            "pitcher_name",
            "batter_name",
            "p_throws",
            "stand",
            "pitch_type",
            "balls",
            "strikes",
            "outs",
            "inning",
            "on_1b",
            "on_2b",
            "on_3b",
            "score_diff",
            "prev_pitch",
            "count",
        ]
        batch_df = batch_df[cols]

        records = batch_df.to_dict(orient="records")
        for j in range(0, len(records), batch_size):
            chunk = records[j:j+batch_size]
            execute_with_retry(
                lambda chunk=chunk: supabase.table("pitches").upsert(
                    chunk, on_conflict="pitch_uid"
                ),
                operation_name=f"upsert pitches for season {season}",
            )

        upsert_pitchers(batch_df)

        mark_games_done(list(set(batch_games)), season)

        print(f"  Batch {i//50 + 1} — {min(i+50, len(remaining))}/{len(remaining)} games")
        time.sleep(1)

    print(f"Season {season} complete.")

if __name__ == "__main__":
    for season in [2023, 2024, 2025, 2026]:
        upload_season(season)
    print("\nAll done!")
