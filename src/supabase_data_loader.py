import os
from typing import Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
import time

load_dotenv()

DEFAULT_PITCH_COLUMNS = [
    "id",
    "pitch_uid",
    "game_id",
    "season",
    "at_bat_index",
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
    "at_bat_number",
    "play_event_index",
    "pitch_number",
    "prev_pitch",
    "count",
]
MAX_RETRIES = 5
BACKOFF_SECONDS = 2


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

    return create_client(url, key)


def execute_with_retry(query_factory, operation_name):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return query_factory().execute()
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(BACKOFF_SECONDS ** attempt)

    raise RuntimeError(f"{operation_name} failed after {MAX_RETRIES} attempts") from last_error


def get_season_high_water_mark(supabase, season):
    response = execute_with_retry(
        lambda: (
            supabase.table("pitches")
            .select("id")
            .eq("season", season)
            .order("id", desc=True)
            .limit(1)
        ),
        operation_name=f"get high-water mark for season {season}",
    )
    if not response.data:
        return None
    return response.data[0]["id"]


def iter_pitches_for_season(season, columns=None, batch_size=1000):
    supabase = get_supabase_client()
    selected_columns = columns or DEFAULT_PITCH_COLUMNS

    last_id = 0
    high_water_mark = get_season_high_water_mark(supabase, season)

    if high_water_mark is None:
        return

    while True:
        response = execute_with_retry(
            lambda: (
                supabase.table("pitches")
                .select(",".join(selected_columns))
                .eq("season", season)
                .gt("id", last_id)
                .lte("id", high_water_mark)
                .order("id")
                .limit(batch_size)
            ),
            operation_name=f"load pitches for season {season}",
        )
        rows = response.data or []

        if not rows:
            break

        last_id = rows[-1]["id"]
        yield pd.DataFrame(rows)

        if len(rows) < batch_size:
            break


def load_pitches_for_season(season, columns=None, batch_size=1000):
    frames = list(iter_pitches_for_season(season=season, columns=columns, batch_size=batch_size))
    if not frames:
        return pd.DataFrame(columns=columns or DEFAULT_PITCH_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def iter_pitches_from_supabase(
    seasons: Optional[Iterable[int]] = None,
    columns=None,
    batch_size: int = 1000,
):
    seasons = list(seasons or [])
    if not seasons:
        raise ValueError("You must provide at least one season.")

    for season in seasons:
        yield from iter_pitches_for_season(
            season=season,
            columns=columns,
            batch_size=batch_size,
        )


def load_pitches_from_supabase(
    seasons: Optional[Iterable[int]] = None,
    columns=None,
    batch_size: int = 1000,
) -> pd.DataFrame:
    seasons = list(seasons or [])
    if not seasons:
        raise ValueError("You must provide at least one season.")

    frames = list(iter_pitches_from_supabase(seasons=seasons, columns=columns, batch_size=batch_size))
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=columns or DEFAULT_PITCH_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    df = load_pitches_from_supabase(seasons=[2023, 2024, 2025])

