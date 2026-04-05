import os
from typing import Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

DEFAULT_PITCH_COLUMNS = [
    "id",
    "game_id",
    "season",
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
    "pitch_number",
    "prev_pitch",
    "count",
]


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

    return create_client(url, key)


def load_pitches_for_season(season, columns=None, batch_size=1000):
    supabase = get_supabase_client()
    selected_columns = columns or DEFAULT_PITCH_COLUMNS

    all_rows = []
    last_id = 0

    while True:
        query = (
            supabase.table("pitches")
            .select(",".join(selected_columns))
            .eq("season", season)
            .gt("id", last_id)
            .order("id")
            .limit(batch_size)
        )

        response = query.execute()
        rows = response.data or []

        if not rows:
            break

        all_rows.extend(rows)
        last_id = rows[-1]["id"]


        if len(rows) < batch_size:
            break

    df = pd.DataFrame(all_rows)
    return df


def load_pitches_from_supabase(
    seasons: Optional[Iterable[int]] = None,
    columns=None,
    batch_size: int = 1000,
) -> pd.DataFrame:
    seasons = list(seasons or [])
    if not seasons:
        raise ValueError("You must provide at least one season.")

    frames = [
        load_pitches_for_season(
            season=season,
            columns=columns,
            batch_size=batch_size,
        )
        for season in seasons
    ]

    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=columns or DEFAULT_PITCH_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    return combined


if __name__ == "__main__":
    df = load_pitches_from_supabase(seasons=[2023, 2024, 2025])

