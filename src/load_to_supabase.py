import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append("src")
from mlb_api import (
    get_all_games_for_season,
    get_games_in_date_range_chunked,
    get_pitches_from_game,
)
from prepare_full_data import prepare_pitch_data
from supabase_client import get_supabase_service_client

load_dotenv()

VALID_PITCHES = ["FF", "SL", "SI", "CH", "FC", "CU", "ST", "FS", "KC"]
MAX_RETRIES = 5
BACKOFF_SECONDS = 2
DEFAULT_PIPELINE_NAME = "pitch_ingest"
DEFAULT_OVERLAP_DAYS = 3
DEFAULT_BATCH_SIZE = 5000
DEFAULT_GAME_BATCH_SIZE = 25
DEFAULT_MAX_FAILURE_RATE = 0.1
DEFAULT_MAX_FAILED_GAMES = 5
OUTPUT_DIR = Path("output")
LOCK_PATH = OUTPUT_DIR / f"{DEFAULT_PIPELINE_NAME}.lock"
FALLBACK_STATE_PATH = OUTPUT_DIR / f"{DEFAULT_PIPELINE_NAME}_state.json"
SUMMARY_PATH = OUTPUT_DIR / f"{DEFAULT_PIPELINE_NAME}_latest_summary.json"
PITCH_COLUMNS = [
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


def get_supabase_client():
    """Get service role client for upsert/system operations.
    
    Note: This uses the service role key which bypasses RLS policies.
    This is appropriate for:
    - System operations (upserting data)
    - Managing pipeline state
    - Operations that need unrestricted access
    
    Read-only user queries should use get_supabase_authenticated_client()
    if RLS protection is required.
    """
    return get_supabase_service_client()


supabase = get_supabase_client()


def utc_now():
    return datetime.now(timezone.utc)


def utc_today():
    return utc_now().date()


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
            print(
                f"{operation_name} failed on attempt {attempt}/{MAX_RETRIES}: "
                f"{exc}. Retrying in {sleep_seconds}s..."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"{operation_name} failed after {MAX_RETRIES} attempts") from last_error


def clean_pitch_data(df, season):
    if df.empty:
        return df

    prepared_df = prepare_pitch_data(df, valid_pitches=VALID_PITCHES)
    if prepared_df.empty:
        return prepared_df

    prepared_df["season"] = season
    return prepared_df.reindex(columns=PITCH_COLUMNS)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def read_json_file(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json_file(path, payload):
    ensure_output_dir()
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def read_pipeline_state_from_supabase(pipeline_name):
    response = execute_with_retry(
        lambda: (
            supabase.table("pipeline_state")
            .select("*")
            .eq("pipeline_name", pipeline_name)
            .limit(1)
        ),
        operation_name=f"load pipeline state for {pipeline_name}",
    )
    rows = response.data or []
    return rows[0] if rows else None


def write_pipeline_state_to_supabase(state):
    execute_with_retry(
        lambda: supabase.table("pipeline_state").upsert(state, on_conflict="pipeline_name"),
        operation_name=f"save pipeline state for {state['pipeline_name']}",
    )


def read_pipeline_state(pipeline_name):
    try:
        state = read_pipeline_state_from_supabase(pipeline_name)
        if state:
            return state, "supabase"
    except Exception as exc:
        print(f"Falling back to local pipeline state because Supabase state read failed: {exc}")

    state = read_json_file(FALLBACK_STATE_PATH) or {}
    if state.get("pipeline_name") == pipeline_name:
        return state, "json"

    return {
        "pipeline_name": pipeline_name,
        "last_successful_ingest_utc": None,
        "last_ingested_game_date": None,
        "overlap_days": DEFAULT_OVERLAP_DAYS,
    }, "json"


def write_pipeline_state(state, preferred_backend):
    if preferred_backend == "supabase":
        try:
            write_pipeline_state_to_supabase(state)
            write_json_file(FALLBACK_STATE_PATH, state)
            return "supabase"
        except Exception as exc:
            print(f"Supabase state write failed; saving local fallback instead: {exc}")

    write_json_file(FALLBACK_STATE_PATH, state)
    return "json"


@contextmanager
def fail_fast_lock(lock_path=LOCK_PATH):
    ensure_output_dir()
    if lock_path.exists():
        raise RuntimeError(f"Another ingest run appears active because {lock_path} already exists.")

    lock_payload = {
        "pipeline_name": DEFAULT_PIPELINE_NAME,
        "acquired_utc": utc_now().isoformat(),
        "pid": os.getpid(),
    }
    write_json_file(lock_path, lock_payload)

    try:
        yield
    finally:
        if lock_path.exists():
            lock_path.unlink()


def parse_date(value):
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def infer_default_season_start():
    today = utc_today()
    return date(today.year, 3, 1)


def compute_window(state, season_start, overlap_days, full_refresh):
    season_start = parse_date(season_start) or infer_default_season_start()
    today = utc_today()

    if full_refresh or not state.get("last_ingested_game_date"):
        start_date = season_start
    else:
        last_ingested_game_date = parse_date(state["last_ingested_game_date"])
        start_date = max(last_ingested_game_date - timedelta(days=overlap_days), season_start)

    return start_date, today


def upsert_pitchers(df):
    if df.empty:
        return 0

    pitcher_df = df[["pitcher_name", "p_throws"]].drop_duplicates()
    pitcher_df = pitcher_df.rename(columns={"pitcher_name": "name", "p_throws": "throws"})
    records = pitcher_df.to_dict(orient="records")
    if not records:
        return 0

    execute_with_retry(
        lambda: supabase.table("pitchers").upsert(records, on_conflict="name"),
        operation_name="upsert pitchers",
    )
    return len(records)


def upsert_pitch_records(df, batch_size):
    if df.empty:
        return 0

    deduped_df = (
        df.sort_values(["game_id", "at_bat_index", "play_event_index"])
        .drop_duplicates(subset=["pitch_uid"], keep="last")
        .reset_index(drop=True)
    )
    records = deduped_df.to_dict(orient="records")
    rows_written = 0
    for start_idx in range(0, len(records), batch_size):
        chunk = records[start_idx : start_idx + batch_size]
        execute_with_retry(
            lambda chunk=chunk: supabase.table("pitches").upsert(chunk, on_conflict="pitch_uid"),
            operation_name="upsert pitches",
        )
        rows_written += len(chunk)

    return rows_written


def summarize_failures(games_attempted, failed_games, max_failed_games, max_failure_rate):
    failed_count = len(failed_games)
    failure_rate = failed_count / games_attempted if games_attempted else 0
    too_many_failed_games = failed_count > max_failed_games
    too_high_failure_rate = games_attempted > 0 and failure_rate > max_failure_rate
    return failed_count, failure_rate, too_many_failed_games or too_high_failure_rate


def ingest_games(games, game_batch_size, pitch_batch_size):
    game_failures = []
    rows_upserted = 0
    pitcher_rows_upserted = 0
    max_game_date_seen = None
    processed_game_ids = []

    for start_idx in range(0, len(games), game_batch_size):
        batch_games = games[start_idx : start_idx + game_batch_size]
        batch_frames = []

        for game in batch_games:
            game_id = game["game_id"]
            game_date = game["game_date"]
            season = game["season"]
            try:
                pitches = get_pitches_from_game(game_id)
                cleaned_df = clean_pitch_data(pitches, season=season)
                if cleaned_df.empty:
                    processed_game_ids.append(game_id)
                    max_game_date_seen = max(max_game_date_seen or parse_date(game_date), parse_date(game_date))
                    continue

                batch_frames.append(cleaned_df)
                processed_game_ids.append(game_id)
                max_game_date_seen = max(max_game_date_seen or parse_date(game_date), parse_date(game_date))
            except Exception as exc:
                game_failures.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date,
                        "error": str(exc),
                    }
                )

        if not batch_frames:
            continue

        batch_df = pd.concat(batch_frames, ignore_index=True)
        rows_upserted += upsert_pitch_records(batch_df, batch_size=pitch_batch_size)
        pitcher_rows_upserted += upsert_pitchers(batch_df)
        print(
            f"Processed {min(start_idx + game_batch_size, len(games))}/{len(games)} games "
            f"with {len(batch_df)} cleaned pitch rows."
        )

    return {
        "rows_upserted": rows_upserted,
        "pitchers_upserted": pitcher_rows_upserted,
        "failed_games": game_failures,
        "processed_game_ids": processed_game_ids,
        "max_game_date_seen": max_game_date_seen.isoformat() if max_game_date_seen else None,
    }


def build_summary(
    pipeline_name,
    state_before,
    start_date,
    end_date,
    overlap_days,
    games,
    ingest_result,
    success,
    state_backend,
    dry_run=False,
):
    failed_count, failure_rate, _ = summarize_failures(
        games_attempted=len(games),
        failed_games=ingest_result.get("failed_games", []),
        max_failed_games=DEFAULT_MAX_FAILED_GAMES,
        max_failure_rate=DEFAULT_MAX_FAILURE_RATE,
    )
    return {
        "pipeline_name": pipeline_name,
        "success": success,
        "dry_run": dry_run,
        "state_backend": state_backend,
        "window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "overlap_days": overlap_days,
        },
        "games_attempted": len(games),
        "games_failed": failed_count,
        "failure_rate": failure_rate,
        "failed_games": ingest_result.get("failed_games", []),
        "rows_upserted": ingest_result.get("rows_upserted", 0),
        "pitchers_upserted": ingest_result.get("pitchers_upserted", 0),
        "processed_game_ids": ingest_result.get("processed_game_ids", []),
        "watermark_before": state_before.get("last_ingested_game_date"),
        "watermark_after": ingest_result.get("max_game_date_seen") or state_before.get("last_ingested_game_date"),
        "last_successful_ingest_utc_before": state_before.get("last_successful_ingest_utc"),
        "generated_utc": utc_now().isoformat(),
    }


def print_summary(summary):
    print(json.dumps(summary, indent=2))


def run_incremental_ingest(
    pipeline_name=DEFAULT_PIPELINE_NAME,
    season_start=None,
    overlap_days=DEFAULT_OVERLAP_DAYS,
    dry_run=False,
    full_refresh=False,
    game_batch_size=DEFAULT_GAME_BATCH_SIZE,
    pitch_batch_size=DEFAULT_BATCH_SIZE,
    max_failed_games=DEFAULT_MAX_FAILED_GAMES,
    max_failure_rate=DEFAULT_MAX_FAILURE_RATE,
):
    state_before, state_backend = read_pipeline_state(pipeline_name)
    start_date, end_date = compute_window(
        state=state_before,
        season_start=season_start,
        overlap_days=overlap_days,
        full_refresh=full_refresh,
    )
    games = get_games_in_date_range_chunked(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        game_type="R",
        include_statuses={"Final"},
    )

    if dry_run:
        dry_summary = build_summary(
            pipeline_name=pipeline_name,
            state_before=state_before,
            start_date=start_date,
            end_date=end_date,
            overlap_days=overlap_days,
            games=games,
            ingest_result={},
            success=True,
            state_backend=state_backend,
            dry_run=True,
        )
        write_json_file(SUMMARY_PATH, dry_summary)
        print_summary(dry_summary)
        return dry_summary

    with fail_fast_lock():
        ingest_result = ingest_games(
            games=games,
            game_batch_size=game_batch_size,
            pitch_batch_size=pitch_batch_size,
        )
        failed_count, failure_rate, should_fail = summarize_failures(
            games_attempted=len(games),
            failed_games=ingest_result["failed_games"],
            max_failed_games=max_failed_games,
            max_failure_rate=max_failure_rate,
        )

        if should_fail:
            summary = build_summary(
                pipeline_name=pipeline_name,
                state_before=state_before,
                start_date=start_date,
                end_date=end_date,
                overlap_days=overlap_days,
                games=games,
                ingest_result=ingest_result,
                success=False,
                state_backend=state_backend,
            )
            summary["failure_reason"] = (
                f"Too many game failures: {failed_count} failures "
                f"({failure_rate:.1%}) in {len(games)} attempted games."
            )
            write_json_file(SUMMARY_PATH, summary)
            print_summary(summary)
            raise RuntimeError(summary["failure_reason"])

        next_state = {
            "pipeline_name": pipeline_name,
            "last_successful_ingest_utc": utc_now().isoformat(),
            "last_ingested_game_date": (
                ingest_result["max_game_date_seen"] or state_before.get("last_ingested_game_date")
            ),
            "overlap_days": overlap_days,
        }
        state_backend = write_pipeline_state(next_state, preferred_backend=state_backend)

        summary = build_summary(
            pipeline_name=pipeline_name,
            state_before=state_before,
            start_date=start_date,
            end_date=end_date,
            overlap_days=overlap_days,
            games=games,
            ingest_result=ingest_result,
            success=True,
            state_backend=state_backend,
        )
        summary["final_watermark"] = next_state["last_ingested_game_date"]
        write_json_file(SUMMARY_PATH, summary)
        print_summary(summary)
        return summary


def upload_season(season, batch_size=DEFAULT_BATCH_SIZE):
    game_ids = get_all_games_for_season(season)
    games = [
        {"game_id": game_id, "game_date": f"{season}-12-31", "season": season}
        for game_id in game_ids
    ]
    with fail_fast_lock():
        return ingest_games(
            games=games,
            game_batch_size=DEFAULT_GAME_BATCH_SIZE,
            pitch_batch_size=batch_size,
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Incrementally load MLB pitch data into Supabase.")
    parser.add_argument("--pipeline-name", default=DEFAULT_PIPELINE_NAME)
    parser.add_argument("--season-start", default=None, help="Season start date in YYYY-MM-DD format.")
    parser.add_argument("--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS)
    parser.add_argument("--dry-run", action="store_true", help="Compute ingest window and game count only.")
    parser.add_argument("--full-refresh", action="store_true", help="Ignore watermark and rescan from season start.")
    parser.add_argument("--game-batch-size", type=int, default=DEFAULT_GAME_BATCH_SIZE)
    parser.add_argument("--pitch-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-failed-games", type=int, default=DEFAULT_MAX_FAILED_GAMES)
    parser.add_argument("--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_incremental_ingest(
        pipeline_name=args.pipeline_name,
        season_start=args.season_start,
        overlap_days=args.overlap_days,
        dry_run=args.dry_run,
        full_refresh=args.full_refresh,
        game_batch_size=args.game_batch_size,
        pitch_batch_size=args.pitch_batch_size,
        max_failed_games=args.max_failed_games,
        max_failure_rate=args.max_failure_rate,
    )


if __name__ == "__main__":
    main()
