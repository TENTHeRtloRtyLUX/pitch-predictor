from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://statsapi.mlb.com/api/v1"
LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5
ALLOWED_SERIES = {
    "Regular Season",
    "Spring Training",
    "Wild Card",
    "Division Series",
    "Championship Series",
    "World Series",
}

PITCH_NAMES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "KN": "Knuckleball",
    "FA": "Fastball",
    "FO": "Forkball",
}


def build_retry_session():
    retry = Retry(
        total=MAX_RETRIES,
        read=MAX_RETRIES,
        connect=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = build_retry_session()


def fetch_json(url):
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def normalize_game_record(game, game_date):
    return {
        "game_id": game["gamePk"],
        "game_date": game_date,
        "season": datetime.strptime(game_date, "%Y-%m-%d").year,
        "game_type": game.get("gameType"),
        "home_team": game["teams"]["home"]["team"]["name"],
        "away_team": game["teams"]["away"]["team"]["name"],
        "status": game["status"]["detailedState"],
        "series_description": game.get("seriesDescription", ""),
    }


def iter_games_from_schedule(data, include_statuses=None):
    include_statuses = set(include_statuses or [])
    for date_entry in data.get("dates", []):
        game_date = date_entry["date"]
        for game in date_entry.get("games", []):
            if game.get("seriesDescription") not in ALLOWED_SERIES:
                continue
            detailed_state = game["status"]["detailedState"]
            if include_statuses and detailed_state not in include_statuses:
                continue
            yield normalize_game_record(game, game_date)


def get_games_on_date(date):
    url = f"{BASE_URL}/schedule?sportId=1&date={date}"
    data = fetch_json(url)
    return list(iter_games_from_schedule(data))


def get_games_in_date_range(start_date, end_date, game_type="R", include_statuses=None):
    url = (
        f"{BASE_URL}/schedule?sportId=1"
        f"&startDate={start_date}&endDate={end_date}&gameType={game_type}"
    )
    data = fetch_json(url)
    return list(iter_games_from_schedule(data, include_statuses=include_statuses or {"Final"}))


def last_day_of_month(day):
    next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month - timedelta(days=1)


def get_games_in_date_range_chunked(start_date, end_date, game_type="R", include_statuses=None):
    start_day = datetime.strptime(str(start_date), "%Y-%m-%d").date()
    end_day = datetime.strptime(str(end_date), "%Y-%m-%d").date()

    all_games = []
    current_day = start_day
    while current_day <= end_day:
        chunk_end = min(last_day_of_month(current_day), end_day)
        all_games.extend(
            get_games_in_date_range(
                start_date=current_day.isoformat(),
                end_date=chunk_end.isoformat(),
                game_type=game_type,
                include_statuses=include_statuses,
            )
        )
        current_day = chunk_end + timedelta(days=1)

    deduped_games = {}
    for game in all_games:
        deduped_games[game["game_id"]] = game

    return sorted(
        deduped_games.values(),
        key=lambda game: (game["game_date"], game["game_id"]),
    )


def get_pitches_from_game(game_id):
    url = f"{LIVE_FEED_URL}/{game_id}/feed/live"
    data = fetch_json(url)

    pitches = []
    all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

    home_score = 0
    away_score = 0

    for play in all_plays:
        result = play.get("result", {})
        if result.get("homeScore") is not None:
            home_score = result.get("homeScore", 0)
            away_score = result.get("awayScore", 0)

        matchup = play.get("matchup", {})
        pitcher_name = matchup.get("pitcher", {}).get("fullName", "")
        batter_name = matchup.get("batter", {}).get("fullName", "")
        p_throws = matchup.get("pitchHand", {}).get("code", "")
        stand = matchup.get("batSide", {}).get("code", "")
        is_home = play.get("about", {}).get("isTopInning", True) is False
        at_bat_index = play.get("about", {}).get("atBatIndex")
        if at_bat_index is None:
            continue

        pitch_number = 0

        for play_event_index, event in enumerate(play.get("playEvents", [])):
            if not event.get("isPitch"):
                continue

            count = event.get("count", {})
            pitch_number += 1

            if is_home:
                score_diff = home_score - away_score
            else:
                score_diff = away_score - home_score

            pitches.append(
                {
                    "pitch_uid": f"{game_id}:{at_bat_index}:{play_event_index}",
                    "game_id": game_id,
                    "at_bat_index": at_bat_index,
                    "at_bat_number": at_bat_index + 1,
                    "play_event_index": play_event_index,
                    "pitch_number": pitch_number,
                    "pitcher_name": pitcher_name,
                    "batter_name": batter_name,
                    "p_throws": p_throws,
                    "stand": stand,
                    "pitch_type": event.get("details", {}).get("type", {}).get("code", ""),
                    "balls": count.get("balls", 0),
                    "strikes": count.get("strikes", 0),
                    "outs": count.get("outs", 0),
                    "inning": play.get("about", {}).get("inning", 0),
                    "on_1b": int(bool(play.get("matchup", {}).get("postOnFirst"))),
                    "on_2b": int(bool(play.get("matchup", {}).get("postOnSecond"))),
                    "on_3b": int(bool(play.get("matchup", {}).get("postOnThird"))),
                    "score_diff": score_diff,
                }
            )

    pitch_df = pd.DataFrame(pitches)
    if pitch_df.empty:
        return pitch_df

    return pitch_df.sort_values(
        ["game_id", "at_bat_index", "play_event_index"]
    ).reset_index(drop=True)


def get_live_game_state(game_id):
    url = f"{LIVE_FEED_URL}/{game_id}/feed/live"
    data = fetch_json(url)

    live_data = data.get("liveData", {})
    plays = live_data.get("plays", {})
    current_play = plays.get("currentPlay", {})

    if not current_play:
        return None

    matchup = current_play.get("matchup", {})
    count = current_play.get("count", {})
    linescore = live_data.get("linescore", {})

    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)
    is_home = current_play.get("about", {}).get("isTopInning", True) is False

    if is_home:
        score_diff = home_score - away_score
    else:
        score_diff = away_score - home_score

    state = {
        "pitcher": matchup.get("pitcher", {}).get("fullName", ""),
        "batter": matchup.get("batter", {}).get("fullName", ""),
        "p_throws": matchup.get("pitchHand", {}).get("code", ""),
        "stand": matchup.get("batSide", {}).get("code", ""),
        "balls": count.get("balls", 0),
        "strikes": count.get("strikes", 0),
        "outs": count.get("outs", 0),
        "inning": current_play.get("about", {}).get("inning", 0),
        "inning_half": current_play.get("about", {}).get("halfInning", ""),
        "on_1b": int(bool(linescore.get("offense", {}).get("first"))),
        "on_2b": int(bool(linescore.get("offense", {}).get("second"))),
        "on_3b": int(bool(linescore.get("offense", {}).get("third"))),
        "home_score": home_score,
        "away_score": away_score,
        "score_diff": score_diff,
    }

    recent_pitches = []
    for play_event_index, event in enumerate(current_play.get("playEvents", [])):
        if not event.get("isPitch"):
            continue
        pitch_code = event.get("details", {}).get("type", {}).get("code", "")
        event_count = event.get("count", {})
        recent_pitches.append(
            {
                "pitch_type": PITCH_NAMES.get(pitch_code, pitch_code),
                "pitch_code": pitch_code,
                "description": event.get("details", {}).get("description", ""),
                "speed": event.get("pitchData", {}).get("startSpeed", ""),
                "balls": event_count.get("balls", 0),
                "strikes": event_count.get("strikes", 0),
                "outs": event_count.get("outs", 0),
                "play_event_index": play_event_index,
            }
        )

    state["recent_pitches"] = recent_pitches
    state["prev_pitch"] = recent_pitches[-1]["pitch_type"] if recent_pitches else "FF"
    return state


def get_all_games_for_season(year):
    url = f"{BASE_URL}/schedule?sportId=1&season={year}&gameType=R"
    data = fetch_json(url)
    game_records = list(iter_games_from_schedule(data, include_statuses={"Final"}))
    print(f"Found {len(game_records)} completed games in {year}")
    return [game["game_id"] for game in game_records]


def fetch_season_pitches(year, batch_size=50):
    game_ids = get_all_games_for_season(year)
    all_batches = []

    for start_idx in range(0, len(game_ids), batch_size):
        batch = game_ids[start_idx : start_idx + batch_size]
        batch_data = []

        for game_id in batch:
            try:
                pitches = get_pitches_from_game(game_id)
                if not pitches.empty:
                    batch_data.append(pitches)
            except Exception as exc:
                print(f"Failed game {game_id}: {exc}")

        if batch_data:
            all_batches.append(pd.concat(batch_data, ignore_index=True))

    if not all_batches:
        return pd.DataFrame()

    return pd.concat(all_batches, ignore_index=True)


if __name__ == "__main__":
    fetch_season_pitches(2023)
