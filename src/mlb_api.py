import requests
import pandas as pd
import os
import time

BASE_URL = "https://statsapi.mlb.com/api/v1"

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

def get_games_on_date(date):
    """Get all game IDs for a given date (format: YYYY-MM-DD)"""
    url = f"{BASE_URL}/schedule?sportId=1&date={date}"
    response = requests.get(url)
    data = response.json()

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("seriesDescription") in ["Regular Season", "Spring Training", "Wild Card", "Division Series", "Championship Series", "World Series"]:
                games.append({
                    "game_id": game["gamePk"],
                    "home_team": game["teams"]["home"]["team"]["name"],
                    "away_team": game["teams"]["away"]["team"]["name"],
                    "status": game["status"]["detailedState"]
                })

    return games

def get_pitches_from_game(game_id):
    """Get all pitches from a game with score differential"""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    response = requests.get(url)
    data = response.json()

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
        is_home = play.get("about", {}).get("isTopInning", True) == False

        for event in play.get("playEvents", []):
            if event.get("isPitch"):
                count = event.get("count", {})

                if is_home:
                    score_diff = home_score - away_score
                else:
                    score_diff = away_score - home_score

                pitches.append({
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
                })

    return pd.DataFrame(pitches)

def get_live_game_state(game_id):
    """Get current state of a live or recent game"""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
    response = requests.get(url)
    data = response.json()

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
    is_home = current_play.get("about", {}).get("isTopInning", True) == False

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
    for event in current_play.get("playEvents", []):
        if event.get("isPitch"):
            recent_pitches.append({
                "pitch_type": PITCH_NAMES.get(
                    event.get("details", {}).get("type", {}).get("code", ""),
                    event.get("details", {}).get("type", {}).get("code", "")
                ),
                "description": event.get("details", {}).get("description", ""),
                "speed": event.get("pitchData", {}).get("startSpeed", ""),
            })

    state["recent_pitches"] = recent_pitches
    state["prev_pitch"] = recent_pitches[-1]["pitch_type"] if recent_pitches else "FF"

    return state

def get_all_games_for_season(year):
    """Get all game IDs for a full season"""
    url = f"{BASE_URL}/schedule?sportId=1&season={year}&gameType=R"
    response = requests.get(url)
    data = response.json()

    game_ids = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game["status"]["detailedState"] == "Final":
                game_ids.append(game["gamePk"])

    print(f"Found {len(game_ids)} completed games in {year}")
    return game_ids

def fetch_season_pitches(year, batch_size=50):
    """Pull pitch data for a full season in batches, saving progress"""
    game_ids = get_all_games_for_season(year)

    out_path = f"data/season_{year}_pitches.csv"

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        done_games = set(existing["game_id"].unique())
        print(f"Resuming — {len(done_games)} games already pulled")
    else:
        existing = pd.DataFrame()
        done_games = set()

    remaining = [g for g in game_ids if g not in done_games]
    print(f"{len(remaining)} games remaining")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i+batch_size]
        batch_data = []

        for game_id in batch:
            try:
                pitches = get_pitches_from_game(game_id)
                pitches["game_id"] = game_id
                batch_data.append(pitches)
            except Exception as e:
                print(f"  Failed game {game_id}: {e}")
                continue

        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            existing = pd.concat([existing, batch_df], ignore_index=True)
            existing.to_csv(out_path, index=False)
            print(f"Saved batch {i//batch_size + 1} — {len(existing)} total pitches")

        time.sleep(1)

    print(f"Done! {len(existing)} total pitches saved to {out_path}")
    return existing

if __name__ == "__main__":
    fetch_season_pitches(2023)