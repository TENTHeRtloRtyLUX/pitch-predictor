import requests
import pandas as pd
import os
import time

BASE_URL = "https://statsapi.mlb.com/api/v1"

def get_games_on_date(date):
    url = f"{BASE_URL}/schedule?sportId=1&date={date}"
    response = requests.get(url)
    data = response.json()
    
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            games.append({
                "game_id": game["gamePk"],
                "home_team": game["teams"]["home"]["team"]["name"],
                "away_team": game["teams"]["away"]["team"]["name"],
                "status": game["status"]["detailedState"]
            })
    
    return games

def get_pitches_from_game(game_id):
    url = f"{BASE_URL.replace('v1', 'v1.1')}/game/{game_id}/feed/live"
    response = requests.get(url)
    data = response.json()
    
    pitches = []
    all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    
    for play in all_plays:
        matchup = play.get("matchup", {})
        pitcher_name = matchup.get("pitcher", {}).get("fullName", "")
        batter_name = matchup.get("batter", {}).get("fullName", "")
        p_throws = matchup.get("pitchHand", {}).get("code", "")
        stand = matchup.get("batSide", {}).get("code", "")
        
        for event in play.get("playEvents", []):
            if event.get("isPitch"):
                pitch_data = event.get("pitchData", {})
                count = event.get("count", {})
                
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
                })
    
    return pd.DataFrame(pitches)

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
    games = get_games_on_date("2024-04-15")
    for g in games:
        print(g)
    
    # Test pitch pull on first game
    print("\nPulling pitches from first game...")
    pitches = get_pitches_from_game(games[0]["game_id"])
    print(pitches.head())
    print("Total pitches:", len(pitches))
    game_ids = get_all_games_for_season(2023)
    print("First 5 game IDs:", game_ids[:5])

    fetch_season_pitches(2023)