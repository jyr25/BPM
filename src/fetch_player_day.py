import os
import json
import time
import pandas as pd
from api_client import call_api

PLAYERS_PATH = "data/processed/all_players.csv"
OUT_DIR = "data/raw/player_day"
YEARS = [2023, 2024, 2025]

os.makedirs(OUT_DIR, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_player_day(player_id: str, year: int):
    return call_api(
        METHOD="GET",
        PATH="prediction/playerDay",
        QUERY={
            "p_no": str(player_id),
            "year": str(year)
        }
    )


def main():
    df = pd.read_csv(PLAYERS_PATH)
    
    player_ids = (
       df["player_id"]
       .dropna()
       .astype(str)
       .unique()
    )

    print("선수 수:", len(player_ids))

    for player_id in player_ids:
        for year in YEARS:
            save_path = os.path.join(OUT_DIR, f"{player_id}_{year}.json")

            if os.path.exists(save_path):
                print("skip", player_id, year)
                continue

            try:
                data = fetch_player_day(player_id, year)

                if data is not None:
                    save_json(data, save_path)
                    print("saved", player_id, year)
                else:
                    print("failed", player_id, year)

            except Exception as e:
                print("error", player_id, year, e)

            time.sleep(0.2)


if __name__ == "__main__":
    main()