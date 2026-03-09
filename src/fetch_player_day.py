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
    if not os.path.exists(PLAYERS_PATH):
        print(f"Error: {PLAYERS_PATH} not found.")
        return

    df = pd.read_csv(PLAYERS_PATH)
    
    player_ids = df["player_id"].dropna().unique()
    print(f"총 수집 대상 선수: {len(player_ids)}명")

    for player_id in player_ids:
        p_id_str = str(player_id)

        for year in YEARS:
            save_path = os.path.join(OUT_DIR, f"{p_id_str}_{year}.json")

            if os.path.exists(save_path):
                
                if os.path.getsize(save_path) > 10:
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
                time.sleep(1)  # API 오류 시 잠시 대기 후 재시도

            time.sleep(0.2)


if __name__ == "__main__":
    main()