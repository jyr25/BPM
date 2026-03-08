import os
import json
import time
import pandas as pd
from api_client import call_api

GAME_INDEX = "data/processed/game_index.csv"
OUT_DIR = "data/raw/lineups"


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_lineup(game_id):

    data = call_api(
        METHOD = "GET",
        PATH   = "prediction/gameLineup",
        QUERY  = {"s_no": game_id}
    )

    return data


def main():

    game_index = pd.read_csv(GAME_INDEX)

    for _, row in game_index.iterrows():

        game_id = str(row["game_id"])

        save_path = f"{OUT_DIR}/{game_id}.json"

        if os.path.exists(save_path):
            print("skip", game_id)
            continue

        data = fetch_lineup(game_id)

        if data:
            save_json(data, save_path)
            print("saved", game_id)

        time.sleep(0.2)


if __name__ == "__main__":
    main()