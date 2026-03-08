import os
import json
import time
from api_client import call_api

OUT_DIR = "data/raw/team_batting"
YEARS = [2023, 2024, 2025]

os.makedirs(OUT_DIR, exist_ok=True)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_team_batting(year, pe=None):
    query = {
        "m2": "batting",
        "year": str(year),
    }

    if pe is not None:
        query["pe"] = pe

    return call_api(
        METHOD="GET",
        PATH="prediction/teamRecord",
        QUERY=query
    )


def main():
    for year in YEARS:
        path_full = os.path.join(OUT_DIR, f"team_batting_{year}_full.json")
        if not os.path.exists(path_full):
            data = fetch_team_batting(year)
            if data is not None:
                save_json(data, path_full)
                print("saved", path_full)
            time.sleep(0.2)

        path_d30 = os.path.join(OUT_DIR, f"team_batting_{year}_d30.json")
        if not os.path.exists(path_d30):
            data = fetch_team_batting(year, pe="D30")
            if data is not None:
                save_json(data, path_d30)
                print("saved", path_d30)
            time.sleep(0.2)


if __name__ == "__main__":
    main()