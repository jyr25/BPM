import os
import json
import time
from api_client import call_api

OUT_DIR = "data/raw/roster"

# 시즌별 대표 날짜 (월 1일 기준)
DATES = [
    "2023-04-01","2023-05-01","2023-06-01","2023-07-01","2023-08-01","2023-09-01","2023-10-01",
    "2024-04-01","2024-05-01","2024-06-01","2024-07-01","2024-08-01","2024-09-01","2024-10-01",
    "2025-04-01","2025-05-01","2025-06-01","2025-07-01","2025-08-01","2025-09-01"
]

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_roster(date):

    data = call_api(
        METHOD="GET",
        PATH="prediction/playerRoster",
        QUERY={"date": date}
    )

    return data


def main():

    for date in DATES:

        save_path = f"{OUT_DIR}/{date}.json"

        if os.path.exists(save_path):
            print("skip", date)
            continue

        data = fetch_roster(date)

        if data:
            save_json(data, save_path)
            print("saved", date)

        time.sleep(0.2)


if __name__ == "__main__":
    main()