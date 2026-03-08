import os
import json
import pandas as pd

RAW_DIR = "data/raw/player_day"
OUTPUT = "data/processed/pitcher_game_logs.csv"

rows = []

for file in os.listdir(RAW_DIR):

    if not file.endswith(".json"):
        continue

    player_id = file.split("_")[0]

    path = os.path.join(RAW_DIR, file)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for game_id, value in data.items():

        if game_id in ["result_cd", "result_msg", "update_time"]:
            continue

        # 투수만
        if value.get("IP") is None and value.get("ERA") is None and value.get("WHIP") is None:
            continue

        row = {
            "game_id": game_id,
            "player_id": player_id,
            "team_code": value.get("t_code"),
            "vs_team": value.get("vs_tCode"),
            "game_date": value.get("gameDate"),

            "G": value.get("G"),
            "GS": value.get("GS"),

            "IP": value.get("IP"),
            "R": value.get("R"),
            "ER": value.get("ER"),
            "TBF": value.get("TBF"),
            "H": value.get("H"),
            "HR": value.get("HR"),
            "BB": value.get("BB"),
            "SO": value.get("SO"),
            "NP": value.get("NP"),

            "ERA": value.get("ERA"),
            "WHIP": value.get("WHIP"),
            "OPS": value.get("OPS"),
        }

        rows.append(row)

df = pd.DataFrame(rows)

numeric_cols = [
    "G", "GS", "IP", "R", "ER", "TBF", "H", "HR", "BB", "SO", "NP",
    "ERA", "WHIP"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT)
print("rows:", len(df))
print(df.head())