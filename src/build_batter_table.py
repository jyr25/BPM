import os
import json
import pandas as pd

RAW_DIR = "data/raw/player_day"
OUTPUT = "data/processed/batter_game_logs.csv"

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

        # 타자만
        if value.get("battingOrder") is None:
            continue

        row = {
            "game_id": game_id,
            "player_id": player_id,
            "team_code": value.get("t_code"),

            "PA": value.get("PA"),
            "AB": value.get("AB"),
            "H": value.get("H"),
            "HR": value.get("HR"),
            "RBI": value.get("RBI"),
            "BB": value.get("BB"),
            "SO": value.get("SO"),

            "R": value.get("R"),         
            "H2": value.get("H2"),        
            "H3": value.get("H3"),       
            "HBP": value.get("HBP"),     
            "SB": value.get("SB"),        
            "CS": value.get("CS"),        
            "SF": value.get("SF"),        

            "AVG": value.get("AVG"),
            "OBP": value.get("OBP"),
            "SLG": value.get("SLG"),
            "OPS": value.get("OPS"),
        }

        rows.append(row)

df = pd.DataFrame(rows)

numeric_cols = [
    "PA","AB","H","HR","RBI","BB","SO",
    "R","H2","H3","HBP","SB","CS","SF",
    "AVG","OBP","SLG","OPS"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT)
print("rows:", len(df))