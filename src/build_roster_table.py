import os
import json
import pandas as pd

RAW_DIR = "data/raw/roster"
OUT_FILE = "data/processed/all_players.csv"

rows = []

for file in os.listdir(RAW_DIR):
    if not file.endswith(".json"):
        continue

    path = os.path.join(RAW_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        # 메타데이터 제외
        if key in ["result_cd", "result_msg", "update_time"]:
            continue

        # 선수 정보 dict만 처리
        if not isinstance(value, dict):
            continue

        rows.append({
            "player_id": value.get("p_no"),
            "player_name": value.get("name"),
            "team_code": value.get("t_code"),
            "pj_date": value.get("pj_date"),
        })

df = pd.DataFrame(rows)

if not df.empty:
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)
    df = df.drop_duplicates(subset=["player_id"]).reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

print("saved:", OUT_FILE)
print("players:", len(df))
print(df.head())

df = df.drop_duplicates(subset=["player_id"])

df.to_csv(OUT_FILE, index=False)

print("players:", len(df))