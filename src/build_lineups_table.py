import os
import json
import pandas as pd

LINEUP_DIR = "data/raw/lineups"
OUTPUT = "data/processed/lineups_flat.csv"

rows = []

for file in os.listdir(LINEUP_DIR):
    if not file.endswith(".json"):
        continue

    path = os.path.join(LINEUP_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_game_id = file.replace(".json", "")

    # 최상위 dict에서 값이 list인 것만 선수 리스트로 처리
    for top_key, player_list in data.items():
        if not isinstance(player_list, list):
            continue

        # top_key는 팀코드(예: "3001", "6002")
        for p in player_list:
            row = {
                "game_id": str(p.get("s_no", file_game_id)),
                "file_game_id": file_game_id,
                "team_code": p.get("t_code", top_key),
                "player_id": p.get("p_no"),
                "player_name": p.get("p_name"),
                "starting": p.get("starting"),
                "lineup_state": p.get("lineupState"),
                "position": p.get("position"),
                "batting_order": p.get("battingOrder"),
                "bat_type": p.get("p_bat"),
                "throw_type": p.get("p_throw"),
                "back_number": p.get("p_backNumber"),
            }
            rows.append(row)

df = pd.DataFrame(rows)

# 자료형 정리
if not df.empty:
    df["game_id"] = df["game_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")

    # batting_order는 "1"~"9", 투수는 "P"일 수 있음
    # 숫자 타순만 따로 필요하면 아래 컬럼 추가
    df["batting_order_num"] = pd.to_numeric(df["batting_order"], errors="coerce").astype("Int64")

    # 선발 여부 필터용
    df["is_starting"] = df["starting"].eq("Y")

os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT)
print("rows:", len(df))
print(df.head())