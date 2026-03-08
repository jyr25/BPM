import pandas as pd
import numpy as np

LINEUP_PATH = "data/processed/lineups_flat.csv"
STARTER_PATH = "data/processed/starter_game_logs.csv"
OUTPUT_PATH = "data/processed/battery_features.csv"

# -------------------------------------------------
# 1. load
# -------------------------------------------------
lineups = pd.read_csv(LINEUP_PATH)
starter = pd.read_csv(STARTER_PATH)

lineups["game_id"] = lineups["game_id"].astype(str)
lineups["team_code"] = lineups["team_code"].astype(str)
lineups["player_id"] = lineups["player_id"].astype(str)

starter["game_id"] = starter["game_id"].astype(str)
starter["team_code"] = starter["team_code"].astype(str)
starter["player_id"] = starter["player_id"].astype(str)

# 날짜 복구
starter["game_date"] = pd.to_datetime(starter["game_date"], errors="coerce")

# -------------------------------------------------
# 2. 라인업에서 선발투수 / 포수 추출
# -------------------------------------------------
# 선발투수
sp = lineups[
    (lineups["position"] == 1) &
    (lineups["is_starting"] == True)
][["game_id", "team_code", "player_id"]].copy()

sp = sp.rename(columns={"player_id": "sp_id"})

# 포수
catcher = lineups[
    (lineups["position"] == 2) &
    (lineups["is_starting"] == True)
][["game_id", "team_code", "player_id"]].copy()

catcher = catcher.rename(columns={"player_id": "catcher_id"})

# -------------------------------------------------
# 3. 경기별 배터리 조합 생성
# -------------------------------------------------
battery = sp.merge(
    catcher,
    on=["game_id", "team_code"],
    how="inner"
)

# starter_game_logs에서 ERA, WHIP 붙이기
starter_stats = starter[[
    "game_id", "team_code", "player_id", "game_date", "ERA", "WHIP"
]].copy()

starter_stats = starter_stats.rename(columns={"player_id": "sp_id"})

battery = battery.merge(
    starter_stats,
    on=["game_id", "team_code", "sp_id"],
    how="left"
)

battery = battery.sort_values(["sp_id", "catcher_id", "game_date"]).reset_index(drop=True)

# 숫자형
battery["ERA"] = pd.to_numeric(battery["ERA"], errors="coerce")
battery["WHIP"] = pd.to_numeric(battery["WHIP"], errors="coerce")

# -------------------------------------------------
# 4. 과거 배터리 조합 feature
# 현재 경기 제외하고 이전 경기들만 사용
# -------------------------------------------------

group_cols = ["sp_id", "catcher_id"]

# 같이 나온 경기 수 (현재 경기 이전)
battery["battery_games_together"] = (
    battery.groupby(group_cols).cumcount()
)

# 과거 평균 ERA
battery["battery_avg_era"] = (
    battery.groupby(group_cols)["ERA"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# 과거 평균 WHIP
battery["battery_avg_whip"] = (
    battery.groupby(group_cols)["WHIP"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# -------------------------------------------------
# 5. 저장
# -------------------------------------------------
out_cols = [
    "game_id",
    "team_code",
    "sp_id",
    "catcher_id",
    "battery_games_together",
    "battery_avg_era",
    "battery_avg_whip"
]

battery_features = battery[out_cols].copy()

battery_features.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_PATH)
print("rows:", len(battery_features))
print(battery_features.head())
print("\nmissing values:")
print(battery_features.isna().sum())