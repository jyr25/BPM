import pandas as pd
import numpy as np

# 경로 설정 유지
LINEUP_PATH = "data/processed/lineups_flat.csv"
STARTER_LOG_PATH = "data/processed/starter_game_logs.csv"
STARTER_FEAT_PATH = "data/processed/starter_features.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv" # 날짜 정렬용 추가
OUTPUT_PATH = "data/processed/battery_features.csv"


# 1. 로드 및 날짜 병합
lineups = pd.read_csv(LINEUP_PATH)
starter_logs = pd.read_csv(STARTER_LOG_PATH)
starter_feats = pd.read_csv(STARTER_FEAT_PATH)
games = pd.read_csv(GAME_INDEX_PATH)[["game_id", "date"]]

for df in [lineups, starter_logs, starter_feats, games]:
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

# ID 정리 및 날짜 추가
for df in [lineups, starter_logs, starter_feats]:
    df["game_id"] = df["game_id"].astype(str)
    if "team_code" in df.columns:
        df["team_code"] = df["team_code"].astype(str)

# 2. 선발/포수 추출 및 결합
# position 1: 투수, 2: 포수
sp = lineups[(lineups["position"] == 1) & (lineups["is_starting"] == True)][["game_id", "team_code", "player_id"]].rename(columns={"player_id": "sp_id"})
catcher = lineups[(lineups["position"] == 2) & (lineups["is_starting"] == True)][["game_id", "team_code", "player_id"]].rename(columns={"player_id": "catcher_id"})
battery = sp.merge(catcher, on=["game_id", "team_code"], how="inner")
battery = battery.merge(games, on="game_id", how="left")
battery["date"] = pd.to_datetime(battery["date"])

# 3. 투수 당일 성적 및 Baseline 병합
starter_stats = starter_logs[["game_id", "player_id", "ERA", "WHIP"]].rename(columns={"player_id": "sp_id"})
base_feats = starter_feats[["game_id", "player_id", "sp_weighted_era", "sp_weighted_whip"]].rename(columns={"player_id": "sp_id"})

battery = battery.merge(starter_stats, on=["game_id", "sp_id"], how="left")
battery = battery.merge(base_feats, on=["game_id", "sp_id"], how="left")

# 4. 배터리 조합별 누적 성적 계산 (과거 경기만 반영)
battery = battery.sort_values(["sp_id", "catcher_id", "date"])
group = battery.groupby(["sp_id", "catcher_id"])

battery["n_games"] = group.cumcount() # 이전까지 맞춘 호흡 횟수
battery["raw_avg_era"] = group["ERA"].transform(lambda x: x.shift(1).expanding().mean())
battery["raw_avg_whip"] = group["WHIP"].transform(lambda x: x.shift(1).expanding().mean())

# 5. [Empirical Bayes Smoothing] 보정 로직
K = 3 # 배터리 표본은 쌓이기 힘들므로 K값을 조금 낮춰 실제 호흡 성적을 더 빨리 반영하게 함

def smooth_metric(row, raw_col, base_col):
    n = row["n_games"]
    # 데이터가 아예 없거나 첫 경기면 투수 기본 실력 사용
    if n == 0 or pd.isna(row[raw_col]):
        return row[base_col]
    
    # 가중 평균: (배터리성적 * n + 투수기본실력 * K) / (n + K)
    # n이 커질수록 raw_avg(배터리 실제 성적) 비중이 커짐
    smoothed = (row[raw_col] * n + row[base_col] * K) / (n + K)
    return smoothed

battery["battery_avg_era"] = battery.apply(lambda x: smooth_metric(x, "raw_avg_era", "sp_weighted_era"), axis=1)
battery["battery_avg_whip"] = battery.apply(lambda x: smooth_metric(x, "raw_avg_whip", "sp_weighted_whip"), axis=1)

# 6. 추가 보정: 배터리 궁합 지수 (투수 기본 대비 얼마나 좋아졌나?)
# 이 지표가 음수면 포수가 투수를 잘 리드해서 성적이 좋아졌다는 뜻 (ERA 기준)
battery["battery_synergy_era"] = battery["battery_avg_era"] - battery["sp_weighted_era"]

# 7. 저장
battery["battery_games_together"] = battery["n_games"]
battery["is_battery_reliable"] = (battery["n_games"] >= 5).astype(int)

out_cols = ["game_id", "team_code", "sp_id", "catcher_id", 
            "battery_games_together", "battery_avg_era", "battery_avg_whip", 
            "battery_synergy_era", "is_battery_reliable"]

battery[out_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 보정된 배터리 피처 저장 완료: {OUTPUT_PATH}")