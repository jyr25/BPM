import pandas as pd
import numpy as np

# 경로 설정
LINEUP_PATH = "data/processed/lineups_flat.csv"
STARTER_LOG_PATH = "data/processed/starter_game_logs.csv"
STARTER_FEAT_PATH = "data/processed/starter_features.csv" # 투수 기본 실력 참조용
OUTPUT_PATH = "data/processed/battery_features.csv"

# 1. 로드
lineups = pd.read_csv(LINEUP_PATH)
starter_logs = pd.read_csv(STARTER_LOG_PATH)
starter_feats = pd.read_csv(STARTER_FEAT_PATH)

# ID 및 타입 정리
for df in [lineups, starter_logs, starter_feats]:
    df["game_id"] = df["game_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)

# 2. 선발/포수 추출 및 결합
sp = lineups[(lineups["position"] == 1) & (lineups["is_starting"] == True)][["game_id", "team_code", "player_id"]].rename(columns={"player_id": "sp_id"})
catcher = lineups[(lineups["position"] == 2) & (lineups["is_starting"] == True)][["game_id", "team_code", "player_id"]].rename(columns={"player_id": "catcher_id"})
battery = sp.merge(catcher, on=["game_id", "team_code"], how="inner")

# 3. 투수 당일 성적 및 기본 실력(Baseline) 병합

# A. 로그 데이터 정리
if "player_id" in starter_logs.columns:
    starter_stats = starter_logs[["game_id", "player_id", "ERA", "WHIP"]].rename(columns={"player_id": "sp_id"})
else:
    # 혹시 이미 sp_id로 되어 있을 경우를 대비
    starter_stats = starter_logs[["game_id", "sp_id", "ERA", "WHIP"]]

# B. 피처 데이터 정리 (에러 발생 지점 수정)
if "player_id" in starter_feats.columns:
    base_feats = starter_feats[["game_id", "player_id", "sp_weighted_era", "sp_weighted_whip"]].rename(columns={"player_id": "sp_id"})
else:
    # 이미 sp_id로 되어 있다면 그대로 사용
    base_feats = starter_feats[["game_id", "sp_id", "sp_weighted_era", "sp_weighted_whip"]]

# C. 병합
battery = battery.merge(starter_stats, on=["game_id", "sp_id"], how="left")
battery = battery.merge(base_feats, on=["game_id", "sp_id"], how="left")
# 4. 배터리 조합별 누적 성적 계산 (현재 경기 제외)
battery = battery.sort_values(["sp_id", "catcher_id", "game_id"])
group = battery.groupby(["sp_id", "catcher_id"])

battery["n_games"] = group.cumcount() # 호흡 맞춘 횟수
battery["raw_avg_era"] = group["ERA"].transform(lambda x: x.shift(1).expanding().mean())
battery["raw_avg_whip"] = group["WHIP"].transform(lambda x: x.shift(1).expanding().mean())

# 5. [전략 2] Empirical Bayes Smoothing 적용
# K(신뢰도 상수)가 5일 때, 5경기를 넘어야 실제 배터리 성적이 50% 이상 반영됨
K = 5 

def smooth_metric(row, raw_col, base_col):
    n = row["n_games"]
    if n == 0 or pd.isna(row[raw_col]):
        return row[base_col] # 경험 없으면 투수 기본 실력 사용
    
    # 가중 평균 식: (배터리성적 * n + 투수기본실력 * K) / (n + K)
    return (row[raw_col] * n + row[base_col] * K) / (n + K)

battery["battery_avg_era"] = battery.apply(lambda x: smooth_metric(x, "raw_avg_era", "sp_weighted_era"), axis=1)
battery["battery_avg_whip"] = battery.apply(lambda x: smooth_metric(x, "raw_avg_whip", "sp_weighted_whip"), axis=1)

# 6. [전략 3] 신뢰도 마스크 피처 추가
# 5경기 이상이면 신뢰할 수 있는 조합(1), 아니면(0)
battery["is_battery_reliable"] = (battery["n_games"] >= 5).astype(int)

# 7. 저장
out_cols = ["game_id", "team_code", "sp_id", "catcher_id", 
            "battery_games_together", "battery_avg_era", "battery_avg_whip", "is_battery_reliable"]
battery["battery_games_together"] = battery["n_games"]

battery[out_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 보정된 배터리 피처 저장 완료: {OUTPUT_PATH}")