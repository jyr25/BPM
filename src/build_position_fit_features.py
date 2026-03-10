import pandas as pd
import numpy as np

LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/position_fit_features.csv"

# -------------------------------------------------
# 1. Load
# -------------------------------------------------
lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

for df in [lineups, batter, games]:
    df["game_id"] = df["game_id"].astype(str)

lineups["player_id"] = lineups["player_id"].astype(str)
lineups["team_code"] = lineups["team_code"].astype(str)
lineups["position"] = lineups["position"].astype(str) # 포지션 비교를 위해 문자열 변환

batter["player_id"] = batter["player_id"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")

# -------------------------------------------------
# 2. 선수별 전체 평균 OPS 계산 (Baseline)
# -------------------------------------------------
batter = batter.merge(games[["game_id", "date"]], on="game_id", how="left")
batter = batter.sort_values(["player_id", "date"])

batter["player_overall_ops_prior"] = (
    batter.groupby("player_id")["OPS"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# -------------------------------------------------
# 3. 선수-포지션별 과거 성적 계산
# -------------------------------------------------
# 라인업 정보와 타자 로그 결합
lineup_stats = lineups.merge(
    batter[["game_id", "player_id", "OPS", "player_overall_ops_prior"]],
    on=["game_id", "player_id"],
    how="left"
)

# 선발 타자만 필터링 (투수 '1' 제외)
lineup_stats = lineup_stats[(lineup_stats["is_starting"] == True) & (lineup_stats["position"] != "1")].copy()

# 시점 정렬
lineup_stats = lineup_stats.sort_values(["player_id", "position", "game_id"])
group = lineup_stats.groupby(["player_id", "position"])

# 해당 포지션으로 출전한 누적 경기 수 계산 (현재 경기 제외)
lineup_stats["pos_game_count"] = group.cumcount() 
# 해당 포지션에서의 평균 OPS (현재 경기 제외)
lineup_stats["pos_ops_prior"] = group["OPS"].transform(lambda x: x.shift(1).expanding().mean())

# -------------------------------------------------
# 4. [핵심] 최소 8경기 이상일 때만 Fit Score 적용
# -------------------------------------------------
MIN_GAMES = 8

def calculate_pos_fit(row):
    # 해당 포지션 출전 경험이 8경기 미만이면 데이터 노이즈로 간주하고 0점 처리
    if row["pos_game_count"] < MIN_GAMES or pd.isna(row["pos_ops_prior"]):
        return 0.0
    
    # (특정 포지션 OPS) - (선수 전체 평균 OPS)
    return row["pos_ops_prior"] - row["player_overall_ops_prior"]

lineup_stats["position_fit_score"] = lineup_stats.apply(calculate_pos_fit, axis=1)

# -------------------------------------------------
# 5. 팀 단위 합산 및 저장
# -------------------------------------------------
team_position_fit = (
    lineup_stats.groupby(["game_id", "team_code"])["position_fit_score"]
    .sum()
    .reset_index()
)

team_position_fit.rename(columns={"position_fit_score": "team_position_fit_score"}, inplace=True)
team_position_fit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 수정 완료: 최소 {MIN_GAMES}경기 이상 데이터만 반영됨")
print(team_position_fit.head())