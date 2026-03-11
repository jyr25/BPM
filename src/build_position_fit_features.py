import pandas as pd
import numpy as np

LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/position_fit_features.csv"

# 1. Load & Preprocessing
lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

# 데이터 타입 통일 및 날짜 처리
for df in [lineups, batter, games]:
    df["game_id"] = df["game_id"].astype(str)

lineups["player_id"] = lineups["player_id"].astype(str)
lineups["team_code"] = lineups["team_code"].astype(str)
lineups["position"] = lineups["position"].astype(str) 

batter["player_id"] = batter["player_id"].astype(str)
if 'isop' not in batter.columns:
    batter['isop'] = batter['SLG'] - batter['AVG']

games["date"] = pd.to_datetime(games["date"], errors="coerce")

# 2. 선수별 전체 평균 Baseline (누수 방지 shift)
batter = batter.merge(games[["game_id", "date"]], on="game_id", how="left")
batter = batter.sort_values(["player_id", "date"])

group_p = batter.groupby("player_id")
batter["player_overall_ops_prior"] = group_p["OPS"].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
batter["player_overall_isop_prior"] = group_p["isop"].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())

# 3. 선수-포지션별 과거 성적 계산
lineup_stats = lineups.merge(
    batter[["game_id", "player_id", "OPS", "isop", "player_overall_ops_prior", "player_overall_isop_prior", "date"]],
    on=["game_id", "player_id"],
    how="left"
)

# 선발 타자만 (투수 '1' 제외)
lineup_stats = lineup_stats[(lineup_stats["is_starting"] == True) & (lineup_stats["position"] != "1")].copy()
lineup_stats = lineup_stats.sort_values(["player_id", "position", "date"])

group_pos = lineup_stats.groupby(["player_id", "position"])
lineup_stats["pos_game_count"] = group_pos.cumcount() 
lineup_stats["pos_ops_prior"] = group_pos["OPS"].transform(lambda x: x.shift(1).expanding().mean())
lineup_stats["pos_isop_prior"] = group_pos["isop"].transform(lambda x: x.shift(1).expanding().mean())

# 4. Fit Score 계산 (Confidence 가중치 적용)
MIN_GAMES = 8

def calculate_pos_fit(row):
    res = {'ops_fit': 0.0, 'isop_fit': 0.0}
    
    # 최소 경기수 미달 시 0점
    if row["pos_game_count"] < MIN_GAMES:
        return pd.Series(res)
    
    # 신뢰도 가중치: 해당 포지션 경험이 많을수록 데이터 신뢰 (20경기 이상 시 1.0)
    conf = min(row["pos_game_count"] / 20, 1.0)
    
    # OPS Fit
    if not pd.isna(row["pos_ops_prior"]) and not pd.isna(row["player_overall_ops_prior"]):
        res['ops_fit'] = (row["pos_ops_prior"] - row["player_overall_ops_prior"]) * conf
        
    # IsoP Fit
    if not pd.isna(row["pos_isop_prior"]) and not pd.isna(row["player_overall_isop_prior"]):
        res['isop_fit'] = (row["pos_isop_prior"] - row["player_overall_isop_prior"]) * conf
        
    return pd.Series(res)

fit_results = lineup_stats.apply(calculate_pos_fit, axis=1)
lineup_stats[['pos_fit_ops', 'pos_fit_isop']] = fit_results

# 5. 팀 단위 평균 합산 및 저장
team_position_fit = (
    lineup_stats.groupby(["game_id", "team_code"])[['pos_fit_ops', 'pos_fit_isop']]
    .mean() # 팀 전체 수비 최적화 수준
    .reset_index()
)

team_position_fit.rename(columns={
    "pos_fit_ops": "team_position_fit_ops_score",
    "pos_fit_isop": "team_position_fit_isop_score"
}, inplace=True)

team_position_fit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 포지션 적합도(OPS/IsoP) 피처 생성 완료: {OUTPUT_PATH}")