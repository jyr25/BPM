import pandas as pd
import numpy as np

# 경로 설정 유지
LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/order_fit_features.csv"

# 1. Load & Basic Preprocessing
lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

for df in [lineups, batter, games]:
    df["game_id"] = df["game_id"].astype(str)

lineups["player_id"] = lineups["player_id"].astype(str)
lineups["team_code"] = lineups["team_code"].astype(str)
batter["player_id"] = batter["player_id"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")

# 2. IsoP 계산 (SLG - AVG)
# batter_game_logs에 이미 있다면 생략 가능하지만, 안전을 위해 재계산 로직 포함
if 'isop' not in batter.columns:
    batter['isop'] = batter['SLG'] - batter['AVG']

# 3. 선수별 전체 평균 Baseline (OPS & IsoP)
batter = batter.merge(games[["game_id", "date"]], on="game_id", how="left")
batter = batter.sort_values(["player_id", "date"])

# 이전 경기까지의 누적 평균 (최소 5경기)
group_p = batter.groupby("player_id")
batter["player_overall_ops_prior"] = group_p["OPS"].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
batter["player_overall_isop_prior"] = group_p["isop"].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())

# 4. 라인업 정보와 결합 및 역할군 정의
def get_order_role(order):
    try:
        order_num = int(float(order))
        if order_num in [1, 2]: return "TableSetter"
        if order_num in [3, 4, 5]: return "Cleanup"
        if order_num in [6, 7, 8, 9]: return "Lower"
    except: return "Other"
    return "Other"

lineups["order_role"] = lineups["batting_order"].apply(get_order_role)

lineup_stats = lineups.merge(
    batter[["game_id", "player_id", "OPS", "isop", "PA", "player_overall_ops_prior", "player_overall_isop_prior", "date"]],
    on=["game_id", "player_id"],
    how="left"
)

# 선발 타자 필터링
lineup_stats = lineup_stats[(lineup_stats["is_starting"] == True) & (lineup_stats["order_role"] != "Other")].copy()
lineup_stats = lineup_stats.sort_values(["player_id", "order_role", "date"])

# 5. 역할군별 과거 성적 계산
group_r = lineup_stats.groupby(["player_id", "order_role"])
lineup_stats["role_pa_cum"] = group_r["PA"].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
lineup_stats["role_ops_prior"] = group_r["OPS"].transform(lambda x: x.shift(1).expanding().mean())
lineup_stats["role_isop_prior"] = group_r["isop"].transform(lambda x: x.shift(1).expanding().mean())

# 6. Fit Score 계산 함수 (Confidence 가중치 포함)
MIN_PA = 15

def calculate_fit_scores(row):
    # 기본값 설정
    res = {'ops_fit': 0.0, 'isop_fit': 0.0}
    
    if row["role_pa_cum"] < MIN_PA:
        return pd.Series(res)
    
    # 신뢰도 가중치 (표본이 많을수록 1.0에 수렴)
    conf = min(row["role_pa_cum"] / 50, 1.0)
    
    # OPS Fit
    if not pd.isna(row["role_ops_prior"]) and not pd.isna(row["player_overall_ops_prior"]):
        res['ops_fit'] = (row["role_ops_prior"] - row["player_overall_ops_prior"]) * conf
        
    # IsoP Fit (오늘 추가된 핵심!)
    if not pd.isna(row["role_isop_prior"]) and not pd.isna(row["player_overall_isop_prior"]):
        res['isop_fit'] = (row["role_isop_prior"] - row["player_overall_isop_prior"]) * conf
        
    return pd.Series(res)

fit_results = lineup_stats.apply(calculate_fit_scores, axis=1)
lineup_stats[['order_fit_ops_score', 'order_fit_isop_score']] = fit_results

# 7. 팀 단위 평균 합산 및 저장
team_order_fit = (
    lineup_stats.groupby(["game_id", "team_code"])[['order_fit_ops_score', 'order_fit_isop_score']]
    .mean()
    .reset_index()
)

team_order_fit.rename(columns={
    "order_fit_ops_score": "team_order_fit_ops_score",
    "order_fit_isop_score": "team_order_fit_isop_score"
}, inplace=True)

team_order_fit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ OPS & IsoP 적합도가 포함된 피처 저장 완료: {OUTPUT_PATH}")
print(team_order_fit.head())