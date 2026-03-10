import pandas as pd
import numpy as np

LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/order_fit_features.csv"

# -------------------------------------------------
# 1. Load & Basic Preprocessing
# -------------------------------------------------
lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

for df in [lineups, batter, games]:
    df["game_id"] = df["game_id"].astype(str)

lineups["player_id"] = lineups["player_id"].astype(str)
lineups["team_code"] = lineups["team_code"].astype(str)
batter["player_id"] = batter["player_id"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")

# -------------------------------------------------
# 2. 타순 역할군(Role) 정의 함수
# -------------------------------------------------
def get_order_role(order):
    try:
        order_num = int(float(order))
        if order_num in [1, 2]: return "TableSetter"
        if order_num in [3, 4, 5]: return "Cleanup"
        if order_num in [6, 7, 8, 9]: return "Lower"
    except:
        return "Other"
    return "Other"

lineups["order_role"] = lineups["batting_order"].apply(get_order_role)

# -------------------------------------------------
# 3. 선수별 전체 평균 OPS (Baseline)
# -------------------------------------------------
batter = batter.merge(games[["game_id", "date"]], on="game_id", how="left")
batter = batter.sort_values(["player_id", "date"])

batter["player_overall_ops_prior"] = (
    batter.groupby("player_id")["OPS"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# -------------------------------------------------
# 4. 선수-역할군별 과거 성적 계산
# -------------------------------------------------
# 라인업 정보와 타자 로그 결합
lineup_stats = lineups.merge(
    batter[["game_id", "player_id", "OPS", "PA", "player_overall_ops_prior"]],
    on=["game_id", "player_id"],
    how="left"
)

# 선발 타자만 필터링 (투수 제외)
lineup_stats = lineup_stats[(lineup_stats["is_starting"] == True) & (lineups["order_role"] != "Other")].copy()

# 시점 정렬
lineup_stats = lineup_stats.sort_values(["player_id", "order_role", "game_id"])
group = lineup_stats.groupby(["player_id", "order_role"])

# 역할군별 누적 타석(PA) 및 평균 OPS 계산 (현재 경기 제외)
lineup_stats["role_pa_cum"] = group["PA"].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
lineup_stats["role_ops_prior"] = group["OPS"].transform(lambda x: x.shift(1).expanding().mean())

# -------------------------------------------------
# 5. [핵심] 최소 표본(10타석) 적용 및 Fit Score 계산
# -------------------------------------------------
MIN_PA = 10

def calculate_fit(row):
    # 타석수가 부족하면 본인 평균과 차이가 없다고 간주 (0점)
    if row["role_pa_cum"] < MIN_PA or pd.isna(row["role_ops_prior"]):
        return 0.0
    
    # (역할군 성적) - (본인 전체 평균)
    return row["role_ops_prior"] - row["player_overall_ops_prior"]

lineup_stats["order_fit_score"] = lineup_stats.apply(calculate_fit, axis=1)

# -------------------------------------------------
# 6. 팀 단위 합산 및 저장
# -------------------------------------------------
team_order_fit = (
    lineup_stats.groupby(["game_id", "team_code"])["order_fit_score"]
    .sum()
    .reset_index()
)

team_order_fit.rename(columns={"order_fit_score": "team_order_fit_score"}, inplace=True)
team_order_fit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 수정된 타순 적합도 피처 저장 완료: {OUTPUT_PATH}")
print(team_order_fit.head())