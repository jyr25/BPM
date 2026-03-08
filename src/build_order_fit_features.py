import pandas as pd

LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/order_fit_features.csv"

# -------------------------------------------------
# load
# -------------------------------------------------
lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

lineups["game_id"] = lineups["game_id"].astype(str)
lineups["player_id"] = lineups["player_id"].astype(str)
lineups["team_code"] = lineups["team_code"].astype(str)
lineups["batting_order"] = lineups["batting_order"].astype(str)

batter["game_id"] = batter["game_id"].astype(str)
batter["player_id"] = batter["player_id"].astype(str)
batter["OPS"] = pd.to_numeric(batter["OPS"], errors="coerce")

games["game_id"] = games["game_id"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")

# -------------------------------------------------
# batter에 날짜 붙이기
# -------------------------------------------------
batter = batter.merge(
    games[["game_id", "date"]],
    on="game_id",
    how="left"
)

batter = batter.sort_values(["player_id", "date"])

# -------------------------------------------------
# 선수 전체 평균 OPS (현재 경기 이전)
# -------------------------------------------------
batter["player_overall_ops_prior"] = (
    batter.groupby("player_id")["OPS"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# -------------------------------------------------
# 라인업에 batter OPS 붙이기
# -------------------------------------------------
lineup_batter = lineups.merge(
    batter[["game_id", "player_id", "OPS", "player_overall_ops_prior"]],
    on=["game_id", "player_id"],
    how="left"
)

# 선발 타자만
lineup_batter = lineup_batter[lineup_batter["is_starting"] == True].copy()
lineup_batter = lineup_batter[lineup_batter["batting_order"] != "P"].copy()

# -------------------------------------------------
# 선수-타순별 과거 평균 OPS
# 현재 경기 이전만 사용하려면 누적 방식으로 계산
# -------------------------------------------------
batter_for_order = lineup_batter[[
    "game_id", "player_id", "batting_order", "OPS"
]].merge(
    games[["game_id", "date"]],
    on="game_id",
    how="left"
).sort_values(["player_id", "batting_order", "date"])

batter_for_order["player_order_ops_prior"] = (
    batter_for_order.groupby(["player_id", "batting_order"])["OPS"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# 다시 라인업 데이터에 붙이기
lineup_batter = lineup_batter.merge(
    batter_for_order[["game_id", "player_id", "batting_order", "player_order_ops_prior"]],
    on=["game_id", "player_id", "batting_order"],
    how="left"
)

# 기본값 처리
lineup_batter["player_overall_ops_prior"] = lineup_batter["player_overall_ops_prior"].fillna(0.700)
lineup_batter["player_order_ops_prior"] = lineup_batter["player_order_ops_prior"].fillna(
    lineup_batter["player_overall_ops_prior"]
)

# 타순 적합성
lineup_batter["order_fit_score"] = (
    lineup_batter["player_order_ops_prior"] - lineup_batter["player_overall_ops_prior"]
)

# 팀 단위 합산
team_order_fit = (
    lineup_batter.groupby(["game_id", "team_code"])["order_fit_score"]
    .sum()
    .reset_index()
)

team_order_fit.rename(
    columns={"order_fit_score": "team_order_fit_score"},
    inplace=True
)

team_order_fit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_PATH)
print("rows:", len(team_order_fit))
print(team_order_fit.head())