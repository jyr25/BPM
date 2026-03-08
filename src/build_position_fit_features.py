import pandas as pd

LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/position_fit_features.csv"

# -------------------------------------------------
# load
# -------------------------------------------------
lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

lineups["game_id"] = lineups["game_id"].astype(str)
lineups["player_id"] = lineups["player_id"].astype(str)
lineups["team_code"] = lineups["team_code"].astype(str)

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

# 선수 전체 평균 OPS (현재 경기 이전)
batter["player_overall_ops_prior"] = (
    batter.groupby("player_id")["OPS"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# 라인업과 merge
lineup_batter = lineups.merge(
    batter[["game_id", "player_id", "OPS", "player_overall_ops_prior"]],
    on=["game_id", "player_id"],
    how="left"
)

# 선발 타자만
lineup_batter = lineup_batter[lineup_batter["is_starting"] == True].copy()
lineup_batter = lineup_batter[lineup_batter["batting_order"] != "P"].copy()

# 선수-포지션별 과거 평균 OPS
batter_for_pos = lineup_batter[[
    "game_id", "player_id", "position", "OPS"
]].merge(
    games[["game_id", "date"]],
    on="game_id",
    how="left"
).sort_values(["player_id", "position", "date"])

batter_for_pos["player_position_ops_prior"] = (
    batter_for_pos.groupby(["player_id", "position"])["OPS"]
    .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
)

# 다시 merge
lineup_batter = lineup_batter.merge(
    batter_for_pos[["game_id", "player_id", "position", "player_position_ops_prior"]],
    on=["game_id", "player_id", "position"],
    how="left"
)

# 기본값
lineup_batter["player_overall_ops_prior"] = lineup_batter["player_overall_ops_prior"].fillna(0.700)
lineup_batter["player_position_ops_prior"] = lineup_batter["player_position_ops_prior"].fillna(
    lineup_batter["player_overall_ops_prior"]
)

# 포지션 적합성
lineup_batter["position_fit_score"] = (
    lineup_batter["player_position_ops_prior"] - lineup_batter["player_overall_ops_prior"]
)

# 팀 단위 합산
team_position_fit = (
    lineup_batter.groupby(["game_id", "team_code"])["position_fit_score"]
    .sum()
    .reset_index()
)

team_position_fit.rename(
    columns={"position_fit_score": "team_position_fit_score"},
    inplace=True
)

team_position_fit.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_PATH)
print("rows:", len(team_position_fit))
print(team_position_fit.head())