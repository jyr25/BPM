import pandas as pd
import numpy as np

LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"

OUTPUT = "data/processed/lineup_features.csv"

# -------------------------------------------------
# load
# -------------------------------------------------

lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

lineups["game_id"] = lineups["game_id"].astype(str)
batter["game_id"] = batter["game_id"].astype(str)
batter["player_id"] = batter["player_id"].astype(str)

lineups["player_id"] = lineups["player_id"].astype(str)

games["game_id"] = games["game_id"].astype(str)
games["date"] = pd.to_datetime(games["date"])

# -------------------------------------------------
# OPS history 계산
# -------------------------------------------------

batter = batter.merge(
    games[["game_id", "date"]],
    on="game_id",
    how="left"
)

batter = batter.sort_values(["player_id", "date"])

# 이전 경기까지 평균 OPS
batter["ops_prior"] = (
    batter.groupby("player_id")["OPS"]
    .apply(lambda x: x.expanding().mean().shift(1))
    .reset_index(level=0, drop=True)
)

# -------------------------------------------------
# lineup merge
# -------------------------------------------------

df = lineups.merge(
    batter[["player_id", "game_id", "ops_prior"]],
    on=["player_id", "game_id"],
    how="left"
)

# OPS 없으면 리그 평균 근사값
df["ops_prior"] = df["ops_prior"].fillna(0.700)

# -------------------------------------------------
# batting order weight
# -------------------------------------------------

order_weight = {
    "1":1.10,
    "2":1.08,
    "3":1.05,
    "4":1.03,
    "5":1.01,
    "6":0.98,
    "7":0.95,
    "8":0.93,
    "9":0.89
}

df = df[df["is_starting"] == True]

df["batting_order"] = df["batting_order"].astype(str)

df["order_weight"] = df["batting_order"].map(order_weight)

# 투수 타석(P) 제거
df = df[df["batting_order"] != "P"]

# -------------------------------------------------
# weighted lineup score
# -------------------------------------------------

df["weighted_ops"] = df["ops_prior"] * df["order_weight"]

team_lineup = (
    df.groupby(["game_id", "team_code"])["weighted_ops"]
    .sum()
    .reset_index()
)

team_lineup.rename(
    columns={"weighted_ops":"lineup_weighted_ops"},
    inplace=True
)

# -------------------------------------------------
# save
# -------------------------------------------------

team_lineup.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT)
print("rows:", len(team_lineup))
print(team_lineup.head())