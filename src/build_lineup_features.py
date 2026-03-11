import pandas as pd
import numpy as np

LINEUP_PATH = "data/processed/lineups_flat.csv"
BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT = "data/processed/lineup_features.csv"

# 1. Load
lineups = pd.read_csv(LINEUP_PATH)
batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

# ID 및 날짜 정리
for df in [lineups, batter, games]:
    df["game_id"] = df["game_id"].astype(str)
lineups["player_id"] = lineups["player_id"].astype(str)
batter["player_id"] = batter["player_id"].astype(str)
games["date"] = pd.to_datetime(games["date"])

# 2. IsoP 계산 추가
if 'isop' not in batter.columns:
    batter['isop'] = batter['SLG'] - batter['AVG']

# 3. OPS & IsoP History 계산 (Smoothing 적용)
batter = batter.merge(games[["game_id", "date"]], on="game_id", how="left")
batter = batter.sort_values(["player_id", "date"])

# 이전 경기까지의 누적 평균
group = batter.groupby("player_id")
batter["ops_raw_prior"] = group["OPS"].transform(lambda x: x.expanding().mean().shift(1))
batter["isop_raw_prior"] = group["isop"].transform(lambda x: x.expanding().mean().shift(1))
batter["pa_cum"] = group["PA"].transform(lambda x: x.cumsum().shift(1)).fillna(0)

# Bayesian Smoothing (표본 적은 선수 보정)
K_PA = 50
LEAGUE_AVG_OPS = 0.740
LEAGUE_AVG_ISOP = 0.140

batter["ops_prior"] = (batter["ops_raw_prior"] * batter["pa_cum"] + LEAGUE_AVG_OPS * K_PA) / (batter["pa_cum"] + K_PA)
batter["isop_prior"] = (batter["isop_raw_prior"] * batter["pa_cum"] + LEAGUE_AVG_ISOP * K_PA) / (batter["pa_cum"] + K_PA)

# 4. Lineup Merge
df = lineups[lineups["is_starting"] == True].copy()
df = df.merge(
    batter[["player_id", "game_id", "ops_prior", "isop_prior"]],
    on=["player_id", "game_id"],
    how="left"
)

# 신인 또는 데이터 없는 선수 처리
df["ops_prior"] = df["ops_prior"].fillna(0.700)
df["isop_prior"] = df["isop_prior"].fillna(0.120)

# 5. Batting Order Weight (타순별 중요도)
# 1~5번 타자에게 더 높은 가중치를 부여하여 핵심 전력을 강조
order_weight = {
    "1": 1.10, "2": 1.12, "3": 1.15, "4": 1.10, "5": 1.05,
    "6": 1.00, "7": 0.95, "8": 0.90, "9": 0.85
}
df["batting_order_str"] = df["batting_order"].astype(str)
df["weight"] = df["batting_order_str"].map(order_weight).fillna(1.0)

# 투수 타석(P) 제외
df = df[df["batting_order_str"] != "P"]

# 6. Weighted Scores 계산
df["weighted_ops"] = df["ops_prior"] * df["weight"]
df["weighted_isop"] = df["isop_prior"] * df["weight"]

# 팀 단위 합산
team_lineup = (
    df.groupby(["game_id", "team_code"])[["weighted_ops", "weighted_isop"]]
    .sum()
    .reset_index()
)

team_lineup.rename(
    columns={
        "weighted_ops": "lineup_weighted_ops",
        "weighted_isop": "lineup_weighted_isop"
    },
    inplace=True
)

# 7. Save
team_lineup.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"✅ 라인업 가중 지표(OPS/IsoP) 생성 완료: {OUTPUT}")