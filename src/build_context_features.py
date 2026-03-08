import pandas as pd
import numpy as np

GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/context_features.csv"


def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


games = pd.read_csv(GAME_INDEX_PATH)

# -------------------------------------------------
# 기본 정리
# -------------------------------------------------
games["game_id"] = games["game_id"].astype(str)
games["home_team"] = games["home_team"].astype(str)
games["away_team"] = games["away_team"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")

games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")

# 홈팀 승리 여부
games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
games["away_win"] = (games["away_score"] > games["home_score"]).astype(int)

# 구장 컬럼 찾기
stadium_col = None
for c in ["Stadium_code", "s_code"]:
    if c in games.columns:
        stadium_col = c
        break

if stadium_col is None:
    raise ValueError("game_index.csv에서 구장 컬럼(stadium 또는 s_code)을 찾지 못했습니다.")

games["stadium_code"] = games[stadium_col].astype(str)

# -------------------------------------------------
# 팀별 long format
# -------------------------------------------------
home_df = games[["game_id", "date", "home_team", "away_team", "stadium_code", "home_win"]].copy()
home_df = home_df.rename(columns={
    "home_team": "team_code",
    "away_team": "opp_team",
    "home_win": "win"
})

away_df = games[["game_id", "date", "away_team", "home_team", "stadium_code", "away_win"]].copy()
away_df = away_df.rename(columns={
    "away_team": "team_code",
    "home_team": "opp_team",
    "away_win": "win"
})

team_games = pd.concat([home_df, away_df], ignore_index=True)
team_games = team_games.sort_values(["team_code", "date", "game_id"]).reset_index(drop=True)

# -------------------------------------------------
# 상대팀별 승률 (현재 경기 이전)
# -------------------------------------------------
team_games["vs_team_games_before"] = (
    team_games.groupby(["team_code", "opp_team"]).cumcount()
)

team_games["vs_team_wins_cum"] = (
    team_games.groupby(["team_code", "opp_team"])["win"]
    .transform(lambda s: s.shift(1).cumsum())
)

team_games["vs_team_wins_cum"] = team_games["vs_team_wins_cum"].fillna(0)

team_games["vs_team_winrate"] = team_games.apply(
    lambda x: safe_divide(x["vs_team_wins_cum"], x["vs_team_games_before"]),
    axis=1
)

# 데이터 없으면 중립 0.5
team_games["vs_team_winrate"] = team_games["vs_team_winrate"].fillna(0.5)

# -------------------------------------------------
# 구장별 승률 (현재 경기 이전)
# -------------------------------------------------
team_games["park_games_before"] = (
    team_games.groupby(["team_code", "stadium_code"]).cumcount()
)

team_games["park_wins_cum"] = (
    team_games.groupby(["team_code", "stadium_code"])["win"]
    .transform(lambda s: s.shift(1).cumsum())
)

team_games["park_wins_cum"] = team_games["park_wins_cum"].fillna(0)

team_games["park_winrate"] = team_games.apply(
    lambda x: safe_divide(x["park_wins_cum"], x["park_games_before"]),
    axis=1
)

team_games["park_winrate"] = team_games["park_winrate"].fillna(0.5)

# -------------------------------------------------
# 홈 / 원정용 분리
# -------------------------------------------------
home_context = team_games.rename(columns={
    "team_code": "home_team",
    "vs_team_winrate": "home_vs_team_winrate",
    "park_winrate": "home_park_winrate"
})[["game_id", "home_team", "home_vs_team_winrate", "home_park_winrate"]]

away_context = team_games.rename(columns={
    "team_code": "away_team",
    "vs_team_winrate": "away_vs_team_winrate",
    "park_winrate": "away_park_winrate"
})[["game_id", "away_team", "away_vs_team_winrate", "away_park_winrate"]]

# -------------------------------------------------
# 원래 게임 테이블에 merge
# -------------------------------------------------
df = games.merge(
    home_context,
    on=["game_id", "home_team"],
    how="left"
)

df = df.merge(
    away_context,
    on=["game_id", "away_team"],
    how="left"
)

df["vs_team_winrate_diff"] = (
    df["home_vs_team_winrate"] - df["away_vs_team_winrate"]
)

df["park_winrate_diff"] = (
    df["home_park_winrate"] - df["away_park_winrate"]
)

out = df[[
    "game_id",
    "home_team",
    "away_team",
    "home_vs_team_winrate",
    "away_vs_team_winrate",
    "vs_team_winrate_diff",
    "home_park_winrate",
    "away_park_winrate",
    "park_winrate_diff"
]].copy()

out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_PATH)
print("rows:", len(out))
print(out.head())
print("\nmissing values:")
print(out.isna().sum())