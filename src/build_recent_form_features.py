import pandas as pd
import numpy as np

GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/recent_form_features.csv"

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

# 승패 결과
games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
games["away_win"] = (games["away_score"] > games["home_score"]).astype(int)

# -------------------------------------------------
# 팀 단위 long format으로 변환
# -------------------------------------------------
home_df = games[["game_id", "date", "home_team", "home_win"]].copy()
home_df = home_df.rename(columns={
    "home_team": "team_code",
    "home_win": "win"
})

away_df = games[["game_id", "date", "away_team", "away_win"]].copy()
away_df = away_df.rename(columns={
    "away_team": "team_code",
    "away_win": "win"
})

team_games = pd.concat([home_df, away_df], ignore_index=True)

team_games = team_games.sort_values(["team_code", "date", "game_id"]).reset_index(drop=True)

# -------------------------------------------------
# 최근 10경기 승률
# 현재 경기 제외, 이전 10경기 기준
# -------------------------------------------------
team_games["recent10_winrate"] = (
    team_games.groupby("team_code")["win"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
)

# 시즌 첫 경기 등 결측은 0.5로 중립 처리
team_games["recent10_winrate"] = team_games["recent10_winrate"].fillna(0.5)

# -------------------------------------------------
# 홈/원정용으로 다시 분리
# -------------------------------------------------
home_recent = team_games.rename(columns={
    "team_code": "home_team",
    "recent10_winrate": "home_recent10_winrate"
})[["game_id", "home_team", "home_recent10_winrate"]]

away_recent = team_games.rename(columns={
    "team_code": "away_team",
    "recent10_winrate": "away_recent10_winrate"
})[["game_id", "away_team", "away_recent10_winrate"]]

# -------------------------------------------------
# 원래 경기 테이블에 merge
# -------------------------------------------------
df = games.merge(
    home_recent,
    on=["game_id", "home_team"],
    how="left"
)

df = df.merge(
    away_recent,
    on=["game_id", "away_team"],
    how="left"
)

# diff
df["recent10_winrate_diff"] = (
    df["home_recent10_winrate"] - df["away_recent10_winrate"]
)

# -------------------------------------------------
# 저장
# -------------------------------------------------
out = df[[
    "game_id",
    "date",
    "home_team",
    "away_team",
    "home_recent10_winrate",
    "away_recent10_winrate",
    "recent10_winrate_diff"
]].copy()

out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_PATH)
print("rows:", len(out))
print(out.head())

print("\nmissing values:")
print(out.isna().sum())