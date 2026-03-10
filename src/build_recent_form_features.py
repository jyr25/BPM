import pandas as pd
import numpy as np

GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/recent_form_features.csv"

games = pd.read_csv(GAME_INDEX_PATH)

# -------------------------------------------------
# 1. 기본 정리
# -------------------------------------------------
games["game_id"] = games["game_id"].astype(str)
games["home_team"] = games["home_team"].astype(str)
games["away_team"] = games["away_team"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")

# 승패 결과 (무승부는 0으로 처리되어 패배와 동일하게 취급되나, 필요시 0.5로 조정 가능)
games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
games["away_win"] = (games["away_score"] > games["home_score"]).astype(int)

# -------------------------------------------------
# 2. 팀 단위 Long Format 변환
# -------------------------------------------------
home_df = games[["game_id", "date", "home_team", "home_win"]].rename(columns={"home_team": "team_code", "home_win": "win"})
away_df = games[["game_id", "date", "away_team", "away_win"]].rename(columns={"away_team": "team_code", "away_win": "win"})

team_games = pd.concat([home_df, away_df], ignore_index=True)
team_games = team_games.sort_values(["team_code", "date", "game_id"]).reset_index(drop=True)

# -------------------------------------------------
# 3. [핵심] 최근 기세 계산 (단순 평균 vs 가중 평균)
# -------------------------------------------------

# A. 단순 10경기 승률 (기존)
team_games["recent10_winrate"] = (
    team_games.groupby("team_code")["win"]
    .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
)

# B. 지수 가중 이동 평균 (EWMA) - 최근 경기에 더 높은 가중치
# span=10은 대략 최근 10경기의 흐름을 보되 최신 결과에 민감하게 반응함
team_games["recent_trend_ewm"] = (
    team_games.groupby("team_code")["win"]
    .transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
)

# 결측치 중립 처리 (0.5)
team_games["recent10_winrate"] = team_games["recent10_winrate"].fillna(0.5)
team_games["recent_trend_ewm"] = team_games["recent_trend_ewm"].fillna(0.5)

# -------------------------------------------------
# 4. 홈/원정 재분리 및 버그 방지 (중복 제거)
# -------------------------------------------------
# 중복 제거를 통해 더블헤더나 데이터 오류로 인한 행 뻥튀기 방지
home_recent = team_games.rename(columns={
    "team_code": "home_team",
    "recent10_winrate": "home_recent10_winrate",
    "recent_trend_ewm": "home_recent_trend_ewm"
})[["game_id", "home_team", "home_recent10_winrate", "home_recent_trend_ewm"]].drop_duplicates(subset=["game_id", "home_team"])

away_recent = team_games.rename(columns={
    "team_code": "away_team",
    "recent10_winrate": "away_recent10_winrate",
    "recent_trend_ewm": "away_recent_trend_ewm"
})[["game_id", "away_team", "away_recent10_winrate", "away_recent_trend_ewm"]].drop_duplicates(subset=["game_id", "away_team"])

# -------------------------------------------------
# 5. 원래 테이블에 병합 및 Diff 계산
# -------------------------------------------------
df = games.merge(home_recent, on=["game_id", "home_team"], how="left")
df = df.merge(away_recent, on=["game_id", "away_team"], how="left")

# 단순 승률 차이
df["recent10_winrate_diff"] = df["home_recent10_winrate"] - df["away_recent10_winrate"]
# 가중 흐름(기세) 차이
df["recent_trend_diff"] = df["home_recent_trend_ewm"] - df["away_recent_trend_ewm"]

# -------------------------------------------------
# 6. 저장
# -------------------------------------------------
out_cols = [
    "game_id", "date", "home_team", "away_team",
    "home_recent10_winrate", "away_recent10_winrate", "recent10_winrate_diff",
    "home_recent_trend_ewm", "away_recent_trend_ewm", "recent_trend_diff"
]

out = df[out_cols].copy()
out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 기세(Trend) 피처 추가 및 중복 방지 완료: {OUTPUT_PATH}")
print(out[["recent10_winrate_diff", "recent_trend_diff"]].head())