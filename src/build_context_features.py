import pandas as pd
import numpy as np

# 경로 설정
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/context_features.csv"

def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return 0.5 
    return a / b

games = pd.read_csv(GAME_INDEX_PATH)

# 1. 전처리 및 구장 컬럼 식별
games["game_id"] = games["game_id"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")
games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
games["away_win"] = (games["away_score"] > games["home_score"]).astype(int)

# 🔥 구장 컬럼 찾기 추가
stadium_col = next((c for c in ["Stadium_code", "s_code", "stadium", "stadium_code"] if c in games.columns), None)
if stadium_col is None:
    raise ValueError(f"구장 컬럼을 찾을 수 없습니다. 현재 컬럼: {games.columns.tolist()}")

# 2. 팀별 데이터 구조 재편 (Long Format)
# 🔥 stadium_col 앞뒤에 따옴표가 없어야 변수에 담긴 실제 컬럼명(예: 'stadium')을 사용합니다.
home_df = games[["game_id", "date", "home_team", "away_team", stadium_col, "home_win"]].copy()
home_df = home_df.rename(columns={"home_team": "team_code", "away_team": "opp_team", stadium_col: "stadium_code", "home_win": "win"})

away_df = games[["game_id", "date", "away_team", "home_team", stadium_col, "away_win"]].copy()
away_df = away_df.rename(columns={"away_team": "team_code", "home_team": "opp_team", stadium_col: "stadium_code", "away_win": "win"})

team_games = pd.concat([home_df, away_df], ignore_index=True)
team_games = team_games.sort_values(["team_code", "date", "game_id"]).reset_index(drop=True)

# 3. 승률 계산 로직 (Smoothing 적용)
def get_smoothed_winrate(df, group_cols, prior_weight=5):
    group = df.groupby(group_cols)
    wins_cum = group["win"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    games_cum = group.cumcount() 
    
    smoothed_rate = (wins_cum + (0.5 * prior_weight)) / (games_cum + prior_weight)
    return smoothed_rate

# 상대 전적 및 구장별 승률 계산
team_games["vs_team_winrate"] = get_smoothed_winrate(team_games, ["team_code", "opp_team"])
team_games["park_winrate"] = get_smoothed_winrate(team_games, ["stadium_code", "team_code"]) # 구장별 팀 승률

# 4. 재병합 및 Diff 생성
home_context = team_games.copy().add_prefix("home_").rename(columns={"home_game_id": "game_id", "home_team_code": "home_team"})
away_context = team_games.copy().add_prefix("away_").rename(columns={"away_game_id": "game_id", "away_team_code": "away_team"})

df = games[["game_id", "home_team", "away_team"]].merge(
    home_context[["game_id", "home_team", "home_vs_team_winrate", "home_park_winrate"]],
    on=["game_id", "home_team"], how="left"
).merge(
    away_context[["game_id", "away_team", "away_vs_team_winrate", "away_park_winrate"]],
    on=["game_id", "away_team"], how="left"
)

# 최종 차이(Diff) 피처
df["vs_team_winrate_diff"] = df["home_vs_team_winrate"] - df["away_vs_team_winrate"]
df["park_winrate_diff"] = df["home_park_winrate"] - df["away_park_winrate"]

# 5. 저장
df.fillna(0.5).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"✅ Context 피처 저장 완료 (stadium_col 자동 인식 적용)")