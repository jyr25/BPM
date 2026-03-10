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
# 기본 정리 및 전처리
# -------------------------------------------------
games["game_id"] = games["game_id"].astype(str)
games["home_team"] = games["home_team"].astype(str)
games["away_team"] = games["away_team"].astype(str)
games["date"] = pd.to_datetime(games["date"], errors="coerce")

games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")

# 승패 결정
games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
games["away_win"] = (games["away_score"] > games["home_score"]).astype(int)

# 구장 컬럼 처리
stadium_col = next((c for c in ["Stadium_code", "s_code", "stadium"] if c in games.columns), None)
if stadium_col is None:
    raise ValueError("구장 컬럼을 찾을 수 없습니다.")
games["stadium_code"] = games[stadium_col].astype(str)

# -------------------------------------------------
# 팀별 Long Format 변환 (기록 누적용)
# -------------------------------------------------
home_df = games[["game_id", "date", "home_team", "away_team", "stadium_code", "home_win"]].copy()
home_df = home_df.rename(columns={"home_team": "team_code", "away_team": "opp_team", "home_win": "win"})

away_df = games[["game_id", "date", "away_team", "home_team", "stadium_code", "away_win"]].copy()
away_df = away_df.rename(columns={"away_team": "team_code", "home_team": "opp_team", "away_win": "win"})

team_games = pd.concat([home_df, away_df], ignore_index=True)
team_games = team_games.sort_values(["team_code", "date", "game_id"]).reset_index(drop=True)

# -------------------------------------------------
# [핵심] 최소 표본(3경기) 기반 승률 계산 함수
# -------------------------------------------------
MIN_GAMES = 3

def get_reliable_winrate(df, group_cols):
    # 1. 이전 경기까지의 누적 판수와 승수 계산
    df["games_count"] = df.groupby(group_cols).cumcount()
    df["wins_cum"] = df.groupby(group_cols)["win"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    
    # 2. 승률 계산
    df["raw_winrate"] = df.apply(lambda x: safe_divide(x["wins_cum"], x["games_count"]), axis=1)
    
    # 3. 최소 경기수 미달 시 0.5(중립) 처리
    return np.where(df["games_count"] >= MIN_GAMES, df["raw_winrate"], 0.5)

# 상대 전적 승률 적용
team_games["vs_team_winrate"] = get_reliable_winrate(team_games, ["team_code", "opp_team"])

# 구장별 승률 적용
team_games["park_winrate"] = get_reliable_winrate(team_games, ["team_code", "stadium_code"])

# -------------------------------------------------
# 데이터 병합 (버그 방지: drop_duplicates 추가)
# -------------------------------------------------
# 홈/원정 데이터로 재분리
home_context = team_games.rename(columns={
    "team_code": "home_team",
    "vs_team_winrate": "home_vs_team_winrate",
    "park_winrate": "home_park_winrate"
})[["game_id", "home_team", "home_vs_team_winrate", "home_park_winrate"]].drop_duplicates(subset=["game_id", "home_team"])

away_context = team_games.rename(columns={
    "team_code": "away_team",
    "vs_team_winrate": "away_vs_team_winrate",
    "park_winrate": "away_park_winrate"
})[["game_id", "away_team", "away_vs_team_winrate", "away_park_winrate"]].drop_duplicates(subset=["game_id", "away_team"])

# 원본 테이블에 병합
df = games.merge(home_context, on=["game_id", "home_team"], how="left")
df = df.merge(away_context, on=["game_id", "away_team"], how="left")

# 차이(Diff) 피처 생성
df["vs_team_winrate_diff"] = df["home_vs_team_winrate"] - df["away_vs_team_winrate"]
df["park_winrate_diff"] = df["home_park_winrate"] - df["away_park_winrate"]

# 최종 결과물 정리
out_cols = ["game_id", "home_team", "away_team", "home_vs_team_winrate", 
            "away_vs_team_winrate", "vs_team_winrate_diff", 
            "home_park_winrate", "away_park_winrate", "park_winrate_diff"]

out = df[out_cols].fillna(0.5) # 혹시 모를 결측치는 중립 처리
out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Context 피처 저장 완료 (최소 {MIN_GAMES}경기 제한 적용)")
print(out.head())