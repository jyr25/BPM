import pandas as pd
import numpy as np

INPUT = "data/processed/batter_game_logs.csv"
PREV_SUMMARY = "data/processed/team_batting_prev_summary.csv"
OUTPUT = "data/processed/team_batting_features.csv"

def calculate_weighted_val(current_val, prev_val, current_game_count, threshold=20):
    """
    시즌 초반에는 전년도 성적을, 20경기 이상 치르면 올해 성적 비중을 높임
    """
    if pd.isna(prev_val): return current_val
    if pd.isna(current_val): return prev_val
    
    weight = min(current_game_count / threshold, 1.0)
    return (current_val * weight) + (prev_val * (1 - weight))

def main():
    df = pd.read_csv(INPUT)
    try:
        prev_df = pd.read_csv(PREV_SUMMARY)
    except FileNotFoundError:
        print(f"⚠️ {PREV_SUMMARY}가 없습니다. 전년도 보정 없이 진행합니다.")
        prev_df = pd.DataFrame()

    # 1. 타입 및 날짜 정리
    df["game_id"] = df["game_id"].astype(str)
    df["year"] = df["game_id"].str[:4].astype(int)
    # 날짜 복구 (이전 스크립트들과 동일 로직)
    # game_logs에 game_date가 없다면 game_id 등을 통해 병합하거나 생성 필요
    # 여기서는 정렬을 위해 game_id를 기준으로 시계열 처리함
    
    # 2. 경기별 팀 단위 합계 (Daily Team Total)
    team_daily = df.groupby(["game_id", "team_code", "year"]).agg({
        "PA": "sum",
        "AB": "sum",
        "H": "sum",
        "HR": "sum",
        "BB": "sum",
        "SO": "sum"
    }).reset_index()

    # 3. 시계열 순서 정렬 (팀별 경기 순서)
    team_daily = team_daily.sort_values(["team_code", "game_id"])

    # --------------------------------------------------
    # 피처 엔지니어링 (누수 방지 shift(1) 적용)
    # --------------------------------------------------

    # A. 최근 10경기 팀 타율 (AVG)
    # 단순 평균이 아니라 최근 10경기의 (안타 합계 / 타석 합계)로 계산해야 정확함
    rolling_obj = team_daily.groupby("team_code")
    
    team_daily["rolling_h"] = rolling_obj["H"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    team_daily["rolling_ab"] = rolling_obj["AB"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    team_daily["rolling_pa"] = rolling_obj["PA"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    team_daily["rolling_hr"] = rolling_obj["HR"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())

    team_daily["team_avg_recent10"] = team_daily["rolling_h"] / team_daily["rolling_ab"].replace(0, np.nan)
    team_daily["team_hr_rate_recent10"] = team_daily["rolling_hr"] / team_daily["rolling_pa"].replace(0, np.nan)

    # B. 시즌 누적 경기 수 (가중치 계산용)
    team_daily["cum_games"] = team_daily.groupby(["team_code", "year"]).cumcount()

    # C. 전년도 데이터 병합 및 가중 평균 적용
    if not prev_df.empty:
        team_daily = team_daily.merge(prev_df, on=["team_code", "year"], how="left")
        
        # 가중 타율
        team_daily["team_avg_weighted"] = team_daily.apply(
            lambda x: calculate_weighted_val(x["team_avg_recent10"], x["prev_team_avg"], x["cum_games"]), axis=1
        )
        # 가중 홈런율
        team_daily["team_hr_weighted"] = team_daily.apply(
            lambda x: calculate_weighted_val(x["team_hr_rate_recent10"], x["prev_team_hr_rate"], x["cum_games"]), axis=1
        )
    else:
        team_daily["team_avg_weighted"] = team_daily["team_avg_recent10"]
        team_daily["team_hr_weighted"] = team_daily["team_hr_rate_recent10"]

    # 4. 최종 컬럼 선택 (예측 모델용)
    final_cols = [
        "game_id", "team_code", "year",
        "team_avg_weighted", "team_hr_weighted"
    ]
    
    team_batting_features = team_daily[final_cols].copy()
    
    # 결측치 처리 (완전 시즌 극초반 데이터용)
    team_batting_features = team_batting_features.fillna(team_batting_features.mean(numeric_only=True))

    team_batting_features.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ 팀 타격 피처 생성 완료: {OUTPUT}")
    print(team_batting_features.head())

if __name__ == "__main__":
    main()