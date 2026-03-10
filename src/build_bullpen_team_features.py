import pandas as pd
import numpy as np

INPUT = "data/processed/bullpen_game_logs.csv"
PREV_SUMMARY = "data/processed/team_bullpen_prev_summary.csv"
OUTPUT = "data/processed/bullpen_team_features.csv"

def calculate_weighted_val(current_val, prev_val, current_game_count, threshold=15):
    """
    시즌 초반에는 전년도 성적을, 경기가 쌓일수록 올해 성적 비중을 높임
    불펜은 선발보다 변동이 잦으므로 threshold를 15경기 정도로 설정
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
        print(f"⚠️ {PREV_SUMMARY} 파일이 없습니다. 전년도 보정 없이 진행합니다.")
        prev_df = pd.DataFrame()

    # 1. 타입 및 날짜 정리
    df["game_id"] = df["game_id"].astype(str)
    df["year"] = df["game_id"].str[:4].astype(int)
    df["game_date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["game_date"].astype(str), errors="coerce")

    # 2. 경기별 팀 단위 집계
    team_daily = df.groupby(["game_id", "team_code", "vs_team", "game_date", "year"]).agg({
        "IP_real": "sum",
        "H": "sum",
        "BB": "sum",
        "HR": "sum",
        "SO": "sum",
        "ER": "sum",
        "NP": "sum" 
    }).reset_index()

    # 3. 당일 경기 지표 계산 (이 값들은 나중에 shift해서 과거 데이터로만 쓸 것임)
    team_daily["daily_era"] = (team_daily["ER"] * 9) / team_daily["IP_real"].replace(0, np.nan)
    team_daily["daily_whip"] = (team_daily["H"] + team_daily["BB"]) / team_daily["IP_real"].replace(0, np.nan)
    team_daily["daily_hr9"] = (team_daily["HR"] * 9) / team_daily["IP_real"].replace(0, np.nan)
    team_daily["daily_kbb"] = team_daily["SO"] / team_daily["BB"].replace(0, np.nan)
    team_daily["daily_ops"] = ((team_daily["H"] + team_daily["BB"] + team_daily["HR"]) / team_daily["IP_real"].replace(0, np.nan)) * 100

    # 4. 시계열 순서 정렬
    team_daily = team_daily.sort_values(["team_code", "game_date"])

    # --------------------------------------------------
    # 피처 엔지니어링 (누수 방지를 위해 모두 shift(1) 적용)
    # --------------------------------------------------

    # A. 불펜 피로도 (최근 3일간 팀 불펜 총 투구수 합계)
    # 3일(3D) 윈도우를 사용하려면 날짜 기준 rolling 필요
    team_daily = team_daily.set_index('game_date')
    team_daily["bullpen_np_3d"] = (
        team_daily.groupby("team_code")["NP"]
        .transform(lambda x: x.shift(1).rolling('3D').sum())
        .fillna(0)
    )
    team_daily = team_daily.reset_index()

    # B. 최근 7경기 폼 (Recent Form)
    metrics_to_roll = ["era", "whip", "hr9", "kbb", "ops"]
    for m in metrics_to_roll:
        team_daily[f"bullpen_{m}_recent7"] = (
            team_daily.groupby("team_code")[f"daily_{m}"]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        )

    # C. 시즌 누적 경기 수 (가중치 계산용)
    team_daily["cum_games"] = team_daily.groupby(["team_code", "year"]).cumcount() # shift(1) 효과와 동일 (0부터 시작)

    # D. 전년도 데이터 병합 및 가중 평균 적용
    if not prev_df.empty:
        team_daily = team_daily.merge(prev_df, on=["team_code", "year"], how="left")
        
        for m in metrics_to_roll:
            prev_col = f"prev_bullpen_{m}"
            cur_col = f"bullpen_{m}_recent7"
            if prev_col in team_daily.columns:
                team_daily[f"bullpen_{m}_weighted"] = team_daily.apply(
                    lambda x: calculate_weighted_val(x[cur_col], x[prev_col], x["cum_games"]), axis=1
                )
    else:
        for m in metrics_to_roll:
            team_daily[f"bullpen_{m}_weighted"] = team_daily[f"bullpen_{m}_recent7"]

    # 5. 최종 컬럼 선택 (학습에 사용할 '경기 전' 시점의 피처들만)
    final_cols = [
        "game_id", "team_code", "vs_team", "game_date",
        "bullpen_era_weighted", "bullpen_whip_weighted", 
        "bullpen_hr9_weighted", "bullpen_kbb_weighted", 
        "bullpen_ops_weighted", "bullpen_np_3d"
    ]
    
    # 결측치 처리 (완전히 데이터가 없는 극초반은 리그 평균 등으로 대체)
    team_features = team_daily[final_cols].copy()
    team_features = team_features.fillna(team_features.mean(numeric_only=True))

    team_features.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ 불펜 팀 피처 생성 완료: {OUTPUT}")
    print(team_features.head())

if __name__ == "__main__":
    main()