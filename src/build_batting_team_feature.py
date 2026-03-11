import pandas as pd
import numpy as np

INPUT = "data/processed/batter_game_logs.csv"
PREV_SUMMARY = "data/processed/team_batting_prev_summary.csv"
OUTPUT = "data/processed/team_batting_features.csv"

def calculate_weighted_val(current_val, prev_val, current_game_count, threshold=20):
    """시즌 초반에는 전년도 성적을, 20경기 이상 치르면 올해 성적 비중을 높임"""
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

    df["game_id"] = df["game_id"].astype(str)
    df["year"] = df["game_id"].str[:4].astype(int)

    # 1. 경기별 팀 단위 합계 (고급 지표용 컬럼 추가)
    agg_cols = ["PA", "AB", "H", "H2", "H3", "HR", "BB", "SO", "HBP", "SF", "R"]
    existing_cols = [c for c in agg_cols if c in df.columns]
    
    team_daily = df.groupby(["game_id", "team_code", "year"])[existing_cols].sum().reset_index()
    team_daily = team_daily.sort_values(["team_code", "game_id"])

    # 2. 최근 10경기 누적 합계 (Rolling Sum) - 누수 방지 shift(1)
    rolling_obj = team_daily.groupby("team_code")
    
    # 득점 상관관계가 높은 지표들을 위해 누적값 계산
    for col in existing_cols:
        team_daily[f"roll10_{col.lower()}"] = rolling_obj[col].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).sum()
        )

    # 3. 최근 10경기 비율 지표 계산
    # (1) AVG
    team_daily["recent10_avg"] = team_daily["roll10_h"] / team_daily["roll10_ab"].replace(0, np.nan)
    
    # (2) OBP (출루율)
    obp_num = team_daily["roll10_h"] + team_daily["roll10_bb"] + team_daily.get("roll10_hbp", 0)
    obp_den = team_daily["roll10_ab"] + team_daily["roll10_bb"] + team_daily.get("roll10_hbp", 0) + team_daily.get("roll10_sf", 0)
    team_daily["recent10_obp"] = obp_num / obp_den.replace(0, np.nan)
    
    # (3) SLG (장타율)
    h1 = team_daily["roll10_h"] - team_daily.get("roll10_h2", 0) - team_daily.get("roll10_h3", 0) - team_daily["roll10_hr"]
    tb = (h1 * 1) + (team_daily.get("roll10_h2", 0) * 2) + (team_daily.get("roll10_h3", 0) * 3) + (team_daily["roll10_hr"] * 4)
    team_daily["recent10_slg"] = tb / team_daily["roll10_ab"].replace(0, np.nan)
    
    # (4) OPS & BB/K & IsoP
    team_daily["recent10_ops"] = team_daily["recent10_obp"] + team_daily["recent10_slg"]
    team_daily["recent10_bb_k"] = team_daily["roll10_bb"] / team_daily["roll10_so"].replace(0, np.nan)
    team_daily["recent10_isop"] = team_daily["recent10_slg"] - team_daily["recent10_avg"]

    # 4. 시즌 누적 경기 수 및 가중치 적용
    team_daily["cum_games"] = team_daily.groupby(["team_code", "year"]).cumcount()

    if not prev_df.empty:
        team_daily = team_daily.merge(prev_df, on=["team_code", "year"], how="left")
        
        # 가중 지표 생성 루프
        weight_targets = {
            "avg": "prev_team_avg",
            "ops": "prev_team_ops",
            "isop": "prev_team_isop",
            "bb_k": "prev_team_bb_k"
        }
        
        for cur_suffix, prev_col in weight_targets.items():
            if prev_col in team_daily.columns:
                team_daily[f"team_{cur_suffix}_weighted"] = team_daily.apply(
                    lambda x: calculate_weighted_val(x[f"recent10_{cur_suffix}"], x[prev_col], x["cum_games"]), axis=1
                )
    else:
        # 전년도 데이터 없을 시 최근 지표 그대로 사용
        team_daily["team_avg_weighted"] = team_daily["recent10_avg"]
        team_daily["team_ops_weighted"] = team_daily["recent10_ops"]
        team_daily["team_isop_weighted"] = team_daily["recent10_isop"]
        team_daily["team_bb_k_weighted"] = team_daily["recent10_bb_k"]

    # 5. 최종 컬럼 정리
    weighted_cols = [c for c in team_daily.columns if "_weighted" in c]
    final_cols = ["game_id", "team_code", "year"] + weighted_cols
    
    team_batting_features = team_daily[final_cols].copy()
    team_batting_features = team_batting_features.fillna(0) # 혹은 평균값 처리

    team_batting_features.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ 확장된 타격 피처 생성 완료 (컬럼수: {len(weighted_cols)})")

if __name__ == "__main__":
    main()