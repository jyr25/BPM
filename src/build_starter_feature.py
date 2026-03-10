import pandas as pd
import numpy as np

INPUT = "data/processed/starter_game_logs.csv"
PREV_SUMMARY = "data/processed/pitcher_yearly_summary.csv"
OUTPUT = "data/processed/starter_features.csv"

def calculate_weighted_val(current_val, prev_val, current_ip, threshold=50):
    if pd.isna(prev_val): return current_val
    if pd.isna(current_val): return prev_val
    weight = min(current_ip / threshold, 1.0)
    return (current_val * weight) + (prev_val * (1 - weight))

def build_starter_features():
    df = pd.read_csv(INPUT)
    prev_df = pd.read_csv(PREV_SUMMARY)

    # 1. 기본 타입 정리
    df["game_id"] = df["game_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["year"] = df["game_id"].str[:4].astype(int)
    df["game_date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["game_date"], errors="coerce")
    
    prev_df["player_id"] = prev_df["player_id"].astype(str)

    # 2. 전년도 데이터 병합
    df = df.merge(prev_df, on=["player_id", "year"], how="left")

    # 3. 정렬 및 누적 계산 준비
    df = df.sort_values(["player_id", "game_date"])
    groupby_obj = df.groupby(["player_id", "year"])

    # --- 4. 올해 시즌 누적 스탯 계산 (shift(1)로 오늘 경기 제외) ---
    
    # 누적 이닝 및 주요 지표
    cols_to_sum = ["IP_real", "H", "BB", "HR", "SO", "ER"]
    for col in cols_to_sum:
        df[f"cur_{col.lower()}_sum"] = groupby_obj[col].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)

    # 올해 경기 전 시점의 누적 지표들 계산
    # ERA: (ER * 9) / IP
    df["cur_era_running"] = (df["cur_er_sum"] * 9) / df["cur_ip_real_sum"].replace(0, np.nan)
    # WHIP: (H + BB) / IP
    df["cur_whip_running"] = (df["cur_h_sum"] + df["cur_bb_sum"]) / df["cur_ip_real_sum"].replace(0, np.nan)
    # K/BB: SO / BB
    df["cur_kbb_running"] = df["cur_so_sum"] / df["cur_bb_sum"].replace(0, np.nan)
    # FIP
    df["cur_fip_running"] = ((13 * df["cur_hr_sum"] + 3 * df["cur_bb_sum"] - 2 * df["cur_so_sum"]) / df["cur_ip_real_sum"].replace(0, np.nan)) + 3.1

    # --- 5. 가중 평균 적용 (Weighted Features) ---
    
    metrics = {
        "fip": ("cur_fip_running", "prev_fip"),
        "era": ("cur_era_running", "prev_era"),
        "whip": ("cur_whip_running", "prev_whip"), # prev_summary에 whip이 있다면
        "kbb": ("cur_kbb_running", "prev_kbb")    # prev_summary에 kbb가 있다면
    }

    for key, (cur_col, prev_col) in metrics.items():
        if prev_col in df.columns:
            df[f"sp_weighted_{key}"] = df.apply(
                lambda x: calculate_weighted_val(x[cur_col], x[prev_col], x["cur_ip_real_sum"]), axis=1
            )
        else:
            # 전년도 지표가 summary에 없는 경우 올해 기록만 사용
            df[f"sp_weighted_{key}"] = df[cur_col]

    # 6. 휴식일 계산 (기존 동일)
    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["sp_rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days - 1
    df["sp_rest_days"] = df["sp_rest_days"].fillna(5).clip(0, 15)

    # 7. 최종 컬럼 선택 (이 이름들이 phase1_dataset.csv의 리스트와 일치해야 함)
    final_cols = [
        "game_id", "player_id", "team_code", "game_date",
        "sp_weighted_fip", "sp_weighted_era", "sp_weighted_whip", "sp_weighted_kbb",
        "sp_rest_days", "cur_ip_real_sum"
    ]
    
    starter_features = df[final_cols].copy()
    starter_features.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ 가중치 지표 확장 완료: {OUTPUT}")

if __name__ == "__main__":
    build_starter_features()