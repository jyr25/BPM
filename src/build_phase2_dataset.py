import pandas as pd

# 경로 설정
PHASE1_PATH = "data/processed/phase1_dataset.csv"
LINEUP_PATH = "data/processed/lineup_features.csv"
BATTERY_PATH = "data/processed/battery_features.csv"
ORDER_FIT_PATH = "data/processed/order_fit_features.csv"
POSITION_FIT_PATH = "data/processed/position_fit_features.csv"
CONTEXT_PATH = "data/processed/context_features.csv"
OUTPUT_PATH = "data/processed/phase2_dataset.csv"

# 1. 데이터 로드
phase1 = pd.read_csv(PHASE1_PATH)
lineup = pd.read_csv(LINEUP_PATH)
battery = pd.read_csv(BATTERY_PATH)
order_fit = pd.read_csv(ORDER_FIT_PATH)
position_fit = pd.read_csv(POSITION_FIT_PATH)
context = pd.read_csv(CONTEXT_PATH)

# 2. [핵심] 모든 데이터프레임의 결합 키 타입을 str로 강제 통일
# 이 부분이 제대로 안 되면 ValueError: You are trying to merge on int64 and str columns 에러가 납니다.
def unify_types(df):
    cols_to_fix = ["game_id", "home_team", "away_team", "team_code"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

phase1 = unify_types(phase1)
lineup = unify_types(lineup)
battery = unify_types(battery)
order_fit = unify_types(order_fit)
position_fit = unify_types(position_fit)
context = unify_types(context)

# 3. Rename 및 피처 준비 (기존 로직 유지)
lineup_home = lineup.rename(columns={"team_code": "home_team", "lineup_weighted_ops": "home_lineup_ops", "lineup_weighted_isop": "home_lineup_isop"})
lineup_away = lineup.rename(columns={"team_code": "away_team", "lineup_weighted_ops": "away_lineup_ops", "lineup_weighted_isop": "away_lineup_isop"})

battery_home = battery.rename(columns={"team_code": "home_team", "battery_games_together": "home_battery_games", "battery_avg_era": "home_battery_era", "battery_synergy_era": "home_battery_synergy"})
battery_away = battery.rename(columns={"team_code": "away_team", "battery_games_together": "away_battery_games", "battery_avg_era": "away_battery_era", "battery_synergy_era": "away_battery_synergy"})

order_home = order_fit.rename(columns={"team_code": "home_team", "team_order_fit_ops_score": "home_order_fit_ops", "team_order_fit_isop_score": "home_order_fit_isop"})
order_away = order_fit.rename(columns={"team_code": "away_team", "team_order_fit_ops_score": "away_order_fit_ops", "team_order_fit_isop_score": "away_order_fit_isop"})

position_home = position_fit.rename(columns={"team_code": "home_team", "team_position_fit_ops_score": "home_pos_fit_ops", "team_position_fit_isop_score": "home_pos_fit_isop"})
position_away = position_fit.rename(columns={"team_code": "away_team", "team_position_fit_ops_score": "away_pos_fit_ops", "team_position_fit_isop_score": "away_pos_fit_isop"})

# 4. Merge 수행
df = phase1.copy()

# Lineup
df = df.merge(lineup_home[["game_id", "home_team", "home_lineup_ops", "home_lineup_isop"]], on=["game_id", "home_team"], how="left")
df = df.merge(lineup_away[["game_id", "away_team", "away_lineup_ops", "away_lineup_isop"]], on=["game_id", "away_team"], how="left")

# Battery
df = df.merge(battery_home[["game_id", "home_team", "home_battery_games", "home_battery_era", "home_battery_synergy"]], on=["game_id", "home_team"], how="left")
df = df.merge(battery_away[["game_id", "away_team", "away_battery_games", "away_battery_era", "away_battery_synergy"]], on=["game_id", "away_team"], how="left")

# Order Fit
df = df.merge(order_home[["game_id", "home_team", "home_order_fit_ops", "home_order_fit_isop"]], on=["game_id", "home_team"], how="left")
df = df.merge(order_away[["game_id", "away_team", "away_order_fit_ops", "away_order_fit_isop"]], on=["game_id", "away_team"], how="left")

# Position Fit
df = df.merge(position_home[["game_id", "home_team", "home_pos_fit_ops", "home_pos_fit_isop"]], on=["game_id", "home_team"], how="left")
df = df.merge(position_away[["game_id", "away_team", "away_pos_fit_ops", "away_pos_fit_isop"]], on=["game_id", "away_team"], how="left")

# Context
df = df.merge(context[["game_id", "vs_team_winrate_diff", "park_winrate_diff"]], on=["game_id"], how="left")

# 5. Diff(차이) 피처 생성 및 저장
df["lineup_ops_diff"] = df["home_lineup_ops"] - df["away_lineup_ops"]
df["lineup_isop_diff"] = df["home_lineup_isop"] - df["away_lineup_isop"]
df["battery_games_diff"] = df["home_battery_games"] - df["away_battery_games"]
df["battery_synergy_diff"] = df["home_battery_synergy"] - df["away_battery_synergy"]
df["order_fit_ops_diff"] = df["home_order_fit_ops"] - df["away_order_fit_ops"]
df["order_fit_isop_diff"] = df["home_order_fit_isop"] - df["away_order_fit_isop"]
df["pos_fit_ops_diff"] = df["home_pos_fit_ops"] - df["away_pos_fit_ops"]
df["pos_fit_isop_diff"] = df["home_pos_fit_isop"] - df["away_pos_fit_isop"]

df.fillna(0, inplace=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"🚀 Phase 2 데이터셋 구축 완료! (저장경로: {OUTPUT_PATH})")