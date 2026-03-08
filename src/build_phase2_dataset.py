import pandas as pd

PHASE1_PATH = "data/processed/phase1_dataset.csv"
LINEUP_PATH = "data/processed/lineup_features.csv"
BATTERY_PATH = "data/processed/battery_features.csv"
ORDER_FIT_PATH = "data/processed/order_fit_features.csv"
POSITION_FIT_PATH = "data/processed/position_fit_features.csv"
CONTEXT_PATH = "data/processed/context_features.csv"
OUTPUT_PATH = "data/processed/phase2_dataset.csv"

phase1 = pd.read_csv(PHASE1_PATH)
lineup = pd.read_csv(LINEUP_PATH)
battery = pd.read_csv(BATTERY_PATH)
order_fit = pd.read_csv(ORDER_FIT_PATH)
position_fit = pd.read_csv(POSITION_FIT_PATH)
context = pd.read_csv(CONTEXT_PATH)

phase1["game_id"] = phase1["game_id"].astype(str)
phase1["home_team"] = phase1["home_team"].astype(str)
phase1["away_team"] = phase1["away_team"].astype(str)
context["game_id"] = context["game_id"].astype(str)

for df in [lineup, battery, order_fit, position_fit]:
    df["game_id"] = df["game_id"].astype(str)
    df["team_code"] = df["team_code"].astype(str)

# lineup
lineup_home = lineup.rename(columns={
    "team_code": "home_team",
    "lineup_weighted_ops": "home_lineup_weighted_ops"
})
lineup_away = lineup.rename(columns={
    "team_code": "away_team",
    "lineup_weighted_ops": "away_lineup_weighted_ops"
})

# battery
battery_home = battery.rename(columns={
    "team_code": "home_team",
    "battery_games_together": "home_battery_games",
    "battery_avg_era": "home_battery_avg_era",
    "battery_avg_whip": "home_battery_avg_whip"
})
battery_away = battery.rename(columns={
    "team_code": "away_team",
    "battery_games_together": "away_battery_games",
    "battery_avg_era": "away_battery_avg_era",
    "battery_avg_whip": "away_battery_avg_whip"
})

# order fit
order_home = order_fit.rename(columns={
    "team_code": "home_team",
    "team_order_fit_score": "home_team_order_fit_score"
})
order_away = order_fit.rename(columns={
    "team_code": "away_team",
    "team_order_fit_score": "away_team_order_fit_score"
})

# position fit
position_home = position_fit.rename(columns={
    "team_code": "home_team",
    "team_position_fit_score": "home_team_position_fit_score"
})
position_away = position_fit.rename(columns={
    "team_code": "away_team",
    "team_position_fit_score": "away_team_position_fit_score"
})

df = phase1.copy()

# merge
df = df.merge(
    lineup_home[["game_id", "home_team", "home_lineup_weighted_ops"]],
    on=["game_id", "home_team"],
    how="left"
)
df = df.merge(
    lineup_away[["game_id", "away_team", "away_lineup_weighted_ops"]],
    on=["game_id", "away_team"],
    how="left"
)

df = df.merge(
    battery_home[["game_id", "home_team", "home_battery_games", "home_battery_avg_era", "home_battery_avg_whip"]],
    on=["game_id", "home_team"],
    how="left"
)
df = df.merge(
    battery_away[["game_id", "away_team", "away_battery_games", "away_battery_avg_era", "away_battery_avg_whip"]],
    on=["game_id", "away_team"],
    how="left"
)

df = df.merge(
    order_home[["game_id", "home_team", "home_team_order_fit_score"]],
    on=["game_id", "home_team"],
    how="left"
)
df = df.merge(
    order_away[["game_id", "away_team", "away_team_order_fit_score"]],
    on=["game_id", "away_team"],
    how="left"
)

df = df.merge(
    position_home[["game_id", "home_team", "home_team_position_fit_score"]],
    on=["game_id", "home_team"],
    how="left"
)
df = df.merge(
    position_away[["game_id", "away_team", "away_team_position_fit_score"]],
    on=["game_id", "away_team"],
    how="left"
)
df = df.merge(
    context[["game_id", "vs_team_winrate_diff", "park_winrate_diff"]],
    on=["game_id"],
    how="left"
)

# diff
df["lineup_weighted_ops_diff"] = df["home_lineup_weighted_ops"] - df["away_lineup_weighted_ops"]

df["battery_games_diff"] = df["home_battery_games"] - df["away_battery_games"]
df["battery_avg_era_diff"] = df["home_battery_avg_era"] - df["away_battery_avg_era"]
df["battery_avg_whip_diff"] = df["home_battery_avg_whip"] - df["away_battery_avg_whip"]

df["team_order_fit_score_diff"] = (
    df["home_team_order_fit_score"] - df["away_team_order_fit_score"]
)

df["team_position_fit_score_diff"] = (
    df["home_team_position_fit_score"] - df["away_team_position_fit_score"]
)

df["vs_team_winrate_diff"] = df["vs_team_winrate_diff"].fillna(0)
df["park_winrate_diff"] = df["park_winrate_diff"].fillna(0)

df.fillna(0, inplace=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT_PATH)
print("rows:", len(df))
print(df[[
    "lineup_weighted_ops_diff",
    "battery_games_diff",
    "battery_avg_era_diff",
    "battery_avg_whip_diff",
    "team_order_fit_score_diff",
    "team_position_fit_score_diff"
]].head())