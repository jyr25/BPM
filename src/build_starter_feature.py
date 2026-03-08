import pandas as pd
import numpy as np

INPUT = "data/processed/starter_game_logs.csv"
OUTPUT = "data/processed/starter_features.csv"


def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


df = pd.read_csv(INPUT)

# 타입 정리
df["game_id"] = df["game_id"].astype(str)
df["player_id"] = df["player_id"].astype(str)
df["team_code"] = df["team_code"].astype(str)

df["year"] = df["game_id"].str[:4]

df["game_date"] = pd.to_datetime(
    df["year"] + "-" + df["game_date"],
    errors="coerce"
)

numeric_cols = [
    "IP_real","HR","BB","SO","ERA","WHIP"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# --------------------------------------------------
# 기본 선발 지표
# --------------------------------------------------

# FIP
df["sp_fip"] = (
    (13 * df["HR"] + 3 * df["BB"] - 2 * df["SO"])
    / df["IP_real"].replace(0, np.nan)
) + 3.1

# ERA
df["sp_era"] = df["ERA"]

# WHIP
df["sp_whip"] = df["WHIP"]

# K/BB
df["sp_kbb"] = df.apply(lambda x: safe_divide(x["SO"], x["BB"]), axis=1)

# IP/start
df["sp_ip_per_start"] = df["IP_real"]


# --------------------------------------------------
# 시계열 feature
# --------------------------------------------------

df = df.sort_values(["player_id","game_date"])

# 휴식일
df["prev_game"] = df.groupby("player_id")["game_date"].shift(1)

df["sp_rest_days"] = (
    (df["game_date"] - df["prev_game"]).dt.days - 1
)

# 최근 3경기 FIP
df["sp_recent3_fip"] = (
    df.groupby("player_id")["sp_fip"]
    .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
)


# --------------------------------------------------
# 저장
# --------------------------------------------------

cols = [
    "game_id",
    "player_id",
    "team_code",
    "vs_team",
    "game_date",
    "sp_fip",
    "sp_era",
    "sp_whip",
    "sp_kbb",
    "sp_ip_per_start",
    "sp_rest_days",
    "sp_recent3_fip"
]

starter_features = df[cols]

starter_features.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT)
print("rows:", len(starter_features))
print(starter_features.head())