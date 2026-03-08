import pandas as pd
import numpy as np

INPUT = "data/processed/bullpen_game_logs.csv"
OUTPUT = "data/processed/bullpen_team_features.csv"


def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


df = pd.read_csv(INPUT)

# -----------------------------
# 타입 정리
# -----------------------------

df["game_id"] = df["game_id"].astype(str)
df["team_code"] = df["team_code"].astype(str)
df["vs_team"] = df["vs_team"].astype(str)

numeric_cols = [
    "IP_real",
    "H",
    "HR",
    "BB",
    "SO",
    "ER",
    "OPS"
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# -----------------------------
# game_date 복구
# (현재 04-01 같은 형태)
# -----------------------------

df["year"] = df["game_id"].str[:4]

df["game_date"] = pd.to_datetime(
    df["year"] + "-" + df["game_date"].astype(str),
    errors="coerce"
)


# -----------------------------
# 팀 불펜 집계
# -----------------------------

team = (
    df.groupby(["game_id", "team_code", "vs_team", "game_date"])
      .agg({
          "IP_real": "sum",
          "H": "sum",
          "HR": "sum",
          "BB": "sum",
          "SO": "sum",
          "ER": "sum"
      })
      .reset_index()
)


# -----------------------------
# 불펜 지표 계산
# -----------------------------

team["bullpen_whip"] = (team["H"] + team["BB"]) / team["IP_real"]

team["bullpen_hr9"] = team["HR"] * 9 / team["IP_real"]

team["bullpen_kbb"] = np.where(
    team["BB"] == 0,
    team["SO"],
    team["SO"] / team["BB"]
)

team["bullpen_era"] = team["ER"] * 9 / team["IP_real"]

# -----------------------------
# OPS allowed (IP 가중 평균)
# -----------------------------

ops_weighted = (
    df.assign(weighted_ops=df["OPS"] * df["IP_real"])
      .groupby(["game_id", "team_code"])
      .agg({
          "weighted_ops": "sum",
          "IP_real": "sum"
      })
      .reset_index()
)

ops_weighted["bullpen_ops_allowed"] = (
    ops_weighted["weighted_ops"] / ops_weighted["IP_real"]
)

team = team.merge(
    ops_weighted[["game_id", "team_code", "bullpen_ops_allowed"]],
    on=["game_id", "team_code"],
    how="left"
)


# -----------------------------
# recent 7경기 rolling
# -----------------------------

team = team.sort_values(
    ["team_code", "game_date", "game_id"]
).reset_index(drop=True)

rolling_cols = [
    "bullpen_whip",
    "bullpen_hr9",
    "bullpen_kbb",
    "bullpen_era",
    "bullpen_ops_allowed"
]

for col in rolling_cols:
    team[f"{col}_recent7"] = (
        team.groupby("team_code")[col]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )


# -----------------------------
# 저장
# -----------------------------

cols = [
    "game_id",
    "team_code",
    "vs_team",
    "game_date",

    "bullpen_whip",
    "bullpen_hr9",
    "bullpen_kbb",
    "bullpen_era",
    "bullpen_ops_allowed",

    "bullpen_whip_recent7",
    "bullpen_hr9_recent7",
    "bullpen_kbb_recent7",
    "bullpen_era_recent7",
    "bullpen_ops_allowed_recent7"
]

team_features = team[cols]

team_features = team[cols]

# NaN 처리
team_features = team_features.fillna({
    "bullpen_whip_recent7": team_features["bullpen_whip"].mean(),
    "bullpen_hr9_recent7": team_features["bullpen_hr9"].mean(),
    "bullpen_kbb_recent7": team_features["bullpen_kbb"].mean(),
    "bullpen_era_recent7": team_features["bullpen_era"].mean(),
    "bullpen_ops_allowed_recent7": team_features["bullpen_ops_allowed"].mean()
})

team_features.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("saved:", OUTPUT)
print("rows:", len(team_features))
print(team_features.head())