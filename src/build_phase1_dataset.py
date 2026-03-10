import pandas as pd
import numpy as np

GAME_INDEX_PATH = "data/processed/game_index.csv"
STARTER_PATH = "data/processed/starter_features.csv"
BULLPEN_PATH = "data/processed/bullpen_team_features.csv"
TEAM_BATTING_PATH = "data/processed/team_batting_features.csv"
OUTPUT_PATH = "data/processed/phase1_dataset.csv"


def safe_diff(df, home_col, away_col, new_col):
    df[new_col] = df[home_col] - df[away_col]
    return df


# -------------------------------------------------
# 1. game_index
# -------------------------------------------------
games = pd.read_csv(GAME_INDEX_PATH)

games["game_id"] = games["game_id"].astype(str)
games["home_team"] = games["home_team"].astype(str)
games["away_team"] = games["away_team"].astype(str)

# year 추출
games["year"] = games["game_id"].str[:4]

# date 정리
games["date"] = pd.to_datetime(games["date"], errors="coerce")


# -------------------------------------------------
# 2. starter_features
# -------------------------------------------------
starter = pd.read_csv(STARTER_PATH)

starter["game_id"] = starter["game_id"].astype(str)
starter["team_code"] = starter["team_code"].astype(str)

starter_cols = [
    "sp_weighted_fip", 
    "sp_weighted_era", 
    "sp_weighted_whip", 
    "sp_weighted_kbb", 
    "sp_rest_days", 
    "cur_ip_real_sum"
]

starter_home = starter[["game_id", "team_code"] + starter_cols].copy()
starter_home = starter_home.rename(columns={"team_code": "home_team"})
starter_home = starter_home.rename(columns={c: f"home_{c}" for c in starter_cols})

starter_away = starter[["game_id", "team_code"] + starter_cols].copy()
starter_away = starter_away.rename(columns={"team_code": "away_team"})
starter_away = starter_away.rename(columns={c: f"away_{c}" for c in starter_cols})


# -------------------------------------------------
# 3. bullpen_team_features
# -------------------------------------------------
bullpen = pd.read_csv(BULLPEN_PATH)

bullpen["game_id"] = bullpen["game_id"].astype(str)
bullpen["team_code"] = bullpen["team_code"].astype(str)

bullpen_cols = [
    "bullpen_era_weighted", 
    "bullpen_whip_weighted", 
    "bullpen_np_3d"
    "bullpen_hr9_weighted",
    "bullpen_kbb_weighted",
    "bullpen_ops_weighted",
    "bullpen_whip_recent7",
    "bullpen_hr9_recent7",
    "bullpen_kbb_recent7",
    "bullpen_era_recent7",
    "bullpen_ops_recent7"
]

existing_bullpen_cols = [c for c in bullpen_cols if c in bullpen.columns]

bullpen_home = bullpen[["game_id", "team_code"] + existing_bullpen_cols].copy()
bullpen_home = bullpen_home.rename(columns={"team_code": "home_team"})
bullpen_home = bullpen_home.rename(columns={c: f"home_{c}" for c in existing_bullpen_cols})

bullpen_away = bullpen[["game_id", "team_code"] + existing_bullpen_cols].copy()
bullpen_away = bullpen_away.rename(columns={"team_code": "away_team"})
bullpen_away = bullpen_away.rename(columns={c: f"away_{c}" for c in existing_bullpen_cols})


# -------------------------------------------------
# 4. team_batting_features (가중 평균 지표 반영)
# -------------------------------------------------
batting = pd.read_csv(TEAM_BATTING_PATH)

batting["game_id"] = batting["game_id"].astype(str)
batting["team_code"] = batting["team_code"].astype(str)

# 우리가 새로 만든 가중치 컬럼들로 변경
batting_cols = [
    "team_avg_weighted", 
    "team_hr_weighted"
]

batting_home = batting[["game_id", "team_code"] + batting_cols].copy()
batting_home = batting_home.rename(columns={"team_code": "home_team"})
batting_home = batting_home.rename(columns={c: f"home_{c}" for c in batting_cols})

batting_away = batting[["game_id", "team_code"] + batting_cols].copy()
batting_away = batting_away.rename(columns={"team_code": "away_team"})
batting_away = batting_away.rename(columns={c: f"away_{c}" for c in batting_cols})


# -------------------------------------------------
# 6. merge
# -------------------------------------------------
df = games.copy()

# starter
df = df.merge(starter_home, on=["game_id", "home_team"], how="left")
df = df.merge(starter_away, on=["game_id", "away_team"], how="left")

# bullpen
df = df.merge(bullpen_home, on=["game_id", "home_team"], how="left")
df = df.merge(bullpen_away, on=["game_id", "away_team"], how="left")

# team batting
df = df.merge(batting_home, on=["game_id", "home_team"], how="left")
df = df.merge(batting_away, on=["game_id", "away_team"], how="left")


# -------------------------------------------------
# 7. diff feature 생성
# -------------------------------------------------

# starter diff
for c in starter_cols:
    home_c = f"home_{c}"
    away_c = f"away_{c}"
    diff_c = f"{c}_diff"
    if home_c in df.columns and away_c in df.columns:
        df[diff_c] = df[home_c] - df[away_c]

# bullpen diff
for c in existing_bullpen_cols:
    home_c = f"home_{c}"
    away_c = f"away_{c}"
    diff_c = f"{c}_diff"
    if home_c in df.columns and away_c in df.columns:
        df[diff_c] = df[home_c] - df[away_c]

# batting diff
for c in batting_cols:
    home_c = f"home_{c}"
    away_c = f"away_{c}"
    diff_c = f"{c}_diff"
    if home_c in df.columns and away_c in df.columns:
        df[diff_c] = df[home_c] - df[away_c]

# -------------------------------------------------
# 8. context feature placeholder
# 나중에 build_context_features.py 붙이면 여기 merge 확장 가능
# -------------------------------------------------


# -------------------------------------------------
# 9. 최종 컬럼 정리
# -------------------------------------------------
base_cols = [
    "game_id",
    "date",
    "year",
    "home_team",
    "away_team",
    "target_home_win"
]

diff_cols = [c for c in df.columns if c.endswith("_diff")]

final_cols = base_cols + diff_cols

phase1 = df[final_cols].copy()

# 필요하면 NaN 처리
# rolling 첫 경기 등에서 NaN 생길 수 있음
for c in phase1.columns:
    if c.endswith("_diff"):
        phase1[c] = pd.to_numeric(phase1[c], errors="coerce")

phase1.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

fill_zero_cols = [
    "bullpen_hr9_recent7_diff",
    "bullpen_whip_recent7_diff",
    "bullpen_kbb_recent7_diff",
    "bullpen_era_recent7_diff",
    "bullpen_ops_allowed_recent7_diff",
    "team_recent30_rbi_diff",
    "sp_rest_days_diff",
    "sp_recent3_fip_diff",
]

for col in fill_zero_cols:
    if col in phase1.columns:
        phase1[col] = phase1[col].fillna(0)

print("saved:", OUTPUT_PATH)
print("rows:", len(phase1))
print(phase1.head())
print("\nmissing values:")
print(phase1.isna().sum().sort_values(ascending=False).head(20))
