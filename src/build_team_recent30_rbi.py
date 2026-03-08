import pandas as pd

BATTER_PATH = "data/processed/batter_game_logs.csv"
GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT = "data/processed/team_recent30_rbi_features.csv"

# -----------------------------
# 데이터 불러오기
# -----------------------------

batter = pd.read_csv(BATTER_PATH)
games = pd.read_csv(GAME_INDEX_PATH)

# 날짜 변환
games["game_date"] = pd.to_datetime(games["date"])

# 타자 로그에 날짜 붙이기
df = batter.merge(
    games[["game_id", "game_date"]],
    on="game_id",
    how="left"
)

# 숫자 변환
df["RBI"] = pd.to_numeric(df["RBI"], errors="coerce").fillna(0)

# -----------------------------
# 팀 경기별 RBI
# -----------------------------

team_game_rbi = (
    df.groupby(["team_code", "game_id", "game_date"])["RBI"]
    .sum()
    .reset_index()
)

team_game_rbi = team_game_rbi.sort_values(["team_code", "game_date"])

# -----------------------------
# 최근 30일 RBI 계산
# -----------------------------

result = []

for team, group in team_game_rbi.groupby("team_code"):

    group = group.sort_values("game_date")

    for idx, row in group.iterrows():

        current_date = row["game_date"]

        start_date = current_date - pd.Timedelta(days=30)

        mask = (
            (group["game_date"] < current_date) &
            (group["game_date"] >= start_date)
        )

        recent_rbi = group.loc[mask, "RBI"].sum()

        result.append({
            "game_id": row["game_id"],
            "team_code": team,
            "team_recent30_rbi": recent_rbi
        })

result_df = pd.DataFrame(result)

result_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT)
print("rows:", len(result_df))
print(result_df.head())