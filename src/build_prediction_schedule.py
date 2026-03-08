import pandas as pd

GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/prediction_schedule.csv"

df = pd.read_csv(GAME_INDEX_PATH)

# datetime 생성
df["game_datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["game_time"].astype(str),
    errors="coerce"
)

# 1차 예측 기준 (당일 00:00)
df["phase1_time"] = df["game_datetime"].dt.normalize()

# 라인업 확인 시작
df["lineup_poll_start"] = df["game_datetime"] - pd.Timedelta(minutes=90)

# 최종 제출 (경기 30분 전)
df["submission_deadline"] = df["game_datetime"] - pd.Timedelta(minutes=30)

# 참고용 (실제 대회 cutoff)
df["official_deadline"] = df["game_datetime"] - pd.Timedelta(minutes=15)

df.to_csv(OUTPUT_PATH, index=False)

print("saved:", OUTPUT_PATH)
print(df[[
    "game_id",
    "game_datetime",
    "phase1_time",
    "lineup_poll_start",
    "submission_deadline"
]].head())