import pandas as pd
import os

GAME_INDEX_PATH = "data/processed/game_index.csv"
OUTPUT_PATH = "data/processed/prediction_schedule.csv"

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df = pd.read_csv(GAME_INDEX_PATH)

# 1. Robust Datetime Conversion
# Added format inference and error checking
df["game_datetime"] = pd.to_datetime(
    df["date"].astype(str) + " " + df["game_time"].astype(str),
    errors="coerce"
)

# BUG PREVENTION: Drop rows where datetime failed to parse
initial_count = len(df)
df = df.dropna(subset=["game_datetime"])
if len(df) < initial_count:
    print(f"⚠️ Warning: Dropped {initial_count - len(df)} rows due to invalid date/time formats.")

# 2. Logic Calculations
# Phase 1: Morning of the game (00:00)
df["phase1_time"] = df["game_datetime"].dt.normalize()

# Lineup Check: 90 mins before
df["lineup_poll_start"] = df["game_datetime"] - pd.Timedelta(minutes=90)

# Final Submission: 30 mins before
df["submission_deadline"] = df["game_datetime"] - pd.Timedelta(minutes=30)

# Official Cutoff: 15 mins before
df["official_deadline"] = df["game_datetime"] - pd.Timedelta(minutes=15)

# 3. Sort by Game Time (Useful for sequential processing)
df = df.sort_values("game_datetime").reset_index(drop=True)

# Save and Preview
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Schedule saved to: {OUTPUT_PATH}")
print(df[["game_id", "game_datetime", "lineup_poll_start", "submission_deadline"]].head())