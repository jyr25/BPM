import os
import glob
import json
import pandas as pd

SCHEDULE_DIR = "data/raw/games"
OUTPUT_PATH = "data/processed/game_index.csv"


def load_schedule_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    games = []

    # JSON 구조: {"1102":[{...}], "1103":[{...}]}
    for key, game_list in raw.items():
        if not isinstance(game_list, list):
            continue

        for g in game_list:
            games.append(g)

    return pd.DataFrame(games)


def build_game_index():

    files = sorted(glob.glob(os.path.join(SCHEDULE_DIR, "games_*.json")))

    dfs = []

    for f in files:
        df = load_schedule_file(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # ⭐ 정규시즌만 선택
    df = df[df["leagueType"] == 10100]

    # 날짜 생성
    df["date"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(str).str.zfill(2),
        errors="coerce"
    )

    # 컬럼 이름 정리
    rename_map = {
        "s_no": "game_id",
        "state": "status",
        "awayTeam": "away_team",
        "homeTeam": "home_team",
        "awaySP": "away_sp_id",
        "homeSP": "home_sp_id",
        "awaySPName": "away_sp_name",
        "homeSPName": "home_sp_name",
        "awayScore": "away_score",
        "homeScore": "home_score",
        "weather": "weather_code",
        "windDirection": "wind_direction",
        "windSpeed": "wind_speed",
        "rainprobability": "rain_probability",
        "hm": "game_time",
        "s_code": "Stadium_code",
    }

    df = df.rename(columns=rename_map)

    keep_cols = [
        "date",
        "game_id",
        "status",
        "game_time",
        "Stadium_code",
        "away_team",
        "home_team",
        "away_sp_id",
        "home_sp_id",
        "away_sp_name",
        "home_sp_name",
        "away_score",
        "home_score",
        "weather_code",
        "temperature",
        "humidity",
        "wind_direction",
        "wind_speed",
        "rain_probability",
        "leagueType"
    ]

    game_index = df[[c for c in keep_cols if c in df.columns]].copy()

    game_index["season"] = game_index["date"].dt.year
    game_index["game_id"] = game_index["game_id"].astype(str)

    # 중복 제거
    game_index = game_index.drop_duplicates(subset=["game_id"])

    # 정렬
    game_index = game_index.sort_values(["date", "game_id"]).reset_index(drop=True)

    # 홈팀 승 여부
    if "home_score" in game_index.columns and "away_score" in game_index.columns:
        game_index["target_home_win"] = (
            game_index["home_score"] > game_index["away_score"]
        ).astype("Int64")

    os.makedirs("data/processed", exist_ok=True)

    game_index.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("game_index 생성 완료")
    print(game_index.head())

    print("\n시즌별 경기 수")
    print(game_index.groupby("season")["game_id"].count())


if __name__ == "__main__":
    build_game_index()