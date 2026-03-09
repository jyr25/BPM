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
    for key, game_list in raw.items():
        if not isinstance(game_list, list):
            continue
        games.extend(game_list)
    return pd.DataFrame(games)

def build_game_index():
    files = sorted(glob.glob(os.path.join(SCHEDULE_DIR, "games_*.json")))
    if not files:
        print("파일을 찾을 수 없습니다.")
        return

    dfs = []
    for f in files:
        temp_df = load_schedule_file(f)
        temp_df["source_file"] = os.path.basename(f)
        dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)

    # 1. 필터링 및 날짜 생성
    df = df[df["leagueType"] == 10100].copy()
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-" +
        df["day"].astype(str).str.zfill(2),
        errors="coerce"
    )

    # 2. 컬럼명 변경
    rename_map = {
        "s_no": "game_id", "state": "status", "awayTeam": "away_team", "homeTeam": "home_team",
        "awaySP": "away_sp_id", "homeSP": "home_sp_id", "awaySPName": "away_sp_name",
        "homeSPName": "home_sp_name", "awayScore": "away_score", "homeScore": "home_score",
        "weather": "weather_code", "windDirection": "wind_direction", "windSpeed": "wind_speed",
        "rainprobability": "rain_probability", "hm": "game_time", "s_code": "Stadium_code"
    }
    df = df.rename(columns=rename_map)

    # 3. 데이터 타입 변환 (필수 컬럼 위주)
    for col in ["home_score", "away_score", "temperature", "humidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. 대상 컬럼 추출 및 정제
    keep_cols = [
        "date", "game_id", "status", "game_time", "Stadium_code", "away_team", "home_team",
        "away_sp_id", "home_sp_id", "away_sp_name", "home_sp_name", "away_score", "home_score",
        "weather_code", "temperature", "humidity", "wind_direction", "wind_speed", 
        "rain_probability", "leagueType"
    ]
    game_index = df[[c for c in keep_cols if c in df.columns]].copy()
    game_index["season"] = game_index["date"].dt.year
    game_index["game_id"] = game_index["game_id"].astype(str)

    # 5. 중복 제거 및 결측치 처리 (경기 결과가 없는 데이터는 예측 학습에 쓸 수 없음)
    game_index = game_index.drop_duplicates(subset=["game_id"])
    game_index = game_index.dropna(subset=["home_score", "away_score"])

    # 6. 승패 및 무승부 라벨링 (효율적인 loc 방식 사용)
    # 기본값 0 (홈패)
    game_index["target_home_win"] = 0.0
    # 홈승 (1)
    game_index.loc[game_index["home_score"] > game_index["away_score"], "target_home_win"] = 1.0
    # 홈패 (0.0)
    game_index.loc[game_index["home_score"] < game_index["away_score"], "target_home_win"] = 0.0
    # 무승부 (0.5)
    game_index.loc[game_index["home_score"] == game_index["away_score"], "target_home_win"] = 0.5
    
    game_index["is_draw"] = (game_index["home_score"] == game_index["away_score"]).astype(int)

    # 7. 정렬 및 저장
    game_index = game_index.sort_values(["date", "game_id"]).reset_index(drop=True)
    
    os.makedirs("data/processed", exist_ok=True)
    game_index.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ game_index 생성 완료: {OUTPUT_PATH}")
    print(f"전체 경기 수: {len(game_index)}")
    print("\n시즌별 경기 수 및 무승부 비율:")
    stats = game_index.groupby("season").agg({'game_id': 'count', 'is_draw': 'mean'})
    print(stats)

if __name__ == "__main__":
    build_game_index()