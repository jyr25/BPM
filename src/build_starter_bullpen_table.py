import os
import pandas as pd

LINEUPS_PATH = "data/processed/lineups_flat.csv"
PITCHER_PATH = "data/processed/pitcher_game_logs.csv"

STARTER_OUT = "data/processed/starter_game_logs.csv"
BULLPEN_OUT = "data/processed/bullpen_game_logs.csv"


def convert_ip(ip):
    """
    KBO식 이닝 표기 변환
    예:
    5   -> 5.0
    5.1 -> 5 + 1/3
    5.2 -> 5 + 2/3
    """
    if pd.isna(ip):
        return pd.NA

    ip = str(ip)

    if "." not in ip:
        try:
            return float(ip)
        except:
            return pd.NA

    whole, frac = ip.split(".")
    try:
        whole = int(whole)
        frac = int(frac)
    except:
        return pd.NA

    if frac == 0:
        return float(whole)
    elif frac == 1:
        return whole + 1 / 3
    elif frac == 2:
        return whole + 2 / 3
    else:
        return pd.NA


def main():
    # 1) 라인업 데이터 로드
    lineups = pd.read_csv(LINEUPS_PATH)

    # 타입 맞추기
    lineups["game_id"] = lineups["game_id"].astype(str)
    lineups["player_id"] = lineups["player_id"].astype(str)
    lineups["team_code"] = lineups["team_code"].astype(str)

    # 2) 실제 경기 선발투수 추출
    # position == 1 이 선발투수
    starters = lineups[
        (lineups["position"] == 1) &
        (lineups["starting"] == "Y")
    ][["game_id", "player_id", "team_code"]].copy()

    starters["is_starter_actual"] = 1

    # 3) 투수 게임 로그 로드
    pitcher = pd.read_csv(PITCHER_PATH)

    pitcher["game_id"] = pitcher["game_id"].astype(str)
    pitcher["player_id"] = pitcher["player_id"].astype(str)
    pitcher["team_code"] = pitcher["team_code"].astype(str)

    # 숫자형 변환
    numeric_cols = [
        "G", "GS", "IP", "R", "ER", "TBF", "H", "HR", "BB", "SO", "NP",
        "ERA", "WHIP", "OBP", "SLG", "OPS"
    ]

    for col in numeric_cols:
        if col in pitcher.columns:
            pitcher[col] = pd.to_numeric(pitcher[col], errors="coerce")

    # 실제 이닝 계산용 컬럼 추가
    if "IP_real" not in pitcher.columns and "IP" in pitcher.columns:
        pitcher["IP_real"] = pitcher["IP"].apply(convert_ip)

    # 4) 경기별 실제 선발투수 매칭
    pitcher = pitcher.merge(
        starters,
        on=["game_id", "player_id", "team_code"],
        how="left"
    )

    pitcher["is_starter_actual"] = pitcher["is_starter_actual"].fillna(0).astype(int)
    pitcher["role_game"] = pitcher["is_starter_actual"].map({1: "starter", 0: "bullpen"})

    # 5) starter / bullpen 분리
    starter_df = pitcher[pitcher["role_game"] == "starter"].copy()
    bullpen_df = pitcher[pitcher["role_game"] == "bullpen"].copy()

    # 6) 저장
    os.makedirs("data/processed", exist_ok=True)

    starter_df.to_csv(STARTER_OUT, index=False, encoding="utf-8-sig")
    bullpen_df.to_csv(BULLPEN_OUT, index=False, encoding="utf-8-sig")

    print("saved:", STARTER_OUT, len(starter_df))
    print("saved:", BULLPEN_OUT, len(bullpen_df))

    print("\nstarter sample")
    print(starter_df.head())

    print("\nbullpen sample")
    print(bullpen_df.head())

    print("\nrole counts")
    print(pitcher["role_game"].value_counts())


if __name__ == "__main__":
    main()