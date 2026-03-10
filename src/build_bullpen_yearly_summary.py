import pandas as pd
import numpy as np

def build_team_bullpen_summary():
    df = pd.read_csv("data/processed/bullpen_game_logs.csv")
    df["year"] = df["game_id"].astype(str).str[:4].astype(int)
    
    # 1. 지표 합산 (H, BB, HR, SO, ER 등 모두 포함)
    team_summary = df.groupby(["team_code", "year"]).agg({
        "IP_real": "sum",
        "H": "sum",
        "BB": "sum",
        "HR": "sum",
        "ER": "sum",
        "SO": "sum"
    }).reset_index()
    
    # 2. 전년도 비율 지표 계산
    team_summary['prev_bullpen_era'] = (team_summary['ER'] * 9) / team_summary['IP_real'].replace(0, np.nan)
    team_summary['prev_bullpen_whip'] = (team_summary['H'] + team_summary['BB']) / team_summary['IP_real'].replace(0, np.nan)
    team_summary['prev_bullpen_hr9'] = (team_summary['HR'] * 9) / team_summary['IP_real'].replace(0, np.nan)
    team_summary['prev_bullpen_kbb'] = team_summary['SO'] / team_summary['BB'].replace(0, np.nan)
    team_summary['prev_bullpen_ops'] = ((team_summary['H'] + team_summary['BB'] + team_summary['HR']) / team_summary['IP_real'].replace(0, np.nan)) * 100
   
    # 3. 내년도 매칭을 위한 연도 보정
    team_summary['match_year'] = team_summary['year'] + 1
    
    # 필요한 컬럼만 추출
    cols = ['team_code', 'match_year', 'prev_bullpen_era', 'prev_bullpen_whip', 'prev_bullpen_hr9', 'prev_bullpen_kbb', 'prev_bullpen_ops']
    final_summary = team_summary[cols].copy()
    final_summary.columns = ['team_code', 'year', 'prev_bullpen_era', 'prev_bullpen_whip', 'prev_bullpen_hr9', 'prev_bullpen_kbb', 'prev_bullpen_ops']
    final_summary.to_csv("data/processed/team_bullpen_prev_summary.csv", index=False, encoding="utf-8-sig")
    print("✅ 팀별 불펜 과거 요약 업데이트 완료 (HR9, KBB, OPS 포함)")

if __name__ == "__main__":
    build_team_bullpen_summary()