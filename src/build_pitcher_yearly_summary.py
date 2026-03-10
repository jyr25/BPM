import pandas as pd
import numpy as np # np.nan 사용을 위해 추가

def build_pitcher_yearly_summary():
    df = pd.read_csv("data/processed/pitcher_game_logs.csv")
    df["year"] = df["game_id"].astype(str).str[:4].astype(int)
    
    summary = df.groupby(["player_id", "year"]).agg({
        "IP_real": "sum",
        "HR": "sum",
        "BB": "sum",
        "SO": "sum",
        "R" : "sum",
        "ER" : "sum",
        "H" : "sum",
        "ERA": "mean",
        "WHIP": "mean"
    }).reset_index()

    summary['year_fip'] = ((13 * summary["HR"] + 3 * summary["BB"] - 2 * summary["SO"]) / summary["IP_real"].replace(0, np.nan)) + 3.1
    summary['year_era'] = (summary['ER'] * 9) / summary['IP_real'].replace(0, np.nan)

    summary['prev_year'] = summary['year'] + 1
    summary['prev_era'] = (summary['ER'] * 9) / summary['IP_real'].replace(0, np.nan)
    summary['prev_whip'] = (summary['H'] + summary['BB']) / summary['IP_real'].replace(0, np.nan)
    summary['prev_kbb'] = summary['SO'] / summary['BB'].replace(0, np.nan)

    summary = summary[['player_id', 'prev_year', 'year_fip', 'year_era', 'IP_real']]
    summary.columns = ['player_id', 'year', 'prev_fip', 'prev_era', 'prev_ip_total']

    summary.to_csv("data/processed/pitcher_yearly_summary.csv", index=False)
    print("✅ 전년도 통계 요약본 생성 완료")

if __name__ == "__main__":
    build_pitcher_yearly_summary()