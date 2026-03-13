import pandas as pd
import numpy as np # np.nan 사용을 위해 추가

def build_pitcher_yearly_summary():
    df = pd.read_csv("data/processed/starter_game_logs.csv")
    df["year"] = df["game_id"].astype(str).str[:4].astype(int)
    
    summary = df.groupby(["player_id", "year"]).agg({
        "IP_real": "sum",
        "HR": "sum",
        "BB": "sum",
        "SO": "sum",
        "ER" : "sum",
        "H" : "sum"
    }).reset_index()
    
    # 2. 정확한 비율 스탯 계산
    summary['prev_fip'] = ((13 * summary["HR"] + 3 * summary["BB"] - 2 * summary["SO"]) / summary["IP_real"].replace(0, np.nan)) + 3.1
    summary['prev_era'] = (summary['ER'] * 9) / summary['IP_real'].replace(0, np.nan)
    summary['prev_whip'] = (summary['H'] + summary['BB']) / summary['IP_real'].replace(0, np.nan)
    summary['prev_kbb'] = summary['SO'] / summary['BB'].replace(0, np.nan)

    # 3. 다음 연도와 매칭하기 위해 year를 +1로 변경
    summary['year'] = summary['year'] + 1 
    
    # 4. 필요한 컬럼만 추출 (나중에 쓸 지표들 모두 포함)
    final_summary = summary[['player_id', 'year', 'prev_fip', 'prev_era', 'prev_whip', 'prev_kbb', 'IP_real']]
    final_summary.columns = ['player_id', 'year', 'prev_fip', 'prev_era', 'prev_whip', 'prev_kbb', 'prev_ip_total']

    final_summary.to_csv("data/processed/pitcher_yearly_summary.csv", index=False)
    print("✅ 전년도 통계 요약본 생성 완료")

if __name__ == "__main__":
    build_pitcher_yearly_summary()