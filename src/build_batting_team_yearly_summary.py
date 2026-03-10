import pandas as pd
import numpy as np

def build_batting_team_yearly_summary():
    # 1. 타자 게임 로그 로드
    df = pd.read_csv("data/processed/batter_game_logs.csv")
    
    # game_id에서 연도 추출 (문자열 변환 후 슬라이싱)
    df["year"] = df["game_id"].astype(str).str[:4].astype(int)
    
    # 2. 팀별/연도별 누적 합계 계산
    # AVG, OBP 등은 단순 평균하면 타석 수 차이 때문에 왜곡되므로 합계 데이터로 재계산해야 함
    summary = df.groupby(["team_code", "year"]).agg({
        "PA": "sum",
        "AB": "sum",
        "H": "sum",
        "HR": "sum",
        "BB": "sum"
        # SLG 계산을 위해 2루타, 3루타가 있다면 좋지만 없으므로 
        # 제공된 OPS 등을 활용하거나 기본 지표 위주로 요약
    }).reset_index()
    
    # 3. 팀 단위 지표 재계산 (비율 지표)
    # 타율(AVG) 재계산
    summary['year_batting_avg'] = summary['H'] / summary['AB'].replace(0, np.nan)
    
    # 홈런율 (타석당 홈런)
    summary['year_hr_rate'] = summary['HR'] / summary['PA'].replace(0, np.nan)
    
    # 볼넷율 (타석당 볼넷)
    summary['year_bb_rate'] = summary['BB'] / summary['PA'].replace(0, np.nan)

    # 4. 내년도 매칭을 위한 연도 보정
    summary['match_year'] = summary['year'] + 1
    
    # 필요한 컬럼만 추출 및 이름 변경
    # 예측 모델에서 '기준점'으로 쓸 피처들입니다.
    final_summary = summary[[
        'team_code', 'match_year', 'year_batting_avg', 'year_hr_rate', 'year_bb_rate'
    ]]
    final_summary.columns = [
        'team_code', 'year', 'prev_team_avg', 'prev_team_hr_rate', 'prev_team_bb_rate'
    ]
    
    # 5. 저장
    output_path = "data/processed/team_batting_prev_summary.csv"
    final_summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 팀별 타격 과거 요약 생성 완료: {output_path}")

if __name__ == "__main__":
    build_batting_team_yearly_summary()