import pandas as pd
import numpy as np

def build_batting_team_yearly_summary():
    # 1. 타자 게임 로그 로드 (추가된 H2, H3, R, SO 등이 포함된 파일)
    df = pd.read_csv("data/processed/batter_game_logs.csv")
    
    # game_id에서 연도 추출
    df["year"] = df["game_id"].astype(str).str[:4].astype(int)
    
    # 2. 팀별/연도별 누적 합계 계산 (더 정교한 지표를 위해 컬럼 확장)
    # H2(2루타), H3(3루타), SO(삼진), HBP(사구), SF(희플) 등이 로그에 포함되어 있어야 합니다.
    agg_dict = {
        "PA": "sum", "AB": "sum", "H": "sum", "HR": "sum", 
        "BB": "sum", "SO": "sum", "H2": "sum", "H3": "sum",
        "HBP": "sum", "SF": "sum", "R": "sum"
    }
    
    # 로그에 해당 컬럼이 있는지 확인 후 있는 것만 집계
    existing_cols = [c for c in agg_dict.keys() if c in df.columns]
    summary = df.groupby(["team_code", "year"])[existing_cols].sum().reset_index()
    
    # 3. 팀 단위 지표 재계산 (비율 지표)
    # (1) 기본 타율
    summary['year_avg'] = summary['H'] / summary['AB'].replace(0, np.nan)
    
    # (2) 출루율 (OBP): (H + BB + HBP) / (AB + BB + HBP + SF)
    denom_obp = (summary['AB'] + summary['BB'] + summary.get('HBP', 0) + summary.get('SF', 0))
    summary['year_obp'] = (summary['H'] + summary['BB'] + summary.get('HBP', 0)) / denom_obp.replace(0, np.nan)
    
    # (3) 장타율 (SLG): (1루타*1 + 2루타*2 + 3루타*3 + 홈런*4) / AB
    # 1루타(H1) 계산: 전체안타(H) - 2루타 - 3루타 - 홈런
    h1 = summary['H'] - summary.get('H2', 0) - summary.get('H3', 0) - summary['HR']
    total_bases = (h1 * 1) + (summary.get('H2', 0) * 2) + (summary.get('H3', 0) * 3) + (summary['HR'] * 4)
    summary['year_slg'] = total_bases / summary['AB'].replace(0, np.nan)
    
    # (4) OPS 및 IsoP (순장타율)
    summary['year_ops'] = summary['year_obp'] + summary['year_slg']
    summary['year_isop'] = summary['year_slg'] - summary['year_avg']
    
    # (5) 선구안 및 득점력
    summary['year_bb_k'] = summary['BB'] / summary['SO'].replace(0, np.nan) # 볼삼비
    summary['year_run_per_pa'] = summary['R'] / summary['PA'].replace(0, np.nan) # 타석당 득점력

    # 4. 내년도 매칭을 위한 연도 보정 (Year-1 스탯으로 활용하기 위함)
    summary['match_year'] = summary['year'] + 1
    
    # 모델에 사용할 핵심 피처 선택 및 이름 변경
    feature_map = {
        'team_code': 'team_code',
        'match_year': 'year',
        'year_avg': 'prev_team_avg',
        'year_ops': 'prev_team_ops',
        'year_isop': 'prev_team_isop',
        'year_bb_k': 'prev_team_bb_k',
        'year_run_per_pa': 'prev_team_run_rate'
    }
    
    final_summary = summary[list(feature_map.keys())].copy()
    final_summary.columns = [feature_map[c] for c in final_summary.columns]
    
    # 5. 저장
    output_path = "data/processed/team_batting_prev_summary.csv"
    final_summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 팀별 타격 과거 요약 생성 완료: {output_path}")
    print(final_summary.head())

if __name__ == "__main__":
    build_batting_team_yearly_summary()