# BPM
KBO 승률 예측 데이터 파이프라인

본 프로젝트는 Statiz API 데이터를 활용하여 KBO 경기 승률 예측 모델을 위한 데이터셋을 구축하는 것을 목표로 합니다.

데이터 파이프라인은 다음과 같은 단계로 구성됩니다.
API 수집 → Raw Data 저장 → 전처리 → Feature Engineering → Dataset 생성 → Train/Test 분리

1. 프로젝트 구조

BPM/
│
├─ data
│   ├─ raw
│   │   ├─ lineups
│   │   ├─ player_day
│   │   ├─ roster
│   │   └─ team_batting
│   │
│   └─ processed
│       ├─ game_index.csv
│       ├─ batter_game_logs.csv
│       ├─ pitcher_game_logs.csv
│       ├─ starter_game_logs.csv
│       ├─ bullpen_game_logs.csv
│       ├─ starter_features.csv
│       ├─ bullpen_team_features.csv
│       ├─ team_batting_features.csv
│       ├─ lineup_features.csv
│       ├─ order_fit_features.csv
│       ├─ position_fit_features.csv
│       ├─ battery_features.csv
│       ├─ context_features.csv
│       ├─ phase1_dataset.csv
│       ├─ phase2_dataset.csv
│       ├─ train.csv
│       └─ test.csv
│
└─ src
    ├─ fetch_lineups.py
    ├─ fetch_player_day.py
    ├─ fetch_roster.py
    ├─ fetch_team_batting.py
    │
    ├─ build_lineups_table.py
    ├─ build_batter_table.py
    ├─ build_pitcher_table.py
    ├─ split_pitcher_roles.py
    │
    ├─ build_starter_features.py
    ├─ build_bullpen_team_features.py
    ├─ build_batting_team_feature.py
    │
    ├─ build_lineup_features.py
    ├─ build_order_fit_features.py
    ├─ build_position_fit_features.py
    ├─ build_battery_features.py
    │
    ├─ build_context_features.py
    ├─ build_phase1_dataset.py
    ├─ build_phase2_dataset.py
    └─ build_train_test_split.py

2. 데이터 수집

# 경기 일정
경기 기본 정보 (팀, 날짜, 선발, 결과 등)
game_index.csv

포함 정보
- 경기 날짜
- 홈/원정 팀
- 선발 투수
- 날씨
- 경기 결과

# 라인업
API
prediction/gameLineup

수집 코드
fetch_lineups.py

Raw 저장
data/raw/lineups

전처리
build_lineups_table.py

결과
lineups_flat.csv

# 선수 경기 기록
API
prediction/playerDay

수집 코드
fetch_player_day.py

Raw
data/raw/player_day

# 3. 선수 데이터 전처리
# 타자 기록
build_batter_table.py

생성파일
batter_game_logs.csv

포함 지표
- PA
- AB
- H
- HR
- RBI
- BB
- SO
- AVG
- OBP
- SLG
- OPS

# 투수 기록
build_pitcher_table.py

생성
pitcher_game_logs.csv

# 선발 / 불펜 분리
split_pitcher_rolse.py

기준
GS > 0 → starter
GS = 0 → bullpen

생성
starter_game_logs.csv
bullpen_game_logs.csv

# 4. Feature Engineering
# 선발 투수 Feature
build_starter_features.py

포함 변수
- FIP
- ERA
- WHIP
- K/BB
- IP/start
- Rest days
- Recent 3 game FIP

# 불펜 Feature
build_bullpen_team_features.py

팀 단위 집계
- WHIP
- HR/9
- K/BB
- ERA
- 피OPS
- 최근 7경기 불펜 성적

# 팀 타격 Feature
build_batting_team_feature.py

포함 변수
- WHIP
- HR/9
- K/BB
- ERA
- 피OPS
- 최근 7경기 불펜 성적

# 5. Phase2 Feature (라인업 기반)
# 라인업 공격력
build_lineup_features.py

타순 가중치
1번 1.10
2번 1.08
3번 1.05
4번 1.03
5번 1.01
6번 0.98
7번 0.95
8번 0.93
9번 0.89

라인업 공격력
weighted OPS

# 타순 적합성
build_order_fit_features.py
특정 타순에서 OPS 변화 측정

# 포지션 적합성
build_position_fit_features.py
특정 포지션 출전 시 OPS 변화

# 배터리 조합
build_battery_features.py
포수 + 투수 조합 경험

# 6. Context Feature
build_context_features.py

포함 변수
- 상대전적 : vs_team_winrate_diff
- 구장 승률 : park_winrate_diff
- 최근 10경기 승률 : recent10_winrate_diff

# 7. Phase1 Dataset
build_phase1_dataset.py

1차 예측 데이터

구성
- 선발투수
- 팀 타격
- 팀 불펜
- 기타 context

결과
phase1_dataset.csv

# 8. Phase2 Dataset
build_phase2_dataset.py
Phase1 + 라인업 Feature

추가 변수
- lineup_weighted_ops_diff
- order_fit_diff
- position_fit_diff
- battery_feature

결과
phase2_dataset.csv

# 9. Train / Test Dataset
build_train_test_split.py

분할 방식
Time-based split

비율
Train 80%
Test 20%

결과
train.csv
test.csv

# 10. 최종 Feature 목록

# Phase1
sp_fip_diff
sp_era_diff
sp_whip_diff
sp_kbb_diff
sp_ip_per_start_diff
sp_rest_days_diff
sp_recent3_fip_diff

bullpen_whip_diff
bullpen_hr9_diff
bullpen_kbb_diff
bullpen_era_diff
bullpen_ops_allowed_diff

team_ops_diff
team_bb_rate_diff
team_k_rate_diff
team_iso_diff
team_recent30_rbi_diff

# Phase2 추가
lineup_weighted_ops_diff
team_order_fit_score_diff
team_position_fit_score_diff
battery_games_diff
recent10_winrate_diff
vs_team_winrate_diff
park_winrate_diff

# 11. Target
target_home_win

값
1 = 홈팀 승리
0 = 원정팀 승리

# 12. 데이터 규모
전체 경기: 약 2400 경기
feature: 약 48개
train: 1939
test: 485

# 13. 모델 단계 (추후)

예상 모델
Phase1
Logistic Regression

Phase2
XGBoost / LightGBM

# 마지막 정리

현재 repository에는
- 데이터 수집
- 전처리
- Feature Engineering
- Dataset 생성
까지 완료된 상태입니다.
이 데이터를 기반으로 승률 예측 모델을 학습할 수 있습니다.
