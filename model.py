# main.py
# Two-stage win probability model
# Stage 1: Pre-game features -> LR + XGB -> Stacking (OOF, Walk-forward)
# Stage 2: Lineup features (after lineup release) -> XGB adjustment

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# -----------------------------
# 0) Data Loading (example)
# -----------------------------
def load_data():
    """
    Expected columns (example):
      - date: YYYY-MM-DD (must be sortable)
      - y: 0/1 (home win = 1, lose = 0)
      - pregame features: FIP_diff, OPS_diff, bullpen_diff, ...
      - lineup features (optional): lineup_strength_diff, batting_order_sens_diff, ...
    """
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    lineup_path = "data/lineup.csv"  # optional for stage2 (after lineup release)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    lineup_df = None
    if os.path.exists(lineup_path):
        lineup_df = pd.read_csv(lineup_path)

    return train_df, test_df, lineup_df


# -----------------------------
# 1) Feature Config
# -----------------------------
PREGAME_FEATURES = [
    # Starter pitching
    "FIP_diff", "ERA_diff", "WHIP_diff", "KBB_diff", "IP_per_start_diff",
    "Rest_days_diff", "Recent3_FIP_diff",
    # Team hitting
    "OPS_diff", "BBp_diff", "Kp_diff", "ISO_diff", "Recent30_RBI_diff",
    # Bullpen
    "Bullpen_WHIP_diff", "HR9_diff", "Bullpen_KBB_diff", "Bullpen_pOPS_diff",
    # Others
    "vs_opp_winrate_diff", "park_winrate_diff",
]

LINEUP_FEATURES = [
    "lineup_strength_diff",     # weighted OPS(or wOBA) sum diff
    "batting_order_sens_diff",  # order sensitivity
    "position_sens_diff",       # position sensitivity
    "defense_sens_diff",        # defense effect
    "battery_sens",             # starter-catcher battery effect
]


def safe_select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Keep only existing columns (so code doesn't crash if some are missing)
    exist = [c for c in cols if c in df.columns]
    return df[exist].copy()


# -----------------------------
# 2) Walk-forward OOF for Stacking
# -----------------------------
def generate_walkforward_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 5,
    min_train_size_ratio: float = 0.5,
):
    """
    Returns list of (train_idx, valid_idx) with time-ordered splits.
    Simple version: sort by date, then split into n_splits consecutive validation blocks.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    min_train = int(n * min_train_size_ratio)

    # validation blocks after min_train
    remain = n - min_train
    block = max(1, remain // n_splits)

    splits = []
    start = min_train
    for i in range(n_splits):
        v_start = start + i * block
        v_end = min(n, v_start + block)
        if v_start >= n or v_end <= v_start:
            break

        train_idx = np.arange(0, v_start)
        valid_idx = np.arange(v_start, v_end)
        splits.append((train_idx, valid_idx))

    return df_sorted, splits


def stage1_oof_stacking(train_df: pd.DataFrame):
    """
    1) Train LR and XGB in walk-forward fashion
    2) Create OOF predictions for both models
    3) Train a meta model (Logistic Regression) on OOF preds
    """
    df_sorted, splits = generate_walkforward_splits(train_df, date_col="date", n_splits=5)

    y = df_sorted["y"].astype(int).values
    X = safe_select(df_sorted, PREGAME_FEATURES).fillna(0.0).values

    # Base models
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, n_jobs=None))
    ])

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    oof_lr = np.zeros(len(df_sorted), dtype=float)
    oof_xgb = np.zeros(len(df_sorted), dtype=float)

    for train_idx, valid_idx in splits:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[valid_idx], y[valid_idx]

        lr.fit(X_tr, y_tr)
        oof_lr[valid_idx] = lr.predict_proba(X_va)[:, 1]

        xgb.fit(X_tr, y_tr)
        oof_xgb[valid_idx] = xgb.predict_proba(X_va)[:, 1]

    # Meta model uses OOF predictions (stacking)
    meta_X = np.column_stack([oof_lr, oof_xgb])
    meta = LogisticRegression(max_iter=300)
    # Use only rows that actually got OOF predictions
    valid_mask = (oof_lr > 0) | (oof_xgb > 0)
    meta.fit(meta_X[valid_mask], y[valid_mask])

    # Refit base models on full training for final inference
    lr.fit(X, y)
    xgb.fit(X, y)

    artifacts = {
        "df_sorted": df_sorted,
        "lr": lr,
        "xgb": xgb,
        "meta": meta,
    }
    return artifacts


def stage1_predict(artifacts, df: pd.DataFrame) -> np.ndarray:
    X = safe_select(df, PREGAME_FEATURES).fillna(0.0).values
    p_lr = artifacts["lr"].predict_proba(X)[:, 1]
    p_xgb = artifacts["xgb"].predict_proba(X)[:, 1]
    meta_X = np.column_stack([p_lr, p_xgb])
    p_stage1 = artifacts["meta"].predict_proba(meta_X)[:, 1]
    return p_stage1


# -----------------------------
# 3) Stage 2: Lineup adjustment model
# -----------------------------
def train_stage2_model(train_df: pd.DataFrame, p_stage1_train: np.ndarray):
    """
    Train XGBoost that learns adjustment from lineup features + stage1 prob.
    Assumption: lineup features are available for training historical games too.
    """
    df = train_df.copy()
    df["p_stage1"] = p_stage1_train

    y = df["y"].astype(int).values
    X2 = safe_select(df, ["p_stage1"] + LINEUP_FEATURES).fillna(0.0).values

    stage2 = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    stage2.fit(X2, y)
    return stage2


def stage2_predict(stage2_model, df: pd.DataFrame, p_stage1: np.ndarray) -> np.ndarray:
    """
    If lineup features are missing at inference time, just return stage1 probs.
    """
    df2 = df.copy()
    df2["p_stage1"] = p_stage1

    needed = ["p_stage1"] + LINEUP_FEATURES
    has_any_lineup = any(col in df2.columns for col in LINEUP_FEATURES)

    if not has_any_lineup:
        return p_stage1

    X2 = safe_select(df2, needed).fillna(0.0).values
    p_final = stage2_model.predict_proba(X2)[:, 1]
    return p_final


# -----------------------------
# 4) Main Run
# -----------------------------
def main():
    train_df, test_df, lineup_df = load_data()

    # (Optional) merge lineup features into train/test if you store them separately
    # You need a key like game_id or (date, home_team, away_team)
    # Here is a placeholder merge:
    if lineup_df is not None:
        key_cols = [c for c in ["game_id"] if c in train_df.columns and c in lineup_df.columns]
        if key_cols:
            train_df = train_df.merge(lineup_df, on=key_cols, how="left")
            test_df = test_df.merge(lineup_df, on=key_cols, how="left")

    # Stage 1
    s1 = stage1_oof_stacking(train_df)
    p1_train = stage1_predict(s1, train_df)
    p1_test = stage1_predict(s1, test_df)

    # Stage 2 (only if lineup features exist in train)
    if any(col in train_df.columns for col in LINEUP_FEATURES):
        stage2 = train_stage2_model(train_df, p1_train)
        p_final = stage2_predict(stage2, test_df, p1_test)
    else:
        stage2 = None
        p_final = p1_test

    # Save submission
    # Expect test has "game_id" or similar identifier
    out = pd.DataFrame({
        "game_id": test_df["game_id"] if "game_id" in test_df.columns else np.arange(len(test_df)),
        "win_prob": p_final
    })
    os.makedirs("output", exist_ok=True)
    out.to_csv("output/submission.csv", index=False)
    print("Saved: output/submission.csv")


if __name__ == "__main__":
    main()
