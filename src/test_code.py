import pandas as pd

files = {
    "phase1": "data/processed/phase1_dataset.csv",
    "phase2": "data/processed/phase2_dataset.csv",
    "train": "data/processed/train.csv",
    "test": "data/processed/test.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"\n===== {name} =====")
    print("rows, cols:", df.shape)
    print("columns:", df.columns.tolist()[:10], "...")

    if "target_home_win" in df.columns:
        print("target values:", sorted(df["target_home_win"].dropna().unique().tolist()))

    print("missing top 10:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    print("num diff cols:", len(diff_cols))

    if diff_cols:
        print("sample diff stats:")
        print(df[diff_cols].describe().loc[["mean", "std", "min", "max"]].T.head(10))

# train/test row check
phase2 = pd.read_csv(files["phase2"])
train = pd.read_csv(files["train"])
test = pd.read_csv(files["test"])

print("\n===== split check =====")
print("phase2 rows:", len(phase2))
print("train + test rows:", len(train) + len(test))
print("same:", len(phase2) == len(train) + len(test))

# date order check
for name, df in [("train", train), ("test", test)]:
    if "date" in df.columns:
        temp = pd.to_datetime(df["date"], errors="coerce")
        print(f"{name} min date:", temp.min(), "max date:", temp.max())