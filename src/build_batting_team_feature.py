import os
import json
import pandas as pd

RAW_DIR = "data/raw/team_batting"
OUTPUT = "data/processed/team_batting_features.csv"

rows = []

for file in os.listdir(RAW_DIR):
    if not file.endswith(".json"):
        continue

    if "full" not in file:
        continue

    path = os.path.join(RAW_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parts = file.replace(".json", "").split("_")
    year = parts[2]

    team_list = data.get("list", [])

    for t in team_list:
        rows.append({
            "year": str(t.get("year", year)),
            "team_code": str(t.get("t_code")),
            "PA": t.get("PA"),
            "BB": t.get("BB"),
            "SO": t.get("SO"),
            "AVG": t.get("AVG"),
            "SLG": t.get("SLG"),
            "OPS": t.get("OPS"),
        })

df = pd.DataFrame(rows)

num_cols = ["PA", "BB", "SO", "AVG", "SLG", "OPS"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["team_ops"] = df["OPS"]
df["team_bb_rate"] = df["BB"] / df["PA"]
df["team_k_rate"] = df["SO"] / df["PA"]
df["team_iso"] = df["SLG"] - df["AVG"]

final = df[[
    "year", "team_code",
    "team_ops", "team_bb_rate", "team_k_rate", "team_iso"
]].copy()

final.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("saved:", OUTPUT)
print(final.head())