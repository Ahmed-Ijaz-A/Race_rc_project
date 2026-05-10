
import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("data/raw", exist_ok=True)

# ── Load source ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/source.csv")

print(f"Total rows   : {len(df)}")
print(f"Columns      : {df.columns.tolist()}")
print(f"Null counts  :\n{df.isnull().sum()}")
print(f"\nAnswer distribution:\n{df['answer'].value_counts(normalize=True)}")

# ── Sanity-check required columns ────────────────────────────────────────────
REQUIRED_COLS = {"id", "article", "question", "A", "B", "C", "D", "answer"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise ValueError(f"source.csv is missing columns: {missing}")

# ── Drop rows with any null in key columns ────────────────────────────────────
before = len(df)
df = df.dropna(subset=list(REQUIRED_COLS))
print(f"\nDropped {before - len(df)} rows with nulls. Remaining: {len(df)}")

# ── Add a true unique row key ─────────────────────────────────────────────────
# NOTE: the 'id' column in RACE is the ARTICLE id, not a unique row id.
# Multiple questions share the same article id, so we cannot use it for
# overlap checking. We create row_id from the integer index instead.
df = df.reset_index(drop=True)
df["row_id"] = df.index   # 0 … N-1, guaranteed unique per row

# ── Step 1 : split off 10 % test ─────────────────────────────────────────────
train_val, test_df = train_test_split(
    df,
    test_size=0.10,
    stratify=df["answer"],
    random_state=42,
)

# ── Step 2 : split remaining into 80 % train / 10 % val (of total) ───────────
train_df, val_df = train_test_split(
    train_val,
    test_size=0.111,
    stratify=train_val["answer"],
    random_state=42,
)

# ── Verify zero overlap using row_id (unique per row) ────────────────────────
train_ids = set(train_df["row_id"])
val_ids   = set(val_df["row_id"])
test_ids  = set(test_df["row_id"])

assert len(train_ids & val_ids)  == 0, "❌ Overlap between train and val!"
assert len(train_ids & test_ids) == 0, "❌ Overlap between train and test!"
assert len(val_ids   & test_ids) == 0, "❌ Overlap between val and test!"
print(" No overlap confirmed.")

# ── Save (drop the helper column before saving) ───────────────────────────────
train_df.drop(columns=["row_id"]).to_csv("data/raw/train.csv", index=False)
val_df.drop(  columns=["row_id"]).to_csv("data/raw/val.csv",   index=False)
test_df.drop( columns=["row_id"]).to_csv("data/raw/test.csv",  index=False)

total = len(train_df) + len(val_df) + len(test_df)
print(f"\nSplit summary")
print(f"  Train : {len(train_df):>7,}  ({len(train_df)/total:.1%})")
print(f"  Val   : {len(val_df):>7,}  ({len(val_df)/total:.1%})")
print(f"  Test  : {len(test_df):>7,}  ({len(test_df)/total:.1%})")
print(f"  Total : {total:>7,}")

for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{name} answer distribution:")
    print(split["answer"].value_counts(normalize=True).round(3).to_string())

print("\n Splits saved. Now run:")
print("   git add data/raw/train.csv data/raw/val.csv data/raw/test.csv")
print("   git commit -m 'Add 80/10/10 stratified splits'")
print("   git push origin main")
