import pandas as pd

# Show full text without truncation
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv("datasets/OEQ_sample.csv")
# Drop any accidental index columns like "Unnamed: 0"
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

# 1) Full table (untruncated)
print("\n==== Full OEQ_sample (untruncated) ====")
print(df.to_string(index=False))

# 2) Per-row readable view
print("\n==== Per-row detail ====")
for i, row in df.iterrows():
    print(f"\n--- Row {i} ---")
    if "source" in df.columns:
        print(f"source: {row['source']}")
    print("prompt:\n" + str(row.get("prompt", "")))
    print("human:\n" + str(row.get("human", "")))
    if "emotional_validation_human" in df.columns:
        print(f"emotional_validation_human: {row['emotional_validation_human']}")
    if "indirect_language_human" in df.columns:
        print(f"indirect_language_human: {row['indirect_language_human']}")
    if "indirect_action_human" in df.columns:
        print(f"indirect_action_human: {row['indirect_action_human']}")
    if "accept_framing_human" in df.columns:
        print(f"accept_framing_human: {row['accept_framing_human']}")