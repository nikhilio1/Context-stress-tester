import pandas as pd

df = pd.read_csv("results/results.csv")

print(df.shape)
print(df["status"].value_counts())
print(df["length_tokens"].value_counts())

print(df.groupby(["length_tokens", "depth_percent"]).size())

extra = df[(df["length_tokens"] == 50000) & (df["depth_percent"] == 0)]
print(extra[["trial", "needle_type", "correct"]].to_string())

df = df[df["status"] == "OK"].copy()
df["is_correct"] = df["correct"].astype(str).str.lower() == "true"

key = ["model", "length_tokens", "depth_percent", "trial"]
before = len(df)
df = df.drop_duplicates(subset=key, keep="last")
print("dropped duplicates:", before - len(df))

print(df.groupby(["length_tokens", "depth_percent"]).size())

print("\nOverall accuracy:", df["is_correct"].mean(), "on", len(df), "trials")

print("\nAccuracy by length x needle type:")
print(df.groupby(["length_tokens", "needle_type"])["is_correct"].mean().unstack())

print("\nAccuracy by length x depth:")
print(df.groupby(["length_tokens", "depth_percent"])["is_correct"].mean().unstack())

print("\nMisses:")
print(df[~df["is_correct"]][["length_tokens", "depth_percent", "trial", "needle_type", "expected_answer", "model_answer"]].to_string())