import pandas as pd
import json
import os

# Path to your full dataset
CSV_PATH = os.path.expanduser("~/bb/bizbotapp/data/bitext_full.csv")

# Where to save the sample JSON
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "bitext_sample.json")

# Load dataset
df = pd.read_csv(CSV_PATH)

# Optional: filter rows if you want more specific tone examples
df = df[["instruction", "response"]].dropna()

# Randomly sample 10 entries
sample = df.sample(n=10, random_state=42)

# Convert to list of dicts
sample_records = sample.to_dict(orient="records")

# Write JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(sample_records, f, indent=2)

print(f"✅ Sample saved to {OUTPUT_PATH}")
