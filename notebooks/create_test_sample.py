"""
Create stratified test sample (500 examples) for fair comparison across all models.
Seed=42 for reproducibility.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SAMPLE_SIZE = 500

# Load full test set
df_test = pd.read_csv("../data/df_test.csv")
print(f"Full test set: {len(df_test)} samples")
print(f"Class distribution:\n{df_test['toxic_level'].value_counts().sort_index()}")

# Stratified sample - keep original indices
df_test_indexed = df_test.copy()
df_test_indexed['original_idx'] = df_test_indexed.index

sample_indices, _ = train_test_split(
    df_test_indexed.index.tolist(),
    train_size=SAMPLE_SIZE,
    stratify=df_test['toxic_level'],
    random_state=SEED
)
sample_indices = sorted(sample_indices)

df_sample = df_test.iloc[sample_indices].copy()
df_sample['original_idx'] = sample_indices
df_sample = df_sample.reset_index(drop=True)

print(f"\nSample: {len(df_sample)} samples")
print(f"Class distribution:\n{df_sample['toxic_level'].value_counts().sort_index()}")

# Save
df_sample.to_csv("../data/df_test_sample.csv", index=False)
print(f"\nSaved to ../data/df_test_sample.csv")

# Save indices for extracting BERT predictions
with open("../data/sample_indices.txt", "w") as f:
    f.write(str(sample_indices))
print(f"Saved {len(sample_indices)} indices to ../data/sample_indices.txt")
