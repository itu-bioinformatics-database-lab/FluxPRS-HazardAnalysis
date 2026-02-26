
import pandas as pd
# Load both CSV files
prs_scores = pd.read_csv('../data/raw/prs_scores.csv')               # Contains individual IDs and scores
rosmap_clinical = pd.read_csv('../data/raw/rosmap_clinical.csv')     # Contains ID, Study, and projid

# Assuming the common column for matching is 'ID' (adjust if named differently)
# If the column names differ, rename accordingly before merging
# e.g., rosmap_clinical.rename(columns={'RID': 'ID'}, inplace=True)

# Merge on individual ID
rosmap_clinical['sample_id'] = rosmap_clinical['Study'].astype(str) + rosmap_clinical['projid'].astype(str)
rosmap_clinical["sample_id"] = rosmap_clinical["sample_id"].astype(str)
prs_scores["sample_id"] = prs_scores["sample_id"].astype(str)
merged = prs_scores.merge(rosmap_clinical, on='sample_id', how='inner')
merged = merged[['sample_id', 'score', 'individualID']]

# Save the updated DataFrame to a new CSV
merged.to_csv('../data/raw/prs_scores_with_sample_ids.csv', index=False)