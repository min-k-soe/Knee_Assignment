import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_list = [] # pd.DataFrame(columns=['xray', 'masks'])
xrays_dir = "data/xrays/"

for xray in os.listdir(xrays_dir):
    xray_path = os.path.join(xrays_dir, xray)
    mask_path = xray_path.replace("xrays", "masks")
    if not os.path.exists(mask_path):
        mask_path = None


    data_list.append((xray_path, mask_path))

data_df = pd.DataFrame(data_list, columns=['xray', 'masks'])
data_df.to_csv("data/CSVs/dataset.csv", index=False)
print()

test_df = data_df[data_df['masks'].isna()]

labeled_df = data_df[data_df['masks'].notna()]

# 80% train (80 images), 20% validation (20 images)
train_df, val_df = train_test_split(labeled_df, test_size=0.20, random_state=42)

train_df.to_csv("data/CSVs/train.csv", index=False)
val_df.to_csv("data/CSVs/val.csv", index=False)
test_df.to_csv("data/CSVs/test.csv", index=False)
print("train.csv, val.csv, and test.csv saved.")

# =============================================================
# ADDED: Step 3 - Verification Bar Chart
# =============================================================
counts = {
    'dataset.csv': len(data_df),
    'train.csv': len(train_df),
    'val.csv': len(val_df),
    'test.csv': len(test_df)
}

plt.figure(figsize=(8, 5))
bars = plt.bar(counts.keys(), counts.values(), color=['#2b5c8f', '#4caf50', '#ff9800', '#f44336'])
plt.ylabel('Number of Samples')
plt.title('Sample Counts per CSV File')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, str(int(height)), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("verification_barchart.png")
plt.show()