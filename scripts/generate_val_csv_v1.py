# scripts/generate_val_csv_v1.py

import pandas as pd

input_path = './data/final_training_data/train_folds.csv'
output_path = './data/final_training_data/val.csv'

df = pd.read_csv(input_path)
val_df = df[df['fold'] == 0][['ID', 'target']]  # fold 0만 선택
val_df.to_csv(output_path, index=False)

print(f"Saved val.csv with {len(val_df)} samples")