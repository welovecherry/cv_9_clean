import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

# 경로 설정
csv_path = "./data/final_training_data/train.csv"
output_path = "./data/final_training_data/train_folds.csv"

# 데이터 불러오기
df = pd.read_csv(csv_path)

# fold 열 생성
df["fold"] = -1
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# target 기준으로 Stratified Split
for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["target"])):
    df.loc[val_idx, "fold"] = fold

# 저장
df.to_csv(output_path, index=False)

print(f"[완료] fold 열이 추가된 CSV 저장 완료 → {output_path}")