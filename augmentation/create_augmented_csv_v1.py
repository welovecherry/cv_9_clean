# # scripts/create_augmented_csv_v1.py
# import os
# import pandas as pd

# # 증강 이미지 폴더
# augmented_dir = 'data/augmented_only/class14_augmented'

# # 파일명 리스트
# files = sorted(os.listdir(augmented_dir))

# # 각 이미지에 대해 ID, target=14, fold=0 (예시)로 설정
# rows = []
# for fname in files:
#     if fname.lower().endswith('.jpg'):
#         rows.append({
#             'ID': fname,
#             'target': 14,
#             'fold': 0  # fold는 0으로 지정하거나 랜덤하게 지정할 수도 있어요
#         })

# # DataFrame 생성 및 저장
# df = pd.DataFrame(rows)
# df.to_csv('data/augmented_only/class14_augmented.csv', index=False)
# print(f'Saved {len(df)} rows to data/augmented_only/class14_augmented.csv')

import os
import pandas as pd

# 기존 train.csv 경로
train_csv_path = "./data/final_training_data/train.csv"
# 증강된 이미지 폴더 경로
augmented_dir = "./data/augmented_only/class14_augmented"
# 새로 저장할 CSV 경로
output_csv_path = "./data/final_training_data/train_aug14.csv"

# 1. 기존 train.csv 불러오기
train_df = pd.read_csv(train_csv_path)

# 2. 증강 이미지 파일 목록 불러오기
augmented_files = [f for f in os.listdir(augmented_dir) if f.endswith(('.png', '.jpg'))]

# 3. 클래스 14로 라벨링
augmented_df = pd.DataFrame({
    "ID": augmented_files,
    "target": [14] * len(augmented_files)
})

# 4. 기존 데이터와 합치기
combined_df = pd.concat([train_df, augmented_df], ignore_index=True)

# 5. 저장
combined_df.to_csv(output_csv_path, index=False)
print(f"Saved {len(combined_df)} entries to {output_csv_path}")