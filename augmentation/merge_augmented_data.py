import os
import pandas as pd
import shutil
from datetime import datetime

# 🔧 설정
# version_dirs = {
#     "v3": "../data/augmented_v3_0708",
#     "v5": "../data/augmented_v5_0708"
# }
version_dirs = {
    "v3": "./data/augmented_v3_0708",  # ✅ 수정
    "v5": "./data/augmented_v5_0708"   # ✅ 수정
}
merged_dir = "./data/final_training_data"
merged_image_dir = os.path.join(merged_dir, "train")
merged_csv_path = os.path.join(merged_dir, "train.csv")

# 🔄 디렉토리 초기화
if os.path.exists(merged_dir):
    shutil.rmtree(merged_dir)
os.makedirs(merged_image_dir)

# 🧩 CSV 병합
merged_df = []

for version, base_path in version_dirs.items():
    csv_path = os.path.join(base_path, "train.csv")
    image_dir = os.path.join(base_path, "train")
    
    df = pd.read_csv(csv_path)
    df['ID'] = df['ID'].apply(lambda x: x.strip())  # 공백 제거
    
    # 이미지 복사
    for image_id in df['ID']:
        src = os.path.join(image_dir, image_id)
        dst = os.path.join(merged_image_dir, image_id)
        shutil.copy2(src, dst)
    
    merged_df.append(df)

# 🧾 최종 CSV 저장
final_df = pd.concat(merged_df, ignore_index=True)
final_df.to_csv(merged_csv_path, index=False)

print(f"[완료] 병합된 이미지 수: {len(final_df)}장 → {merged_csv_path}")