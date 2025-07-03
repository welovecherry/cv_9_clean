import os
import shutil
import pandas as pd

# 경로 설정
base_dir = './data/raw'
train_dir = os.path.join(base_dir, 'train')
csv_path = os.path.join(base_dir, 'train.csv')
meta_path = os.path.join(base_dir, 'meta.csv')
output_dir = './data/train_split'

# 클래스 이름 매핑
meta = pd.read_csv(meta_path)
target2class = dict(zip(meta['target'], meta['class_name']))

# train.csv 읽기
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    img_id, target = row['ID'], row['target']
    class_name = target2class[target]
    class_dir = os.path.join(output_dir, f"{target:02d}_{class_name}")
    os.makedirs(class_dir, exist_ok=True)

    src_path = os.path.join(train_dir, img_id)
    dst_path = os.path.join(class_dir, img_id)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)