# scripts/create_final_dataset.py

import pandas as pd
import os
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    # --- ⚙️ 설정 ---
    # 1. 원본 데이터 경로
    RAW_DATA_PATH = './data/raw/'
    
    # 2. '신뢰도 점수'가 포함된 예측 파일 경로
    PSEUDO_CANDIDATES_PATH = './pseudo_label_candidates.csv'
    
    # 3. '이 점수 이상으로 확신하는 예측'만 새로운 학습 데이터로 사용
    CONFIDENCE_THRESHOLD = 0.99 
    
    # 4. 최종 데이터셋을 저장할 새로운 폴더 경로
    FINAL_DATA_PATH = './data/final_training_data/'
    
    # --- 실행 ---
    print("Creating the final, ultimate training dataset...")
    
    # --- 1. 기존의 원본 학습 데이터 먼저 복사하기 ---
    print("Step 1: Copying original training data...")
    original_train_dir = os.path.join(RAW_DATA_PATH, 'train')
    # shutil.copytree를 사용해 train 폴더를 통째로 복사
    if os.path.exists(FINAL_DATA_PATH):
        shutil.rmtree(FINAL_DATA_PATH) # 기존 폴더가 있으면 삭제 후 새로 시작
    shutil.copytree(original_train_dir, FINAL_DATA_PATH)
    
    # --- 2. '의사 라벨(Pseudo-Label)' 추가하기 ---
    print("Step 2: Adding high-confidence pseudo-labels...")
    pseudo_df = pd.read_csv(PSEUDO_CANDIDATES_PATH)
    
    # 신뢰도 점수가 기준치(0.99) 이상인 데이터만 필터링
    high_confidence_df = pseudo_df[pseudo_df['confidence'] >= CONFIDENCE_THRESHOLD]
    
    # 원본 클래스 이름 정보를 가져오기 위해 meta.csv 로드
    meta_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'meta.csv'))
    
    # 각 의사 라벨에 대해, 해당하는 테스트 이미지를 새로운 학습 폴더로 복사
    for _, row in tqdm(high_confidence_df.iterrows(), total=len(high_confidence_df), desc="Adding pseudo-labels"):
        image_id = row['ID']
        pseudo_label = row['target']
        
        # 라벨 번호에 맞는 클래스 이름 찾기
        class_name = meta_df[meta_df['target'] == pseudo_label]['class_name'].iloc[0]
        
        # 해당 클래스 폴더가 없으면 생성
        class_dir = os.path.join(FINAL_DATA_PATH, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 원본 테스트 이미지 경로
        src_path = os.path.join(RAW_DATA_PATH, 'test', image_id)
        # 새로운 학습 데이터 폴더로 복사
        dst_path = os.path.join(class_dir, f"pseudo_{image_id}") # 겹치지 않게 이름 앞에 'pseudo_' 추가
        
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)

    print(f"\nFinal dataset created at '{FINAL_DATA_PATH}'")
    print(f"Added {len(high_confidence_df)} high-confidence pseudo-labeled images.")