# scripts/postprocess_ocr_strike_v1.py
import pandas as pd
import os
from tqdm import tqdm
import cv2
import easyocr
from datetime import datetime

def ocr_correction(image_path, original_prediction, reader):
    """
    OCR로 텍스트를 읽고, 결정적 키워드가 있으면 예측을 수정하는 함수
    """
    try:
        # OCR 실행
        result = reader.readtext(image_path, detail=0, paragraph=True)
        full_text = " ".join(result).lower() # 모든 텍스트를 소문자로 변환하여 비교

        # '이력서(13)'의 결정적 키워드들
        resume_keywords = ["이력서", "경력", "학력", "education", "career", "resume"]
        # '소견서(14)'의 결정적 키워드들
        opinion_keywords = ["소견", "진단", "opinion", "diagnosis"]

        # 키워드 확인 및 예측 수정
        # 1. 이력서 키워드 확인
        if any(keyword in full_text for keyword in resume_keywords):
            # '소견서'로 잘못 예측된 것을 '이력서'로 수정
            if original_prediction == 14:
                print(f"Correction! Image: {os.path.basename(image_path)}, Original: 14 -> Corrected to: 13")
                return 13
        
        # 2. 소견서 키워드 확인
        if any(keyword in full_text for keyword in opinion_keywords):
            # '이력서'로 잘못 예측된 것을 '소견서'로 수정
            if original_prediction == 13:
                print(f"Correction! Image: {os.path.basename(image_path)}, Original: 13 -> Corrected to: 14")
                return 14

    except Exception as e:
        print(f"OCR Error for {os.path.basename(image_path)}: {e}")
    
    # 키워드를 찾지 못하거나, 수정 대상이 아니면 원본 예측을 그대로 반환
    return original_prediction


if __name__ == '__main__':
    # --- ⚙️ 설정 ---
    # 1. 우리가 보강할, 가장 성능이 좋았던 제출 파일 경로
    BASE_SUBMISSION_PATH = 'submissions/submission_avengers_0705_2334.csv' # 네 최고점 파일 이름으로 수정

    # 2. 원본 테스트 이미지 폴더 경로
    TEST_DIR = './data/raw/test/'

    # 3. 우리가 '정밀 타격'할 클래스
    CONFUSING_CLASSES = {13, 14} 
    
    # --- OCR 리더 초기화 ---
    print("Initializing EasyOCR Reader...")
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    print("EasyOCR Reader initialized.")

    # --- 후처리 실행 ---
    base_df = pd.read_csv(BASE_SUBMISSION_PATH)
    corrected_predictions = []

    for _, row in tqdm(base_df.iterrows(), total=len(base_df), desc="Post-processing with OCR Strike"):
        image_id = row['ID']
        prediction = row['target']

        # 모델이 '이력서' 또는 '소견서'로 예측했을 경우에만 OCR 실행
        if prediction in CONFUSING_CLASSES:
            image_path = os.path.join(TEST_DIR, image_id)
            corrected_pred = ocr_correction(image_path, prediction, reader)
            corrected_predictions.append(corrected_pred)
        else:
            # 그 외의 경우에는 원래 예측을 그대로 믿음
            corrected_predictions.append(prediction)
    
    # --- 최종 제출 파일 생성 ---
    base_df['target'] = corrected_predictions
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_filename = f'./submission_ocr_strike_{timestamp}_v1.csv'
    base_df.to_csv(output_filename, index=False)

    print(f"\nOCR Strike finished! `{output_filename}` has been created.")