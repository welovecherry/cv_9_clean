# scripts/01_denoise.py

import cv2
import os
from tqdm import tqdm

def denoise_image(image):
    """
    Non-Local Means Denoising을 사용해 이미지의 노이즈를 제거합니다.
    :param image: 원본 이미지 (OpenCV BGR 포맷)
    :return: 노이즈가 제거된 이미지
    """
    # cv2.fastNlMeansDenoisingColored(입력이미지, 결과이미지, 필터강도, 색상필터강도, 템플릿크기, 검색범위)
    # h, hColor: 값이 클수록 노이즈를 더 많이 제거하지만, 이미지가 약간 뭉개질 수 있음. 10이 표준적인 값.
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image


if __name__ == '__main__':
    # 1단계(노이즈 제거) 파이프라인 설정
    tasks = [
        ('train', './data/raw/train/', './data/processed/01_denoised/train/'),
        ('test', './data/raw/test/', './data/processed/01_denoised/test/')
    ]

    for task_name, src_dir, dst_dir in tasks:
        print(f"--- Step 1: Denoising for: {task_name} data ---")
        os.makedirs(dst_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(src_dir) if not f.startswith('.')]
        
        # [테스트용] 작은 샘플로만 테스트하려면 아래 주석을 해제하세요.
        image_files = image_files[:10]

        for file_name in tqdm(image_files, desc=f"Denoising {task_name}"):
            try:
                src_path = os.path.join(src_dir, file_name)
                raw_image = cv2.imread(src_path)

                if raw_image is None: continue

                # 노이즈 제거 함수 실행
                denoised_image = denoise_image(raw_image)

                # 1단계 결과물 저장
                dst_path = os.path.join(dst_dir, file_name)
                cv2.imwrite(dst_path, denoised_image)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print("\nStep 1 (Denoising) finished!")
    print("Check results in './data/processed/01_denoised/'")