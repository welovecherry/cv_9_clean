
# # scripts/02_debug_ocr_then_line.py
# ''' 49 / 80 회전 교정 성공 '''
import cv2
import numpy as np
import os
from tqdm import tqdm
import easyocr
import math

def correct_skew_with_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 1 or abs(x2 - x1) < 1:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    if len(angles) < 5:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) > 45:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correct_orientation_ocr(image, reader):
    candidates = []
    for angle in [0, 90, 180, 270]:
        rotated = np.rot90(image, k=angle // 90)
        candidates.append(rotated)
    flipped = cv2.flip(image, 1)
    for angle in [0, 90, 180, 270]:
        rotated = np.rot90(flipped, k=angle // 90)
        candidates.append(rotated)

    best_candidate = image
    max_score = -1

    for i, candidate in enumerate(candidates):
        try:
            results = reader.readtext(candidate, detail=1)
            if not results:
                continue

            text_length = sum([len(res[1]) for res in results])
            avg_confidence = np.mean([res[2] for res in results])
            score = text_length * avg_confidence
            if i >= 4:
                score *= 0.95

            if score > max_score:
                max_score = score
                best_candidate = candidate
        except:
            continue

    return best_candidate

if __name__ == '__main__':
    print("Initializing EasyOCR Reader...")
    ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)  # GPU 사용 설정
    print("EasyOCR Reader initialized.")

    src_dir = './data/debug_samples/test_rotated/'
    dst_dir = './data/processed/02_debug_ocr_then_line_49/80/'
    os.makedirs(dst_dir, exist_ok=True)

    image_files = [f for f in os.listdir(src_dir) if not f.startswith('.')]

    for file_name in tqdm(image_files, desc="Correcting orientation with OCR + Lines"):
        try:
            src_path = os.path.join(src_dir, file_name)
            image = cv2.imread(src_path)
            if image is None:
                continue

            ocr_corrected = correct_orientation_ocr(image, reader=ocr_reader)
            final_corrected = correct_skew_with_lines(ocr_corrected)

            dst_path = os.path.join(dst_dir, file_name)
            cv2.imwrite(dst_path, final_corrected)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("\nAll combined correction tasks finished!")
