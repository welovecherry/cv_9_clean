# scripts/01_preprocess_test_set.py
import cv2
import os
from tqdm import tqdm
from skimage import io
from deskew import determine_skew

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None: return

    # 1. 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 2. 미세 각도 교정 (deskew)
    grayscale = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    rotated = io.rotate(denoised, angle, resize=True, cval=255) * 255
    rotated = rotated.astype(np.uint8)

    cv2.imwrite(output_path, rotated)

if __name__ == '__main__':
    src_dir = './data/raw/test/'
    dst_dir = './data/processed/test_cleaned/'
    os.makedirs(dst_dir, exist_ok=True)

    image_files = [f for f in os.listdir(src_dir) if not f.startswith('.')]
    for file_name in tqdm(image_files, desc="Cleaning Test Set"):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        process_image(src_path, dst_path)

print(f"Finished cleaning test set. Results are in {dst_dir}")