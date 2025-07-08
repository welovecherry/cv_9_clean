# scripts/analyze_results.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(csv_path, meta_csv_path):
    # 1. 채점표와 메타 데이터 로드
    df = pd.read_csv(csv_path)
    meta_df = pd.read_csv(meta_csv_path)
    class_names = meta_df.sort_values(by='target')['class_name'].values

    # 2. Confusion Matrix 계산
    cm = confusion_matrix(df['true_label'], df['predicted_label'])
    
    # 3. 히트맵 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() # 레이아웃 최적화
    
    # 4. 이미지 파일로 저장
    output_path = './confusion_matrix.png'
    plt.savefig(output_path)
    
    print(f"Confusion Matrix saved to `{output_path}`")

if __name__ == '__main__':
    RESULTS_CSV_PATH = './validation_results.csv'
    META_CSV_PATH = './data/raw/meta.csv'
    plot_confusion_matrix(RESULTS_CSV_PATH, META_CSV_PATH)