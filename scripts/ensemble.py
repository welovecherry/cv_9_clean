# scripts/ensemble.py
import pandas as pd

# 1. 두 에이스 모델의 답안지(submission 파일)를 읽어옴
df_a = pd.read_csv('submission_A.csv')
df_b = pd.read_csv('submission_B.csv')

# 2. 두 예측이 동일하면, 그 값을 그대로 사용.
#    만약 두 예측이 다르면, 더 성능이 좋았던 A팀의 예측을 우선적으로 믿자.
final_preds = []
for _, row in pd.merge(df_a, df_b, on='ID').iterrows():
    # target_x는 df_a의 예측, target_y는 df_b의 예측
    if row['target_x'] == row['target_y']:
        final_preds.append(row['target_x'])
    else:
        final_preds.append(row['target_x']) # 의견이 다를 땐, 우리 에이스 A팀을 믿는다!

# 3. 최종 앙상블 제출 파일 생성
submission = pd.DataFrame({'ID': df_a['ID'], 'target': final_preds})
submission.to_csv('submission_ensemble_final.csv', index=False)

print("Ensemble submission file 'submission_ensemble_final.csv' created!")