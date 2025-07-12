# # scripts/final_ensemble.py
# import pandas as pd
# from datetime import datetime

# if __name__ == '__main__':
#     # --- 설정 ---
#     # 1. 우리의 최고점 제출 파일 경로
#     OUR_BEST_SUBMISSION = './submission_avengers_0706_1606.csv' # 네가 만든 파일 이름으로 수정

#     # 2. 팀원의 최고점 제출 파일 경로
#     TEAMMATE_BEST_SUBMISSION = './submission_teammate_0.83.csv' # 팀원의 0.83점짜리 파일

#     # --- 실행 ---
#     print("Loading submission files...")
#     our_df = pd.read_csv(OUR_BEST_SUBMISSION)
#     teammate_df = pd.read_csv(TEAMMATE_BEST_SUBMISSION)

#     # 두 데이터프레임을 'ID' 기준으로 합치기
#     merged_df = pd.merge(our_df, teammate_df, on='ID', suffixes=('_our', '_teammate'))

#     final_predictions = []
#     correction_count = 0
#     for _, row in merged_df.iterrows():
#         # 우리 예측과 팀원 예측이 다를 경우
#         if row['target_our'] != row['target_teammate']:
#             # 팀원의 예측(더 높은 점수)을 따름
#             final_predictions.append(row['target_teammate'])
#             correction_count += 1
#         else:
#             # 예측이 같으면 그대로 사용
#             final_predictions.append(row['target_our'])

#     print(f"Total corrections made by teammate's submission: {correction_count}")

#     # --- 최종 제출 파일 생성 ---
#     final_submission = pd.DataFrame({'ID': merged_df['ID'], 'target': final_predictions})
    
#     timestamp = datetime.now().strftime("%m%d_%H%M")
#     output_filename = f'./submission_final_blend_{timestamp}.csv'
#     final_submission.to_csv(output_filename, index=False)
    
#     print(f"\nFinal blended submission created: `{output_filename}`")