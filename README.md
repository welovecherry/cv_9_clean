🚀 Systematic ML Experimentation for Image Classification
💡 Project Focus: This repository is a deep dive into the systematic process of developing a high-performing image classification model. It highlights my experience in building a reproducible experimentation pipeline using WandB, PyTorch Lightning, and Hydra, and showcases a data-driven approach to model optimization.

English | 한국어

English
🎯 Project Overview

This project details the journey of tackling a competitive image classification challenge with 17 classes. My core strategy was not just to train models, but to build an efficient, automated experimentation workflow. I ran over 168 experiments, systematically analyzing failures and testing hypotheses to continuously improve performance. This culminated in creating a sophisticated ensemble model that achieved my personal best score on the final day.





[IMAGE: Graph showing F1 score improving through 168 experiments]

Tracking over 168 experiments in W&B to find the winning strategy.


💻 Tech Stack

Category	Technology	Purpose in Project
ML & Training	Python, PyTorch, PyTorch Lightning	Built and trained all models using PyTorch. Used Lightning to structure the code cleanly and simplify training loops.
Experiment Mgmt	Weights & Biases (W&B), Hydra	Tracked all 168 experiments with W&B. Managed all complex configurations for different models and training runs using Hydra.
Data & Analysis	pandas, OpenCV, Albumentations, Augraphy	Used pandas for data handling. Applied powerful image augmentations with Albumentations and realistic noise with Augraphy to improve model robustness.
Tools & Etc.	Git, scikit-learn	Used Git for version control. Evaluated model performance with scikit-learn's classification reports and F1-score metric.
🏗️ My ML Experiment Workflow

I established a daily workflow to maximize efficiency: running automated hyperparameter sweeps overnight and analyzing the results to build and test new inference strategies during the day.

┌──────────────────┐   ┌────────────────┐   ┌────────────────────┐   ┌─────────────────┐
│  Data Analysis   │──▶│ Augmentation & │──▶│   Training         │──▶│    Inference    │
│ (Error Analysis) │   │ Preprocessing  │   │ (Hydra + W&B Sweep)│   │ (Ensemble/TTA)  │
└──────────────────┘   └────────────────┘   └────────────────────┘   └─────────────────┘
🚀 Key Strategies & Achievements

1. Building a Reproducible Experimentation Framework

My Tools: I used Hydra for clean configuration management, allowing me to run hundreds of experiments by only changing .yaml files.

My Workflow: Every experiment—from model architecture changes to learning rate adjustments—was automatically tracked using Weights & Biases (W&B).

Impact: This systematic approach created a fully reproducible pipeline, making it easy to analyze results and build upon successful ideas.

2. Systematic Error-Driven Analysis

My Method: I didn't just look at metrics. I created an "Error Notebook" (오답노트) by manually reviewing images the baseline model failed on.

Impact: This deep analysis revealed critical model weaknesses, such as confusion between specific class pairs. This data-driven insight guided all subsequent modeling decisions.

3. Hypothesis-Driven Modeling (Successes & Failures)

My goal was to test clear hypotheses, learning from both successes and failures. This allowed for steady, logical progress instead of random guessing.

My Hypothesis	Result	What I Learned
Soft Voting > Hard Voting	✅ Success	Averaging model probabilities (confidence) is far more effective than simple majority voting. This was a key turning point.
TTA Will Boost Stability	✅ Success	Applying 8-way Test-Time Augmentation consistently improved the score by making predictions more robust.
More Models = Better Ensemble	🤔 Mixed	A 3-model ensemble did not automatically beat a fine-tuned 2-model ensemble. This showed that the quality and diversity of the models in an ensemble are more important than sheer quantity.
4. The Final Push: A Data-Driven Finale

The Challenge: On the final day, I had two top-performing but slightly different ensemble models (1814 and 1824).

The Strategy: I used my "Error Notebook"—the small set of images I knew were difficult—as a final, high-stakes test set.

The Decision: I ran both candidate models on just these difficult images and compared their performance side-by-side against my hand-labeled answers.

The Result: The 1814 model (Two-Ace Ensemble with TTA) performed better on these critical cases, and I submitted it with confidence, achieving my personal best score.





[IMAGE: Screenshot of the final comparison table between 1808, 1814, and 1824]

Making the final decision by comparing top models against a curated set of "hard cases".


🏃‍♂️ Quick Start

Bash
# 1. Install dependencies from the requirements file
pip install -r requirements.txt

# 2. Run the final inference script to generate submission
python scripts/inference_two_ace_ensemble.py
🎯 Future Enhancements

Model Optimization: Convert models to TensorRT for faster inference.

Advanced Ensembling: Experiment with stacking and blending techniques.

Post-Processing: Develop more sophisticated rule-based post-processing.

🚀 체계적인 실험으로 완성한 이미지 분류 프로젝트
💡 프로젝트 핵심: 이 저장소는 머신러닝 경진대회를 진행한 전체 과정을 기록합니다. 특히 WandB, PyTorch Lightning, Hydra를 활용하여 재현 가능한 실험 파이프라인을 구축하고, 데이터 기반 분석을 통해 모델 성능을 체계적으로 개선한 경험을 강조합니다.

🎯 프로젝트 개요

17개 클래스를 분류하는 이미지 분류 경진대회에서 높은 점수를 달성하는 것을 목표로 했습니다. 단순히 모델을 학습시키는 것을 넘어, 효율적이고 자동화된 실험 워크플로우를 구축하는 데 집중했습니다. 168회 이상의 실험을 진행하며 실패를 체계적으로 분석하고 가설을 검증하며 점진적으로 성능을 개선했고, 이 과정을 통해 대회 마지막 날 개인 최고 기록을 달성하는 정교한 앙상블 모델을 완성했습니다.





[IMAGE: 168회 실험을 통해 F1 스코어가 향상되는 과정을 보여주는 W&B 그래프]

W&B 대시보드: 168번의 실험을 통해 최적의 전략을 찾아가는 과정


💻 기술 스택

분야	기술	프로젝트 내 역할
ML & 학습	Python, PyTorch, PyTorch Lightning	PyTorch를 기반으로 모든 모델을 구축 및 학습했으며, PyTorch Lightning으로 코드를 효율적으로 구조화하고 재현성을 높였습니다.
데이터 & 분석	pandas, OpenCV, Albumentations, Augraphy	pandas로 데이터를 다루고, Albumentations와 Augraphy로 강력한 이미지 증강 기법을 적용하여 모델의 강건성을 확보했습니다.
실험 관리	Weights & Biases (W&B), Hydra	W&B로 168번의 모든 실험을 추적했으며, Hydra를 이용해 복잡한 실험 설정을 체계적으로 관리했습니다.
도구 & 기타	Git, scikit-learn	Git으로 체계적인 버전 관리를 수행했으며, scikit-learn으로 F1 스코어 등 모델 성능을 정밀하게 평가했습니다.
🏗️ ML 실험 워크플로우

효율적인 실험을 위해 저만의 워크플로우를 구축했습니다. 밤에는 자동화된 하이퍼파라미터 탐색(Sweep)을 실행하고, 낮에는 그 결과를 분석하여 새로운 추론 전략을 수립하고 검증했습니다.

┌──────────────┐   ┌────────────────┐   ┌───────────────────┐   ┌─────────────────┐
│ 데이터 분석  │──▶│   데이터 증강   │──▶│   모델 학습       │──▶│      추론      │
│ (EDA, 오답노트)│   │ (Albumentations) │   │ (Hydra + W&B)    │   │ (앙상블, TTA)   │
└──────────────┘   └────────────────┘   └───────────────────┘   └─────────────────┘
🚀 핵심 전략 및 성과

1. 재현 가능한 실험 프레임워크 구축

나의 도구: Hydra를 사용해 복잡한 실험 설정을 체계적으로 관리함으로써, 소스 코드 변경 없이 .yaml 파일 수정만으로 신속하고 다양한 실험을 진행했습니다.

나의 워크플로우: 모든 실험 결과는 W&B에 자동으로 기록되도록 파이프라인을 구축했습니다.

임팩트: 체계적이고 재현 가능한 이 파이프라인 덕분에 성공과 실패 요인을 명확히 추적하고, 성공적인 아이디어를 빠르게 발전시킬 수 있었습니다.

2. 체계적인 오답 분석 ("오답 노트")

나의 방법: 전체 점수만 보는 대신, 베이스라인 모델이 틀린 이미지들을 수동으로 분석하는 '오답 노트'를 만들었습니다.

임팩트: 이 데이터 기반 접근법을 통해 모델의 정확한 약점(예: 3번과 4번 클래스의 혼동)을 파악했고, 이는 모든 후속 모델링 결정의 길잡이가 되었습니다.

3. 가설 기반 모델링 (성공과 실패의 기록)

막연한 시도가 아닌, 명확한 가설을 세우고 검증하는 과정을 반복했습니다. 실패한 실험조차도 다음 단계를 위한 중요한 데이터가 되었습니다.

나의 가설	결과	배운 점
Soft Voting > Hard Voting	✅ 성공	모델의 '확신'을 반영하는 확률 평균 방식이 단순 다수결 방식보다 훨씬 효과적이었습니다. 이는 프로젝트의 결정적인 전환점이 되었습니다.
TTA는 안정성을 높일 것이다	✅ 성공	8-way Test-Time Augmentation을 적용했을 때, 예측 안정성이 눈에 띄게 향상되며 점수를 추가로 확보할 수 있었습니다.
모델이 많을수록 좋을 것이다	🤔 절반의 성공	3개 모델 앙상블이 2개 모델 앙상블보다 항상 좋지는 않았습니다. 앙상블은 모델의 수보다 **'조합'과 '다양성'**이 더 중요하다는 것을 배웠습니다.
4. 마지막 한 수: 데이터 기반의 최종 결정

도전: 대회 마지막 날, 성능이 비슷한 최상위 앙상블 모델(1814와 1824) 중 최종 제출 모델을 선택해야 했습니다.

전략: 제가 직접 만든 '오답 노트'를 최종 테스트셋으로 활용했습니다.

결정: 두 후보 모델을 이 '가장 어려운 문제'들로 테스트한 후, 정답과 비교하여 1814 모델이 더 안정적인 것을 확인하고 확신을 갖고 제출했습니다.

결과: 이 데이터 기반의 최종 결정 덕분에 개인 최고 기록을 경신할 수 있었습니다.





[IMAGE: 1808, 1814, 1824 결과를 비교했던 최종 분석 테이블 스크린샷]

가장 어려운 문제들로 최종 모델을 직접 검증하여 마지막 결정을 내리는 과정


🏃‍♂️ 빠른 시작

Bash
# 1. 저장소 복제
git clone [Your-Repo-Link]
cd [Repo-Name]

# 2. 필요 라이브러리 설치
pip install -r requirements.txt

# 3. 최종 추론 스크립트 실행
# 이 스크립트는 최종 제출 파일을 생성합니다.
python scripts/inference_two_ace_ensemble.py
🎯 향후 개선 방향

모델: 최신 SOTA 모델(예: ConvNeXt V2)을 앙상블에 추가하여 성능 추가 개선.

후처리: OCR 등 규칙 기반 후처리 기법을 고도화하여 특정 오답 패턴 보정.

최적화: TensorRT와 같은 도구를 활용하여 추론 속도 최적화.
