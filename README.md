# 🏆 Systematic ML Experimentation for Image Classification - (F1_score: 0.9462)

💡 **Project Focus**  
This repository is a deep dive into the systematic process of developing a high-performing image classification model.  
It highlights my experience in building a reproducible experimentation pipeline using WandB, PyTorch Lightning, and Hydra, and showcases a data-driven approach to model optimization.  

---

[English](#-systematic-ml-experimentation-for-image-classification---f1_score-09462) | [한국어](#-체계적인-실험을-통한-이미지-분류-경진대회-f1_score-09462)

---

## 🎯 Project Overview  

This project details my end-to-end process for a Document Image Classification Challenge with 17 classes.  
The primary goal was to systematically improve model performance by moving beyond baseline models.  
I designed and executed a highly organized experimentation workflow, conducting over **167 tracked experiments** to find the optimal data handling, modeling, and inference strategies.  
This rigorous, data-driven approach culminated in a sophisticated ensemble model that achieved my personal best score on the final leaderboard.  

> Tracking over 167 experiments in W&B to find the winning strategy.

<img src="https://github.com/user-attachments/assets/a3472d87-fd36-4e75-b4f2-47d00bf2759d" width="1286" height="526">
<img src="https://github.com/user-attachments/assets/04fe3959-c753-4754-aa64-837f5fc97846" width="1062" height="428">

---

## 💻 Tech Stack  

| Category | Technology | Purpose in Project |
|---|---|---|
| ML & Training | Python, PyTorch, PyTorch Lightning | Used PyTorch Lightning to structure training code cleanly and reduce boilerplate. All models were built on the PyTorch framework. |
| Experiment Mgmt | Weights & Biases (W&B), Hydra | Tracked all 167+ experiments with W&B, visualizing metrics to compare models. Managed all complex configurations with Hydra, enabling rapid iteration. |
| Data & Analysis | pandas, OpenCV, Albumentations, Augraphy | Applied powerful augmentations with Albumentations. Simulated real-world document noise (stains, folds) with Augraphy to enhance model robustness. |
| Tools & Etc. | Git, scikit-learn | Used Git for version control. Evaluated models using the F1-score metric from scikit-learn, which is crucial for imbalanced datasets. |

---

## 🏗️ ML Experiment Workflow  

I established a systematic workflow to ensure all experiments were logical and reproducible.  

```
┌──────────────────┐   ┌────────────────┐   ┌────────────────────┐   ┌─────────────────┐
│ Data Analysis    │──▶│ Augmentation & │──▶│   Model Training   │──▶│    Inference    │
│ (Error Analysis) │   │ Preprocessing  │   │ (Hydra + W&B Sweep)│   │ (Ensemble/TTA)  │
└──────────────────┘   └────────────────┘   └────────────────────┘   └─────────────────┘
```

<img src="https://github.com/user-attachments/assets/8ca5208a-3e9c-4ecc-965c-97255bfe4d43" width="852" height="536">
<img src="https://github.com/user-attachments/assets/3af24b9d-2742-4dfa-a539-602ffa83586a" width="810" height="823">

---

## 📝 Lessons Learned & Reflections

> *Key Takeaways from My Systematic ML Experimentation*

| No. | Topic                                                   | Key Takeaways                                                                                                           | Additional Insights                                                           |
| --- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 1️⃣ | Error Analysis as the Starting Point                    | Manually reviewing model errors provided deep insights beyond metrics.                                                  | Creating an “Error Notebook” helped identify systematic model weaknesses.     |
| 2️⃣ | The Power of Hypothesis-Driven Experimentation          | Each experiment was treated as a hypothesis test, leading to systematic improvements.                                   | Even failed experiments contributed valuable lessons for strategy refinement. |
| 3️⃣ | The Importance of Experiment Tracking & Reproducibility | Rigorous experiment tracking with **Weights & Biases (W\&B)** and config management with **Hydra** prevented confusion. | Enabled easy reproduction of any result at any time.                          |
| 4️⃣ | Quality Over Quantity in Ensembles                      | Adding more models did not automatically improve performance.                                                           | The diversity and complementarity of models mattered far more.                |
| 5️⃣ | Confidence in Data-Driven Decision Making               | Trusted data over intuition, especially for final model selection.                                                      | This approach increased confidence in my final submission.                    |

> These insights will guide my future projects, ensuring every experiment is purposeful, reproducible, and hypothesis-driven.

---

## 🧩 Data Augmentation Strategies & Breakthrough (English Version)

> *Building Model Robustness with Multi-Stage Data Augmentation*

| No. | Strategy                                  | Actions                                                                                                                                                                                                                                         | Analysis & Results                                                                                                                                                         |
| --- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1️⃣ | Initial Approach: Online Augmentation     | **Action:** Applied real-time augmentations (rotation, brightness) using **Albumentations** during training.                                                                                                                                    | **Challenge:** Effective for initial improvement but performance plateaued. Hard to control quality of on-the-fly augmentations.                                           |
| 2️⃣ | Advanced Strategy: Offline Augmentation   | **Action:** Created six distinct, high-quality augmented datasets offline using **Augraphy** (stains, shadows, etc.).                                                                                                                           | **Analysis:** Performance varied across datasets. Visual inspection revealed **'Shadow Effect (v3)'** and **'Stain Effect (v5)'** resembled test data most closely.        |
| 3️⃣ | Breakthrough: Optimal Dataset Combination | **Hypothesis:** Merging datasets with different strengths would enhance robustness against diverse noise.<br>**Action:** Merged the two best-performing datasets (v3, v5) using **pandas.concat** to build a larger, more diverse training set. | **Result:** The combined dataset broke through previous performance plateaus, boosting leaderboard scores significantly and became the baseline for all subsequent models. |

---

## 🚀 Key Strategies and Achievements  

### 1️⃣ Data-Driven Problem Definition via Error Analysis  

I didn't just train models; I started by analyzing why the baseline model failed.  
I created a detailed **"Error Notebook"** by manually reviewing misclassified images.  

> **Key Insight:** The model consistently confused visually similar classes, such as `pharmaceutical_receipt` and `confirmation_of_admission_and_discharge`.  
> This shifted my focus from finding a single "best" model to building an ensemble that could resolve these ambiguities.  

---

### 2️⃣ Hypothesis-Driven Modeling (Successes & Failures)  

| My Hypothesis | Result | What I Learned |
|---|---|---|
| Soft Voting will outperform Hard Voting. | ✅ Major Success | Averaging model probabilities (confidence) was far more effective than simple voting. The F1 score jumped significantly. |
| TTA will enhance stability. | ✅ Success | 8-way Test-Time Augmentation boosted robustness and performance. |
| More models are always better for an ensemble. | 🤔 Valuable Failure | A 3-model ensemble did not beat a fine-tuned 2-model ensemble. Quality and diversity > quantity. |

---

### 3️⃣ The Final Push: A Data-Driven Finale  

**The Challenge:** On the final day, I had two top-performing ensemble models.  

**The Strategy:** I used my **Error Notebook** as a final, high-stakes validation set.  

**The Decision:** I compared both models on these difficult cases, and the **Two-Ace Ensemble with TTA (1814)** performed better.  

**The Result:** This approach gave me confidence to submit the 1814 model, achieving my personal best.  

<img src="https://github.com/user-attachments/assets/f948cd1f-be7c-4265-a025-2436f9d12a0a" width="648" height="567">

---

## 📊 Measurable Results  

- **Final Leaderboard Score:** F1-score of **0.9462**  
- **Systematic Experimentation:** Tracked **167+ experiments** in W&B  

<img src="https://github.com/user-attachments/assets/c663adc0-facd-4f65-9b8d-177187accd12" width="757" height="820">


---
📝 Reflections

Key Takeaways from My Systematic ML Experimentation

1️⃣ Error Analysis as the Starting Point

Manually reviewing model errors provided deep insights beyond metrics.

Creating an “Error Notebook” helped identify systematic model weaknesses.

2️⃣ The Power of Hypothesis-Driven Experimentation

Each experiment was treated as a hypothesis test, leading to systematic improvements.

Even failed experiments contributed valuable lessons for strategy refinement.

3️⃣ The Importance of Experiment Tracking & Reproducibility

Rigorous experiment tracking with Weights & Biases (W&B) and config management with Hydra prevented confusion.

Enabled easy reproduction of any result at any time.

4️⃣ Quality Over Quantity in Ensembles

Adding more models did not automatically improve performance.

The diversity and complementarity of models mattered far more.

5️⃣ Confidence in Data-Driven Decision Making

Trusted data over intuition, especially for final model selection.

This approach increased confidence in my final submission.

These insights will guide my future projects, ensuring every experiment is purposeful, reproducible, and hypothesis-driven.


---

## 🏃‍♂️ Quick Start  

```bash
# 1. Clone the repository
git clone [Your-Repo-Link]
cd [Repo-Name]

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the final inference script
python scripts/inference/inference_two_ace_ensemble.py
```

---

# 🏆 체계적인 실험을 통한 이미지 분류 경진대회 (F1_score: 0.9462)

💡 **프로젝트 핵심**  
이 저장소는 머신러닝 경진대회를 진행한 전체 과정을 기록했습니다.  
WandB, PyTorch Lightning, Hydra를 활용하여 재현 가능한 실험 파이프라인을 구축하고, 데이터 기반 분석을 통해 모델 성능을 체계적으로 개선한 경험을 담았습니다.  

---

## 🎯 프로젝트 개요  

17개 클래스를 분류하는 문서 이미지 분류 경진대회의 전 과정을 담았습니다.  
체계적인 실험 워크플로우를 설계하고 **167회 이상의 실험**을 수행했습니다.  
오답 노트를 활용한 심층 분석으로 약점을 찾아내고, 앙상블 및 추론 전략을 적용해 개인 최고 기록을 달성했습니다.  

<img src="https://github.com/user-attachments/assets/a3472d87-fd36-4e75-b4f2-47d00bf2759d" width="1286" height="526">
<img src="https://github.com/user-attachments/assets/04fe3959-c753-4754-aa64-837f5fc97846" width="1062" height="428">

---

## 💻 기술 스택  

| 분야 | 기술 | 프로젝트 내 역할 |
|---|---|---|
| ML & 학습 | Python, PyTorch, PyTorch Lightning | PyTorch Lightning으로 코드를 구조화하고 PyTorch를 기반으로 모델을 구축 및 학습 |
| 실험 관리 | Weights & Biases (W&B), Hydra | 167회 이상의 실험을 W&B로 추적하고, Hydra로 복잡한 설정을 관리 |
| 데이터 & 분석 | pandas, OpenCV, Albumentations, Augraphy | 데이터 증강 및 노이즈 시뮬레이션으로 모델 강건성 향상 |
| 도구 & 기타 | Git, scikit-learn | Git으로 버전 관리, F1-Score로 성능 평가 |

---

## 🏗️ ML 실험 워크플로우  

```
┌──────────────┐   ┌────────────────┐   ┌────────────────────┐   ┌─────────────────┐
│ 데이터 분석  │──▶│  데이터 증강   │──▶│   모델 학습       │──▶│      추론      │
│ (EDA, 오답노트)│   │ (Albumentations) │   │ (Hydra + W&B)    │   │ (앙상블, TTA)   │
└──────────────┘   └────────────────┘   └───────────────────┘   └─────────────────┘
```

<img src="https://github.com/user-attachments/assets/8ca5208a-3e9c-4ecc-965c-97255bfe4d43" width="852" height="536">
<img src="https://github.com/user-attachments/assets/3af24b9d-2742-4dfa-a539-602ffa83586a" width="810" height="823">

---

## 🧩 데이터 증강 전략 및 성과 (한국어 버전)

> *모델의 강건성과 일반화 성능을 위한 다단계 데이터 증강 전략*

| 번호  | 전략                                    | 실행                                                                                                                                                    | 분석 및 결과                                                                                                           |
| --- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1️⃣ | 초기 접근: 온라인 증강 (Online Augmentation)   | **실행:** **Albumentations**를 사용하여 학습 중 회전, 밝기 조절 등의 증강을 실시간으로 적용했습니다.                                                                                  | **한계점:** 초기 성능 향상에는 효과적이었으나 일정 점수대에서 성능이 정체되었습니다. 실시간 증강의 품질을 통제하기 어려워 한계를 느꼈습니다.                                 |
| 2️⃣ | 심화 전략: 오프라인 증강 (Offline Augmentation) | **실행:** **Augraphy**를 활용해 얼룩, 그림자 등 6가지 테마의 고품질 증강 데이터를 사전에 생성했습니다.                                                                                   | **분석:** 데이터셋 별 성능 편차가 확인되었고, 그 중 \*\*'그림자 효과 (v3)'\*\*와 \*\*'얼룩 효과 (v5)'\*\*가 테스트 데이터와 가장 유사하다는 시각적 분석 결과를 얻었습니다. |
| 3️⃣ | 돌파구: 최적 데이터셋 조합                       | **가설:** 서로 다른 강점을 가진 데이터셋을 통합하면 다양한 노이즈에 더 강인한 모델이 될 것이다.<br>**실행:** 성능이 우수했던 **v3**와 **v5** 데이터를 **pandas.concat**으로 통합하여, 더 크고 다양한 학습 데이터셋을 구축했습니다. | **성과:** 통합 데이터셋으로 재학습한 결과, 기존 성능 정체를 돌파하며 리더보드 점수가 크게 향상되었습니다. 이후 모든 모델 학습의 핵심 베이스라인으로 사용되었습니다.                   |


---

## 🚀 핵심 전략 및 성과  

### 1️⃣ '오답 노트' 기반 문제 분석  

초기 모델의 오답을 수집하여 **오답 노트**를 만들고, 모델의 약점을 분석했습니다.  
특히 **유사 클래스 오분류 문제**를 발견하고 앙상블 전략을 설계했습니다.  

---

### 2️⃣ 가설 기반 모델링 (성공과 실패의 기록)  

| 나의 가설 | 결과 | 배운 점 |
|---|---|---|
| Soft Voting이 Hard Voting보다 우수할 것이다. | ✅ 대성공 | 확률 평균 방식이 단순 투표보다 효과적 |
| TTA가 예측 안정성을 높일 것이다. | ✅ 성공 | 8-way TTA로 성능 및 안정성 향상 |
| 모델은 많을수록 앙상블에 좋다. | 🤔 값진 실패 | 모델 수보다 다양성과 조합이 중요 |

---

### 3️⃣ 마지막 한 수: 데이터 기반의 최종 결정  

마지막 날 두 개의 앙상블 중 오답 노트를 기준으로 테스트하여 **1814 모델**을 최종 선택했습니다.  
데이터 기반의 결정이 개인 최고 기록으로 이어졌습니다.  

<img src="https://github.com/user-attachments/assets/f948cd1f-be7c-4265-a025-2436f9d12a0a" width="648" height="567">

---

## 📊 정량적 성과  

- **리더보드 점수:** **F1-Score 0.9462**  
- **167회 이상의 실험 관리 및 분석**  

<img src="https://github.com/user-attachments/assets/c663adc0-facd-4f65-9b8d-177187accd12" width="757" height="820">

---
📝 핵심 교훈 및 회고

체계적인 ML 실험을 통한 주요 배움과 성찰을 정리했습니다.

1️⃣ 오답 노트 기반의 에러 분석

단순 성능 지표가 아닌 직접 오답을 분석하여 약점을 발견했습니다.
오답 노트는 모델의 체계적인 약점 파악에 큰 도움이 되었습니다.

2️⃣ 가설 기반의 체계적인 실험

무작위 시도보다 가설을 세우고 검증하는 접근이 전략을 강화했습니다.
실패한 실험에서도 중요한 교훈을 얻을 수 있었습니다.

3️⃣ 실험 추적과 재현성의 중요성

W&B로 실험을 기록하고, Hydra로 설정을 관리하여 실험 혼동을 방지했습니다.
언제든지 결과를 재현할 수 있는 환경이 실험의 신뢰성을 높였습니다.

4️⃣ 앙상블은 '수'보다 '다양성'

모델 수를 늘린다고 항상 성능이 좋아지지 않았습니다.
서로 다른 특성을 가진 모델의 조합이 훨씬 더 중요했습니다.

5️⃣ 데이터 기반 의사결정의 확신

직감이 아닌 데이터 분석을 바탕으로 최종 모델을 선택했습니다.
데이터 중심의 결정은 최종 결과 제출에서 확신을 가져다주었습니다.

이 교훈들은 앞으로의 프로젝트에서 목적과 데이터에 기반한 실험을 지속하는 기준이 될 것입니다.
---
## 🏃‍♂️ 빠른 시작  

```bash
# 1. 저장소 복제
git clone [Your-Repo-Link]
cd [Repo-Name]

# 2. 필요 라이브러리 설치
pip install -r requirements.txt

# 3. 최종 추론 스크립트 실행
python scripts/inference/inference_two_ace_ensemble.py
```
