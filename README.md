# 🏆 Systematic ML Experimentation for Image Classification - (F1_score: 0.9462)

💡 **Project Focus**  
This repository is a deep dive into the systematic process of developing a high-performing image classification model.  
It highlights my experience in building a reproducible experimentation pipeline using WandB, PyTorch Lightning, and Hydra, and showcases a data-driven approach to model optimization.  

---

**English | [한국어](#-체계적인-실험을-통한-이미지-분류-경진대회)**  

## 🎯 Project Overview  

This project details my end-to-end process for a Document Image Classification Challenge with 17 classes.  
The primary goal was to systematically improve model performance by moving beyond baseline models.  
I designed and executed a highly organized experimentation workflow, conducting over **168 tracked experiments** to find the optimal data handling, modeling, and inference strategies.  
This rigorous, data-driven approach culminated in a sophisticated ensemble model that achieved my personal best score on the final leaderboard.  
 
> Tracking over 167 experiments in W&B to find the winning strategy.  
<img width="1286" height="526" alt="스크린샷 2025-07-18 오후 5 43 30" src="https://github.com/user-attachments/assets/a3472d87-fd36-4e75-b4f2-47d00bf2759d" />
<img width="1062" height="428" alt="스크린샷 2025-07-18 오후 5 45 17" src="https://github.com/user-attachments/assets/04fe3959-c753-4754-aa64-837f5fc97846" />

---

## 💻 Tech Stack  

| Category | Technology | Purpose in Project |
|---|---|---|
| ML & Training | Python, PyTorch, PyTorch Lightning | Used PyTorch Lightning to structure training code cleanly and reduce boilerplate. All models were built on the PyTorch framework. |
| Experiment Mgmt | Weights & Biases (W&B), Hydra | Tracked all 168+ experiments with W&B, visualizing metrics to compare models. Managed all complex configurations with Hydra, enabling rapid iteration. |
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

---

## 🚀 Key Strategies and Achievements  

### 1️⃣ Data-Driven Problem Definition via Error Analysis  

I didn't just train models; I started by analyzing why the baseline model failed.  
I created a detailed **"Error Notebook"** by manually reviewing misclassified images.  

> **Key Insight:** The model consistently confused visually similar classes, such as `pharmaceutical_receipt` and `confirmation_of_admission_and_discharge`.  
> This shifted my focus from finding a single "best" model to building an ensemble that could resolve these ambiguities.  

---

### 2️⃣ Hypothesis-Driven Modeling (Successes & Failures)  

I treated each experiment as a hypothesis test, learning from both successes and failures.  
This iterative process was the engine of my performance improvement.  

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

> ![IMAGE: Screenshot of the final comparison table used for model selection]  
> Making the final decision by comparing top models against curated "hard cases".  

---

## 📊 Measurable Results  
  
- **Final Leaderboard Score:** F1-score of **0.9462**  
- **Systematic Experimentation:** Tracked **168+ experiments** in W&B  
<img width="957" height="820" alt="스크린샷 2025-07-18 오후 5 49 16" src="https://github.com/user-attachments/assets/c663adc0-facd-4f65-9b8d-177187accd12" />

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
WandB, PyTorch Lightning, Hydra를 활용하여 재현 가능한 실험 파이프라인을 구축하고, 데이터 기반 분석을 통해 모델 성능을 체계적으로 개선한 경험 했습니다.  

---

## 🎯 프로젝트 개요  

17개 클래스를 분류하는 문서 이미지 분류 경진대회의 전 과정을 담았습니다.  
단순히 모델을 학습시키는 것을 넘어, 체계적인 실험 워크플로우를 설계하고 **167회 이상의 실험**을 수행했습니다.  
오답 노트를 활용한 심층 분석으로 약점을 찾아내고, 앙상블 및 추론 전략을 적용해 개인 최고 기록을 달성했습니다.  

<img width="1286" height="526" alt="스크린샷 2025-07-18 오후 5 43 30" src="https://github.com/user-attachments/assets/a3472d87-fd36-4e75-b4f2-47d00bf2759d" />
<img width="1062" height="428" alt="스크린샷 2025-07-18 오후 5 45 17" src="https://github.com/user-attachments/assets/04fe3959-c753-4754-aa64-837f5fc97846" />

---

## 💻 기술 스택  

| 분야 | 기술 | 프로젝트 내 역할 |
|---|---|---|
| ML & 학습 | Python, PyTorch, PyTorch Lightning | PyTorch Lightning으로 코드를 구조화하고 PyTorch를 기반으로 모델을 구축 및 학습 |
| 실험 관리 | Weights & Biases (W&B), Hydra | 168회 이상의 실험을 W&B로 추적하고, Hydra로 복잡한 설정을 관리 |
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
| 모델은 많을수록 앙상블에 좋다. | 값진 실패 | 모델 수보다 다양성과 조합이 중요 |

---

### 3️⃣ 마지막 한 수: 데이터 기반의 최종 결정  

마지막 날 두 개의 앙상블 중 오답 노트를 기준으로 테스트하여 **1814 모델**을 최종 선택했습니다.  
데이터 기반의 결정이 개인 최고 기록으로 이어졌습니다. (F1_score: 0.9462)

---

## 📊 정량적 성과  

- **리더보드 점수:** **F1-Score 0.9462**  
- **167회 이상의 실험 관리 및 분석**  
<img width="957" height="820" alt="스크린샷 2025-07-18 오후 5 49 16" src="https://github.com/user-attachments/assets/c663adc0-facd-4f65-9b8d-177187accd12" />

---

## 🏃‍♂️ 빠른 시작  

```bash
# 1. 저장소 복제
git clone [Your-Repo-Link]
cd [Repo-Name]

# 2. 필요 라이브러리 설치
pip install -r requirements.txt

# 3. 최종 추론 스크립트 실행
python scripts/inference_two_ace_ensemble.py
```

---
