# Stanford Cars Fine-grained Classification

196개 세부 차량 모델을 구분하는 이미지 분류(Fine-grained Classification) 프로젝트입니다. EfficientNet-B0 기반 모델을 설계·최적화하여 **Top-1 Accuracy 90.21%, Top-5 Accuracy 99.15%**를 달성했습니다.

> "AI가 자동차의 브랜드와 모델을 정확히 구분할 수 있을까?"라는 질문에서 시작한 프로젝트입니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터셋 | [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) — 196개 클래스, 16,185장 |
| 프레임워크 | PyTorch |
| 모델 | EfficientNet-B0 |
| 진행 기간 | 2인 팀 프로젝트 |
| 담당 역할 | 데이터 수집·전처리, 모델 세팅, 학습 최적화, Grad-CAM 시각화 분석, 결과 문서화 |

데이터셋 용량이 크고 사이트 네트워크가 불안정해 단독 다운로드가 어려워, **Kaggle API로 Colab에 직접 연결**해 클래스별 폴더 구조를 그대로 확보했습니다.

---

## 2. 모델 설계

**전체 파이프라인**: 입력 데이터 처리 → 전처리 → 특징 추출/분류기 → 학습 전략

### 입력 데이터 & 전처리
- 모든 이미지를 **192×192 픽셀로 통일** (경량 모델 학습 속도 최적화)
- `RandomHorizontalFlip`으로 데이터 증강 — 차량 사진의 방향성 다양화, 실제 도로 환경의 다양한 시점 대응
- RGB 채널별 평균/표준편차 기반 정규화 — 학습 안정화 및 밝기·색감 노이즈 완화

### 특징 추출 — EfficientNet-B0 선정 이유

| | ResNet-50 | EfficientNet-B0 |
|---|---|---|
| 파라미터 수 | 약 2,560만 개 | **약 530만 개** |
| 모델 파일 크기 | 98 MB | **21 MB** |
| 특징 | 무겁고 느림 (Deep Stacking) | 가볍고 빠름 (Balanced Scaling) |

MBConv 구조 기반으로 깊이·너비·해상도를 최적 비율로 동시 조절하는 EfficientNet-B0을 선정해, 적은 연산량으로도 차량의 미세한 특징을 정교하게 포착했습니다.

### 분류기
GAP(Global Average Pooling)으로 특징을 하나의 벡터로 요약한 뒤, FC Layer를 거쳐 196개 클래스 각각에 대한 점수를 계산하고, Softmax로 확률 변환 후 최종 차종을 결정합니다.

### 학습 전략 (베이스라인)

| Optimizer | Loss Function | Learning Rate | Batch Size |
|---|---|---|---|
| Adam | CrossEntropy | 0.001 | 64 |

---

## 3. 성능 개선

베이스라인 학습에서 드러난 한계(파라미터 과다로 인한 메모리 비효율, 느린 학습 속도로 인한 실험 제약)를 분석하고, 아래 3가지 방향으로 최적화했습니다.

**① 입력 해상도 최적화**: 192px → 224px로 상향해 유사 차종 간 구별 능력 향상
**② 데이터 증강 최적화**: RandomErasing 시도 → 성능 저하로 실패 → ColorJitter로 전환해 날씨·조명·그림자 변화에 강건한 모델 구축
**③ 학습 효율 극대화**:
- AMP(FP32→FP16) 도입으로 연산량 50% 감소
- OneCycleLR 스케줄러 적용 — 학습 초반 가속, 후반 정밀 안착
- Optimizer를 Adam → **AdamW**로 전환해 가중치 감쇠를 분리, 학습 안정성과 일반화 성능 향상
- Label Smoothing이 적용된 CrossEntropy로 과적합 방지

### 최종 학습 결과

![학습 곡선](repo_assets/training_curves.png)

15 Epoch 학습 결과, **Validation Accuracy 90.20%**를 달성했습니다.

---

## 4. 검증 — 단순 정확도를 넘어선 신뢰성 확인

정확도 수치만으로는 "왜 잘 맞추는지", "얼마나 믿을 수 있는지"를 알 수 없다고 판단해, 세 가지 방식으로 모델을 추가 검증했습니다.

### ① 수치적 신뢰성 — Top-5 Accuracy

![Top-5 Accuracy](repo_assets/top5_accuracy.png)

Top-1 Accuracy는 90.21%지만, 상위 5개 예측 후보 내 정답 포함 확률(Top-5 Accuracy)은 **99.15%**입니다. 오답이더라도 정답과 유사한 범주(예: 같은 제조사·유사 차급) 내에 있다는 것을 확인해, 모델의 판단이 터무니없지 않음을 검증했습니다.

### ② 실전 범용성 — Wild Test

학습 데이터셋에 없는, 구글에서 직접 검색한 이미지로 테스트했습니다.

| 입력 이미지 | 예측 결과 |
|---|---|
| ![입력 이미지](repo_assets/wildtest_input.png) | ![예측 결과](repo_assets/wildtest_prediction.png) |

"Hyundai Genesis Sedan 2012"를 87.75% 확신도로 정확히 예측했으며, 2·3순위 후보 역시 외형이 유사한 세단 차종(Mercedes-Benz S-Class, E-Class)으로 나타나 모델이 실제 노이즈 환경에서도 합리적으로 작동함을 확인했습니다.

### ③ 논리적 타당성 — Grad-CAM

![Grad-CAM 비교](repo_assets/gradcam_comparison.png)

모델이 배경이 아닌 차량의 핵심 부위(헤드라이트, 그릴, 범퍼 등)를 근거로 판단하는지 Grad-CAM으로 시각화했습니다. 붉은 영역이 실제 차량의 특징부에 집중되어 있어, 모델이 올바른 근거로 판단을 내리고 있음을 확인했습니다. 이 과정에서 오분류 사례들을 모아 어떤 특징에서 판단 기준이 흔들리는지 기록하고, 이후 라벨링 기준을 재설정하는 근거로 활용했습니다.

---

## 5. 실무 응용 아이디어

차종 인식 기술을 다음과 같은 실무 서비스로 확장할 수 있다고 보고 정리했습니다.

- **스마트 주차·톨게이트**: 진입 시 차종 자동 식별 후 요금 할인 적용
- **중고차 플랫폼 자동화**: 판매자 등록 차량 사진 자동 분석으로 허위 매물 방지
- **지능형 방범·대포차 단속**: 실시간 교차 검증을 통한 위조 차량 검거
- **타겟 마케팅**: 차량 유형·가격대 분석 기반 맞춤형 서비스 제공

---

## 기술 스택
`Python` · `PyTorch` · `EfficientNet-B0` · `Grad-CAM` · `Kaggle API` · `Google Colab`
