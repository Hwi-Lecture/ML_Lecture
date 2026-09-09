# Pima Indians 당뇨병 예측 대시보드

Pima Indians Diabetes 데이터셋으로 여러 분류 모델을 학습시키고,
결과를 화면에서 바로 비교/체험할 수 있는 Streamlit 대시보드입니다.

## 실행 방법

```bash
cd pima_dashboard
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속하면 됩니다.

## 화면 구성

- **데이터 살펴보기**: 원본 데이터, 결측 의심값(0으로 잘못 입력된 값), 특성별 분포, 상관관계 히트맵
- **모델 비교**: 로지스틱 회귀 / 결정트리 / 랜덤포레스트 / KNN 중 선택한 모델들의
  Accuracy·Precision·Recall·F1·ROC-AUC, 혼동행렬, ROC 커브를 한 화면에서 비교
- **직접 예측해보기**: 슬라이더로 값을 입력해 선택한 모델의 당뇨병 확률을 실시간 확인

왼쪽 사이드바에서 전처리 옵션(0값 결측 처리, 표준화), 사용할 특성, 모델, 테스트
데이터 비율을 바꾸면 화면 전체가 즉시 다시 계산됩니다.

## 데이터 출처

`data/diabetes.csv` — Pima Indians Diabetes Database
(https://raw.githubusercontent.com/jbrownlee/Datasets, 원출처 UCI/NIDDK)

## AI 에이전트와 함께 기능 확장하기

`app.py`는 기능별로 함수가 나뉘어 있어 대화로 확장하기 쉽습니다. 예시 요청:

- "SVM 모델도 선택할 수 있게 추가해줘" → `MODEL_REGISTRY`에 항목 추가
- "특성 중요도 그래프도 보여줘" → `render_compare_tab`에 트리 기반 모델의
  `feature_importances_` 시각화 추가
- "교차검증 점수도 같이 보여줘" → `train_and_evaluate`에 `cross_val_score` 추가
