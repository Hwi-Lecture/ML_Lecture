import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="당뇨병 예측 모델 대시보드", layout="wide")

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

MODEL_REGISTRY = {
    "로지스틱 회귀": lambda: LogisticRegression(max_iter=1000),
    "결정트리": lambda: DecisionTreeClassifier(random_state=42),
    "랜덤포레스트": lambda: RandomForestClassifier(random_state=42),
    "KNN": lambda: KNeighborsClassifier(),
}


@st.cache_data
def load_data():
    return pd.read_csv("data/diabetes.csv")


def clean_data(df, fix_zeros):
    df = df.copy()
    if fix_zeros:
        for col in ZERO_INVALID_COLS:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    return df


@st.cache_data
def train_and_evaluate(df, selected_features, model_names, test_size, scale, fix_zeros):
    selected_features = list(selected_features)
    df = clean_data(df, fix_zeros)
    X = df[selected_features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

    results = {}
    for name in model_names:
        model = MODEL_REGISTRY[name]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_curve": roc_curve(y_test, y_proba),
        }
    return results, X_train, X_test, y_train, y_test


def render_data_tab(df):
    st.subheader("데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("전체 샘플 수", len(df))
        st.metric("당뇨병 양성 비율", f"{df[TARGET].mean() * 100:.1f}%")
    with col2:
        zero_counts = (df[ZERO_INVALID_COLS] == 0).sum()
        st.write("**0으로 잘못 입력된 값 개수 (결측치 의심)**")
        st.dataframe(zero_counts.rename("0값 개수"), use_container_width=True)

    st.subheader("특성별 분포")
    feature = st.selectbox("확인할 특성을 선택하세요", FEATURES)
    fig = px.histogram(df, x=feature, color=TARGET, barmode="overlay", nbins=30,
                        color_discrete_sequence=["#4C78A8", "#F58518"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("특성 간 상관관계")
    corr = df[FEATURES + [TARGET]].corr()
    fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig2, use_container_width=True)


def render_compare_tab(results):
    st.subheader("모델별 성능 지표")
    summary = pd.DataFrame({
        name: {
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "ROC-AUC": r["roc_auc"],
        }
        for name, r in results.items()
    }).T.round(3)
    st.dataframe(summary.style.highlight_max(axis=0, color="#d4f4dd"), use_container_width=True)

    best_recall = summary["Recall"].idxmax()
    best_acc = summary["Accuracy"].idxmax()
    st.info(
        f"**재현율(Recall)이 가장 높은 모델**: {best_recall} — 당뇨 환자를 놓치지 않는 것이 중요할 때 유리합니다.\n\n"
        f"**정확도(Accuracy)가 가장 높은 모델**: {best_acc} — 전체적으로 가장 잘 맞히는 모델입니다."
    )

    st.subheader("혼동행렬 (Confusion Matrix)")
    cols = st.columns(len(results))
    for col, (name, r) in zip(cols, results.items()):
        with col:
            st.write(f"**{name}**")
            cm = r["confusion_matrix"]
            fig = px.imshow(cm, text_auto=True, x=["예측:정상", "예측:당뇨"], y=["실제:정상", "실제:당뇨"],
                             color_continuous_scale="Blues")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC 커브")
    fig = go.Figure()
    for name, r in results.items():
        fpr, tpr, _ = r["roc_curve"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['roc_auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="기준선", line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)


def render_predict_tab(df, results, selected_features, scale, fix_zeros):
    st.subheader("직접 값을 입력해서 예측해보기")
    if not results:
        st.warning("먼저 왼쪽 사이드바에서 모델을 하나 이상 선택하세요.")
        return

    model_name = st.selectbox("사용할 모델", list(results.keys()))

    clean_df = clean_data(df, fix_zeros)
    input_values = {}
    cols = st.columns(2)
    for i, feat in enumerate(selected_features):
        col = cols[i % 2]
        min_v, max_v = float(clean_df[feat].min()), float(clean_df[feat].max())
        default = float(clean_df[feat].median())
        with col:
            input_values[feat] = st.slider(feat, min_v, max_v, default)

    model = results[model_name]["model"]
    X_input = pd.DataFrame([input_values])[selected_features]

    if scale:
        scaler = StandardScaler()
        scaler.fit(clean_df[selected_features])
        X_input = pd.DataFrame(scaler.transform(X_input), columns=selected_features)

    proba = model.predict_proba(X_input)[0, 1]
    st.metric("예측된 당뇨병 확률", f"{proba * 100:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0))
    if proba >= 0.5:
        st.error("모델 예측: 당뇨병 위험군")
    else:
        st.success("모델 예측: 정상 범위")


def main():
    st.title("🩺 Pima Indians 당뇨병 예측 대시보드")
    st.caption("데이터를 보고, 전처리 옵션을 바꿔보고, 여러 모델을 비교하고, 직접 값을 넣어 예측해보세요.")

    df = load_data()

    with st.sidebar:
        st.header("⚙️ 설정")
        st.subheader("전처리")
        fix_zeros = st.checkbox("0값을 결측치로 보고 중앙값으로 채우기", value=True)
        scale = st.checkbox("특성 표준화(StandardScaler) 적용", value=True)

        st.subheader("특성 선택")
        selected_features = st.multiselect("모델에 사용할 특성", FEATURES, default=FEATURES)

        st.subheader("모델 선택")
        model_names = [name for name in MODEL_REGISTRY if st.checkbox(name, value=(name in ["로지스틱 회귀", "랜덤포레스트"]))]

        test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)

    if not selected_features:
        st.warning("사이드바에서 특성을 하나 이상 선택하세요.")
        return

    results = {}
    if model_names:
        results, *_ = train_and_evaluate(
            df, tuple(selected_features), tuple(model_names), test_size, scale, fix_zeros
        )

    tab1, tab2, tab3 = st.tabs(["📊 데이터 살펴보기", "🏆 모델 비교", "🔮 직접 예측해보기"])
    with tab1:
        render_data_tab(clean_data(df, fix_zeros))
    with tab2:
        if not model_names:
            st.warning("사이드바에서 모델을 하나 이상 선택하세요.")
        else:
            render_compare_tab(results)
    with tab3:
        render_predict_tab(df, results, selected_features, scale, fix_zeros)


if __name__ == "__main__":
    main()
