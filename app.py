import streamlit as st

# 웹페이지 기본 설정
st.set_page_config(
    page_title="기초통계학 학습 Playground",
    page_icon="📊",
    layout="centered"
)

# 대문 타이틀
st.title("📊 기초통계학 학습 Playground")
st.subheader("기초 통계학에서 배우는 모든 것들을 실험해볼 수 있는 페이지입니다.")

st.markdown("---")

st.markdown("### 🛠️ 구현 예정 기능 로드맵")
st.checkbox("1단계: 데이터 수집 (확률분포 샘플링)", value=False, disabled=True)
st.checkbox("2단계: 기술통계량 자동 계산", value=False, disabled=True)
st.checkbox("3단계: 모수 변경 실시간 그래프 시각화", value=False, disabled=True)
st.checkbox("4단계: 가설 검정 및 추정 (p-value 계산기)", value=False, disabled=True)
