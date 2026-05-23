import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 한글 폰트 설정 (웹 환경에서는 기본 폰트 사용)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="기초통계학 학습 Playground", page_icon="📊", layout="wide")
st.title("📊 기초통계학 학습 Playground")
st.markdown("---")

# 4개의 탭 생성
tab1, tab2, tab3, tab4 = st.tabs([
    "1단계: 데이터 수집", 
    "2단계: 기술통계량 계산", 
    "3단계: 실시간 그래프 시각화", 
    "4단계: 가설 검정 및 추정"
])

# 분포 정보 딕셔너리
dist_info = {
    "정규분포 (Normal)": {"type": "continuous", "params": ["평균 (μ)", "표준편차 (σ)"], "desc": "자연계에서 가장 흔하게 나타나는 종 모양의 연속형 확률분포입니다."},
    "이항분포 (Binomial)": {"type": "discrete", "params": ["시행 횟수 (n)", "성공 확률 (p)"], "desc": "성공/실패 두 가지 결과만 나오는 시행을 n번 반복했을 때의 성공 횟수 분포입니다."},
    "포아송분포 (Poisson)": {"type": "discrete", "params": ["발생률 (λ)"], "desc": "단위 시간이나 면적 당 드물게 발생하는 사건의 발생 횟수를 나타내는 분포입니다."},
    "지수분포 (Exponential)": {"type": "continuous", "params": ["비율 모수 (λ)"], "desc": "다음 사건이 일어날 때까지 대기 시간에 대한 연속형 확률분포입니다."},
    "t-분포 (Student's t)": {"type": "continuous", "params": ["자유도 (df)"], "desc": "표본의 크기가 작고 모분산을 모를 때, 모평균을 추정/검정하는 데 사용됩니다."},
    "카이제곱분포 (Chi-square)": {"type": "continuous", "params": ["자유도 (df)"], "desc": "표준정규분포를 따르는 확률변수들의 제곱합의 분포로, 분산 검정에 주로 쓰입니다."}
}

# 공통 함수: 데이터 입력 처리
def get_data_from_input(text_input, uploaded_file):
    data = None
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            # 첫 번째 숫자 컬럼 선택
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                data = df[num_cols[0]].dropna().values
        elif text_input:
            # 쉼표나 줄바꿈, 공백으로 구분된 문자열을 숫자로 변환
            clean_text = text_input.replace('\n', ',').replace('\t', ',').replace(' ', ',')
            data = np.array([float(x.strip()) for x in clean_text.split(',') if x.strip()])
    except Exception as e:
        st.error(f"데이터 입력 오류: {e}")
    return data

# ==========================================
# 탭 1: 데이터 수집 (확률분포 샘플링)
# ==========================================
with tab1:
    st.header("🎲 1단계: 확률분포 샘플링")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        dist_choice = st.selectbox("분포를 선택하세요", list(dist_info.keys()), key="t1_dist")
        st.info(dist_info[dist_choice]["desc"])
        
        params = {}
        for p in dist_info[dist_choice]["params"]:
            if "p" in p: params[p] = st.number_input(p, min_value=0.0, max_value=1.0, value=0.5)
            elif "σ" in p: params[p] = st.number_input(p, min_value=0.01, value=1.0)
            elif "n" in p or "df" in p: params[p] = st.number_input(p, min_value=1, value=10, step=1)
            else: params[p] = st.number_input(p, value=0.0)
            
        n_samples = st.number_input("생성할 데이터 개수 (N)", min_value=1, max_value=10000, value=100)
        
        if st.button("🚀 샘플 데이터 생성", type="primary"):
            np.random.seed()
            if "정규분포" in dist_choice:
                data = np.random.normal(params["평균 (μ)"], params["표준편차 (σ)"], n_samples)
            elif "이항분포" in dist_choice:
                data = np.random.binomial(params["시행 횟수 (n)"], params["성공 확률 (p)"], n_samples)
            elif "포아송" in dist_choice:
                data = np.random.poisson(params["발생률 (λ)"], n_samples)
            elif "지수" in dist_choice:
                data = np.random.exponential(1/params["비율 모수 (λ)"], n_samples) # numpy uses scale = 1/lambda
            elif "t-분포" in dist_choice:
                data = np.random.standard_t(params["자유도 (df)"], n_samples)
            elif "카이제곱" in dist_choice:
                data = np.random.chisquare(params["자유도 (df)"], n_samples)
                
            df_sample = pd.DataFrame({"Sample Value": data})
            st.session_state['sample_data'] = df_sample # 세션 저장 (다른 탭에서 쓰기 위해)
            
    with col2:
        if 'sample_data' in st.session_state:
            st.success("데이터가 성공적으로 생성되었습니다!")
            
            # 1. 기본 데이터프레임 표시 (우측 상단 마우스 오버 시 복사/다운로드 아이콘 표시됨)
            st.dataframe(st.session_state['sample_data'], use_container_width=True)
            
            st.markdown("---")
            
            # 2. 직관적인 복사용 텍스트 박스 제공 (st.code 활용)
            st.subheader("📋 클립보드 복사용 데이터")
            st.caption("아래 박스 우측 상단의 '복사' 버튼을 누른 뒤, 2/4단계 탭의 '직접 입력' 칸에 붙여넣기 하세요.")
            
            # 데이터를 소수점 4자리까지 반올림한 뒤 쉼표로 연결한 문자열 생성
            raw_data_str = ", ".join(map(lambda x: str(round(x, 4)), st.session_state['sample_data']["Sample Value"]))
            
            # st.code를 쓰면 전용 복사 버튼이 아주 명확하게 생깁니다!
            st.code(raw_data_str, language="text")

# ==========================================
# 탭 2: 기술통계량 자동 계산
# ==========================================
with tab2:
    st.header("🧮 2단계: 기술통계량 계산")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("데이터 입력")
        st.caption("복사/붙여넣기를 하거나 파일을 업로드하세요.")
        txt_input = st.text_area("직접 입력 (숫자를 쉼표, 띄어쓰기, 줄바꿈으로 구분)", height=150)
        up_file = st.file_uploader("CSV 또는 Excel 파일 첨부", type=['csv', 'xlsx'])
        
        calc_btn = st.button("📊 기술통계량 계산 및 그래프 그리기", type="primary")
        
    with col2:
        if calc_btn:
            data = get_data_from_input(txt_input, up_file)
            
            if data is not None and len(data) > 0:
                # 통계량 계산
                stats_df = pd.DataFrame({
                    "항목": ["데이터 수 (N)", "평균 (Mean)", "분산 (Variance)", "표준편차 (Std Dev)", 
                           "중앙값 (Median)", "최빈값 (Mode)", "최솟값 (Min)", "최댓값 (Max)", "범위 (Range)"],
                    "값": [
                        len(data), np.mean(data), np.var(data, ddof=1), np.std(data, ddof=1),
                        np.median(data), stats.mode(data, keepdims=False).mode, 
                        np.min(data), np.max(data), np.max(data) - np.min(data)
                    ]
                })
                # 분위수 추가
                percentiles = np.percentile(data, [25, 50, 75])
                stats_df.loc[len(stats_df)] = ["1사분위수 (Q1, 25%)", percentiles[0]]
                stats_df.loc[len(stats_df)] = ["3사분위수 (Q3, 75%)", percentiles[2]]
                
                st.subheader("📝 계산 결과")
                st.dataframe(stats_df.set_index("항목").style.format("{:.4f}"))
                
                st.subheader("📉 히스토그램")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(data, kde=True, color='skyblue', ax=ax)
                ax.set_title("Data Histogram & KDE")
                st.pyplot(fig)
            else:
                st.warning("데이터를 제대로 입력해 주세요.")

# ==========================================
# 탭 3: 모수 변경 실시간 그래프 시각화
# ==========================================
with tab3:
    st.header("📈 3단계: 실시간 그래프 시각화")
    st.caption("아래 슬라이더를 움직여 모수 변화에 따른 그래프 모양을 실시간으로 관찰해보세요.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        dist_choice3 = st.selectbox("분포 선택", list(dist_info.keys()), key="t3_dist")
        st.info(dist_info[dist_choice3]["desc"])
        
        slider_params = {}
        for p in dist_info[dist_choice3]["params"]:
            if "p" in p: slider_params[p] = st.slider(p, 0.0, 1.0, 0.5, 0.01)
            elif "σ" in p: slider_params[p] = st.slider(p, 0.1, 10.0, 1.0, 0.1)
            elif "n" in p or "df" in p: slider_params[p] = st.slider(p, 1, 100, 10, 1)
            elif "μ" in p: slider_params[p] = st.slider(p, -20.0, 20.0, 0.0, 0.5)
            elif "λ" in p: slider_params[p] = st.slider(p, 0.1, 20.0, 5.0, 0.5)
            
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(-20, 20, 1000)
        
        if "정규분포" in dist_choice3:
            y = stats.norm.pdf(x, slider_params["평균 (μ)"], slider_params["표준편차 (σ)"])
            ax.plot(x, y, color='purple', lw=2)
            ax.set_xlim(slider_params["평균 (μ)"]-10, slider_params["평균 (μ)"]+10)
        elif "이항분포" in dist_choice3:
            x_b = np.arange(0, slider_params["시행 횟수 (n)"]+1)
            y_b = stats.binom.pmf(x_b, slider_params["시행 횟수 (n)"], slider_params["성공 확률 (p)"])
            ax.vlines(x_b, 0, y_b, colors='b', lw=5, alpha=0.5)
            ax.set_xlim(-1, slider_params["시행 횟수 (n)"]+1)
        elif "포아송" in dist_choice3:
            x_p = np.arange(0, max(20, int(slider_params["발생률 (λ)"]*2)))
            y_p = stats.poisson.pmf(x_p, slider_params["발생률 (λ)"])
            ax.vlines(x_p, 0, y_p, colors='g', lw=5, alpha=0.5)
        elif "t-분포" in dist_choice3:
            y = stats.t.pdf(x, slider_params["자유도 (df)"])
            y_norm = stats.norm.pdf(x, 0, 1) # 비교용 표준정규분포
            ax.plot(x, y, color='red', lw=2, label=f"t-dist (df={slider_params['자유도 (df)']})")
            ax.plot(x, y_norm, color='gray', linestyle='--', label="Normal(0,1)")
            ax.set_xlim(-5, 5)
            ax.legend()
        elif "카이제곱" in dist_choice3:
            x_c = np.linspace(0, 50, 1000)
            y_c = stats.chi2.pdf(x_c, slider_params["자유도 (df)"])
            ax.plot(x_c, y_c, color='orange', lw=2)
            ax.set_xlim(0, max(20, slider_params["자유도 (df)"]*2))
        
        ax.set_title(f"{dist_choice3} 형태 시각화")
        st.pyplot(fig)

# ==========================================
# 탭 4: 가설 검정 및 추정
# ==========================================
with tab4:
    st.header("⚖️ 4단계: 가설 검정 및 추정")
    
    analysis_type = st.radio("수행할 분석 선택", ["모평균 검정 및 추정 (t-분포)", "모분산 검정 및 추정 (카이제곱 분포)"], horizontal=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("데이터 입력")
        txt_input_test = st.text_area("데이터 직접 입력 (쉼표 구분)", height=100, key="txt_test")
        up_file_test = st.file_uploader("파일 첨부", type=['csv', 'xlsx'], key="file_test")
        
        st.markdown("---")
        conf_level = st.slider("신뢰수준 (%)", 80, 99, 95, 1)
        alpha = 1 - (conf_level / 100)
        
        if "모평균" in analysis_type:
            h0_val = st.number_input("검정할 귀무가설의 모평균 값 (H0)", value=0.0)
            test_type = st.selectbox("대립가설 (H1) 형태", ["같지 않다 (양측)", "크다 (우측)", "작다 (좌측)"])
        else:
            h0_val = st.number_input("검정할 귀무가설의 모분산 값 (H0)", min_value=0.01, value=1.0)
            test_type = st.selectbox("대립가설 (H1) 형태", ["같지 않다 (양측)", "크다 (우측)", "작다 (좌측)"], key="var_test_type")
            
        run_test = st.button("실행하기", type="primary")

    with col2:
        if run_test:
            data = get_data_from_input(txt_input_test, up_file_test)
            if data is not None and len(data) > 1:
                n = len(data)
                sample_mean = np.mean(data)
                sample_var = np.var(data, ddof=1)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # -----------------------------
                # 1. 모평균 검정 및 추정
                # -----------------------------
                if "모평균" in analysis_type:
                    se = np.sqrt(sample_var / n)
                    t_stat = (sample_mean - h0_val) / se
                    df = n - 1
                    
                    # p-value 계산 및 기각역 설정
                    if test_type == "같지 않다 (양측)":
                        p_val = stats.t.sf(np.abs(t_stat), df) * 2
                        cv_lower, cv_upper = stats.t.ppf(alpha/2, df), stats.t.isf(alpha/2, df)
                    elif test_type == "크다 (우측)":
                        p_val = stats.t.sf(t_stat, df)
                        cv_upper = stats.t.isf(alpha, df)
                        cv_lower = -np.inf
                    else: # 작다
                        p_val = stats.t.cdf(t_stat, df)
                        cv_lower = stats.t.ppf(alpha, df)
                        cv_upper = np.inf
                        
                    # 추정 (신뢰구간)
                    margin = stats.t.isf(alpha/2, df) * se
                    ci_lower, ci_upper = sample_mean - margin, sample_mean + margin
                    
                    # 결과 출력
                    st.success(f"**[평균 검정 결과]** 검정통계량 t = {t_stat:.4f}, p-value = {p_val:.4f}")
                    if p_val < alpha: st.error("🚨 p-value가 유의수준보다 작아 귀무가설을 **기각**합니다.")
                    else: st.info("✅ p-value가 유의수준보다 커 귀무가설을 **채택(기각 실패)**합니다.")
                    st.write(f"**[평균 추정]** {conf_level}% 신뢰구간: ({ci_lower:.4f}, {ci_upper:.4f})")
                    
                    # 시각화
                    x = np.linspace(-5, 5, 1000)
                    y = stats.t.pdf(x, df)
                    ax.plot(x, y, label=f"t-dist (df={df})")
                    
                    if test_type == "같지 않다 (양측)":
                        ax.fill_between(x, y, where=(x < cv_lower) | (x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    elif test_type == "크다 (우측)":
                        ax.fill_between(x, y, where=(x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    else:
                        ax.fill_between(x, y, where=(x < cv_lower), color='red', alpha=0.3, label='Rejection Region')
                    
                    ax.axvline(t_stat, color='green', lw=2, label=f'Test Stat: {t_stat:.2f}')
                    ax.legend()
                    st.pyplot(fig)

                # -----------------------------
                # 2. 모분산 검정 및 추정
                # -----------------------------
                else:
                    df = n - 1
                    chi2_stat = (df * sample_var) / h0_val
                    
                    if test_type == "같지 않다 (양측)":
                        p_val = 2 * min(stats.chi2.cdf(chi2_stat, df), stats.chi2.sf(chi2_stat, df))
                        cv_lower, cv_upper = stats.chi2.ppf(alpha/2, df), stats.chi2.isf(alpha/2, df)
                    elif test_type == "크다 (우측)":
                        p_val = stats.chi2.sf(chi2_stat, df)
                        cv_upper = stats.chi2.isf(alpha, df)
                        cv_lower = -np.inf
                    else: # 작다
                        p_val = stats.chi2.cdf(chi2_stat, df)
                        cv_lower = stats.chi2.ppf(alpha, df)
                        cv_upper = np.inf
                        
                    # 추정 (신뢰구간)
                    ci_lower = (df * sample_var) / stats.chi2.isf(alpha/2, df)
                    ci_upper = (df * sample_var) / stats.chi2.ppf(alpha/2, df)
                    
                    # 결과 출력
                    st.success(f"**[분산 검정 결과]** 검정통계량 X² = {chi2_stat:.4f}, p-value = {p_val:.4f}")
                    if p_val < alpha: st.error("🚨 p-value가 유의수준보다 작아 귀무가설을 **기각**합니다.")
                    else: st.info("✅ p-value가 유의수준보다 커 귀무가설을 **채택(기각 실패)**합니다.")
                    st.write(f"**[분산 추정]** {conf_level}% 신뢰구간: ({ci_lower:.4f}, {ci_upper:.4f})")
                    
                    # 시각화
                    x = np.linspace(0, max(chi2_stat * 1.5, df * 2.5), 1000)
                    y = stats.chi2.pdf(x, df)
                    ax.plot(x, y, label=f"Chi2-dist (df={df})")
                    
                    if test_type == "같지 않다 (양측)":
                        ax.fill_between(x, y, where=(x < cv_lower) | (x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    elif test_type == "크다 (우측)":
                        ax.fill_between(x, y, where=(x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    else:
                        ax.fill_between(x, y, where=(x < cv_lower), color='red', alpha=0.3, label='Rejection Region')
                        
                    ax.axvline(chi2_stat, color='green', lw=2, label=f'Test Stat: {chi2_stat:.2f}')
                    ax.legend()
                    st.pyplot(fig)
                    
            else:
                st.warning("데이터가 부족하거나 형식이 잘못되었습니다.")
