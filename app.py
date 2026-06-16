import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 그래프 한글 깨짐 방지를 위해 기본 폰트(영문 표준) 사용 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="기초통계학 학습 Playground", page_icon="📊", layout="wide")
st.title("📊 기초통계학 학습 Playground")
st.markdown("---")

# 4개의 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1단계: 데이터 수집", 
    "2단계: 기술통계량 계산", 
    "3단계: 실시간 그래프 시각화", 
    "4단계: 가설 검정 및 추정",
    "5단계: 상관분석 및 회귀분석"
])

# 구조화된 분포 정보 딕셔너리 (공식 LaTeX 추가)
dist_info = {
    "연속분포 (Continuous)": {
        "정규분포 (Normal)": {
            "params": ["평균 (μ)", "표준편차 (σ)"],
            "desc": "자연계에서 가장 흔하게 나타나는 종 모양의 연속형 확률분포입니다.",
            "formula": r"$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$"
        },
        "지수분포 (Exponential)": {
            "params": ["비율 모수 (λ)"],
            "desc": "다음 사건이 일어날 때까지 대기 시간에 대한 연속형 확률분포입니다.",
            "formula": r"$$f(x) = \lambda e^{-\lambda x} \quad (x \ge 0)$$"
        },
        "t-분포 (Student's t)": {
            "params": ["자유도 (df)"],
            "desc": "표본의 크기가 작고 모분산을 모를 때, 모평균을 추정/검정하는 데 사용됩니다.",
            "formula": r"$$f(t) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})} \left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$"
        },
        "카이제곱분포 (Chi-square)": {
            "params": ["자유도 (df)"],
            "desc": "표준정규분포를 따르는 확률변수들의 제곱합의 분포로, 분산 검정에 주로 쓰입니다.",
            "formula": r"$$f(x) = \frac{1}{2^{\nu/2}\Gamma(\nu/2)} x^{\nu/2-1} e^{-x/2} \quad (x > 0)$$"
        }
    },
    "이산분포 (Discrete)": {
        "이항분포 (Binomial)": {
            "params": ["시행 횟수 (n)", "성공 확률 (p)"],
            "desc": "성공/실패 두 가지 결과만 나오는 시행을 n번 반복했을 때의 성공 횟수 분포입니다.",
            "formula": r"$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$"
        },
        "포아송분포 (Poisson)": {
            "params": ["발생률 (λ)"],
            "desc": "단위 시간이나 면적 당 드물게 발생하는 사건의 발생 횟수를 나타내는 분포입니다.",
            "formula": r"$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$"
        }
    }
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
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                data = df[num_cols[0]].dropna().values
        elif text_input:
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
        dist_type = st.selectbox("분포 유형을 선택하세요", list(dist_info.keys()), key="t1_type")
        dist_choice = st.selectbox("세부 분포를 선택하세요", list(dist_info[dist_type].keys()), key="t1_dist")
        
        st.info(dist_info[dist_type][dist_choice]["desc"])
        st.markdown("**확률밀도/질량함수 공식:**")
        st.markdown(dist_info[dist_type][dist_choice]["formula"])
        
        params = {}
        for p in dist_info[dist_type][dist_choice]["params"]:
            if "p" in p: params[p] = st.number_input(p, min_value=0.0, max_value=1.0, value=0.5)
            elif "σ" in p: params[p] = st.number_input(p, min_value=0.01, value=1.0)
            elif "n" in p or "df" in p: params[p] = st.number_input(p, min_value=1, value=10, step=1)
            elif "λ" in p: params[p] = st.number_input(p, min_value=0.01, value=1.0)
            else: params[p] = st.number_input(p, value=0.0)
            
        n_samples = st.number_input("생성할 데이터 개수 (N)", min_value=1, max_value=10000, value=100)
        
        if st.button("🚀 샘플 데이터 생성", type="primary"):
            np.random.seed()
            if "정규분포" in dist_choice:
                data = np.random.normal(params["평균 (μ)"], params["표준편차 (σ)"], n_samples)
            elif "지수분포" in dist_choice:
                data = np.random.exponential(1/params["비율 모수 (λ)"], n_samples)
            elif "이항분포" in dist_choice:
                data = np.random.binomial(params["시행 횟수 (n)"], params["성공 확률 (p)"], n_samples)
            elif "포아송" in dist_choice:
                data = np.random.poisson(params["발생률 (λ)"], n_samples)
            elif "t-분포" in dist_choice:
                data = np.random.standard_t(params["자유도 (df)"], n_samples)
            elif "카이제곱" in dist_choice:
                data = np.random.chisquare(params["자유도 (df)"], n_samples)
                
            st.session_state['sample_data'] = pd.DataFrame({"Sample Value": data})
            
    with col2:
        if 'sample_data' in st.session_state:
            st.success("데이터가 성공적으로 생성되었습니다!")
            st.dataframe(st.session_state['sample_data'], use_container_width=True)
            st.markdown("---")
            st.subheader("📋 클립보드 복사용 데이터")
            st.caption("아래 박스 우측 상단의 복사 버튼을 눌러 다른 단계의 '직접 입력'에 붙여넣으세요.")
            raw_data_str = ", ".join(map(lambda x: str(round(x, 4)), st.session_state['sample_data']["Sample Value"]))
            st.code(raw_data_str, language="text")

# ==========================================
# 탭 2: 기술통계량 자동 계산
# ==========================================
with tab2:
    st.header("🧮 2단계: 기술통계량 계산")
    col1, col2 = st.columns([1, 2])
    
    if 't2_analyzed' not in st.session_state:
        st.session_state['t2_analyzed'] = False
    
    with col1:
        st.subheader("데이터 입력")
        txt_input = st.text_area("직접 입력 (숫자를 쉼표, 띄어쓰기, 줄바꿈으로 구분)", height=120, key="t2_txt")
        up_file = st.file_uploader("CSV 또는 Excel 파일 첨부", type=['csv', 'xlsx'], key="t2_file")
        
        # 원래 유지하고 싶으셨던 분석 실행용 전용 버튼 추가 (입력 도중 튀는 현상 방지)
        if st.button("🚀 분석 실행", type="primary", key="t2_run_btn"):
            st.session_state['t2_analyzed'] = True
            
        # 데이터가 아예 비어있으면 초기화
        if not txt_input and up_file is None:
            st.session_state['t2_analyzed'] = False
            
        data = None
        if st.session_state['t2_analyzed']:
            data = get_data_from_input(txt_input, up_file)
        
        if data is not None and len(data) > 0:
            st.markdown("---")
            st.subheader("⚙️ 그래프 커스텀 설정")
            show_hist = st.toggle("히스토그램(막대) 표시", value=True)
            show_kde = st.toggle("밀도곡선(Line) 표시", value=True)
            
            # 가로 스크롤 이동에 맞춰 개수와 두께가 정비례/반비례 연동되도록 수정
            hist_bins = st.slider("막대 개수 및 두께 조절 (왼쪽: 두껍고 적게 / 오른쪽: 얇고 많게)", min_value=5, max_value=100, value=30)
            
            custom_axis = st.checkbox("축 범위 직접 지정 (Axis Limits)")
            if custom_axis:
                c1, c2 = st.columns(2)
                xmin = c1.number_input("X축 최솟값 (X-Min)", value=float(np.min(data)))
                xmax = c2.number_input("X축 최댓값 (X-Max)", value=float(np.max(data)))
                ymin = c1.number_input("Y축 최솟값 (Y-Min)", value=0.0, step=0.01)
                ymax = c2.number_input("Y축 최댓값 (Y-Max)", value=1.0, step=0.1)

    with col2:
        if data is not None and len(data) > 0:
            # 통계량 계산 테이블 출력
            stats_df = pd.DataFrame({
                "항목": ["데이터 수 (N)", "평균 (Mean)", "분산 (Variance)", "표준편차 (Std Dev)", 
                       "중앙값 (Median)", "최빈값 (Mode)", "최솟값 (Min)", "최댓값 (Max)", "범위 (Range)"],
                "값": [
                    len(data), np.mean(data), np.var(data, ddof=1), np.std(data, ddof=1),
                    np.median(data), stats.mode(data, keepdims=False).mode, 
                    np.min(data), np.max(data), np.max(data) - np.min(data)
                ]
            })
            percentiles = np.percentile(data, [25, 75])
            stats_df.loc[len(stats_df)] = ["1사분위수 (Q1, 25%)", percentiles[0]]
            stats_df.loc[len(stats_df)] = ["3사분위수 (Q3, 75%)", percentiles[1]]
            
            st.subheader("📝 계산 결과")
            st.dataframe(stats_df.set_index("항목").style.format("{:.4f}"))
            
            # 그래프 그리기
            st.subheader("📉 시각화 결과 (영문 표기 변경)")
            fig, ax = plt.subplots(figsize=(8, 4))
            
            if show_hist and show_kde:
                sns.histplot(data, bins=hist_bins, kde=True, stat="density", color='skyblue', ax=ax)
            elif show_hist:
                sns.histplot(data, bins=hist_bins, kde=False, stat="density", color='skyblue', ax=ax)
            elif show_kde:
                sns.kdeplot(data, color='dodgerblue', fill=True, ax=ax)
                
            ax.set_title("Data Histogram & Density Plot (KDE)", fontsize=12)
            ax.set_xlabel("Measured Value")
            ax.set_ylabel("Density")
            
            if custom_axis:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                
            st.pyplot(fig)
        else:
            st.info("💡 왼쪽 칸에 데이터를 입력하거나 샘플 데이터를 복사해온 후 '🚀 분석 실행' 버튼을 누르면 분석이 시작됩니다.")

# ==========================================
# 탭 3: 모수 변경 실시간 그래프 시각화
# ==========================================
with tab3:
    st.header("📈 3단계: 실시간 그래프 시각화")
    st.caption("모수 변화 및 축 범위 수동 조절에 따른 이론적 확률분포의 형태 변화를 관찰하세요.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        dist_type3 = st.selectbox("분포 유형 선택", list(dist_info.keys()), key="t3_type")
        dist_choice3 = st.selectbox("분포 선택", list(dist_info[dist_type3].keys()), key="t3_dist")
        
        st.info(dist_info[dist_type3][dist_choice3]["desc"])
        st.markdown("**이론 공식:**")
        st.markdown(dist_info[dist_type3][dist_choice3]["formula"])
        
        slider_params = {}
        for p in dist_info[dist_type3][dist_choice3]["params"]:
            if "p" in p: slider_params[p] = st.slider(p, 0.0, 1.0, 0.5, 0.01)
            elif "σ" in p: slider_params[p] = st.slider(p, 0.1, 10.0, 1.0, 0.1)
            elif "n" in p or "df" in p: slider_params[p] = st.slider(p, 1, 100, 10, 1)
            elif "μ" in p: slider_params[p] = st.slider(p, -20.0, 20.0, 0.0, 0.5)
            elif "λ" in p: slider_params[p] = st.slider(p, 0.1, 20.0, 5.0, 0.5)
            
        st.markdown("---")
        st.subheader("⚙️ 축 범위 및 옵션 커스텀")
        
        # 2단계와 통일성을 갖춰 개수와 두께가 동시 연동되는 슬라이더로 변경 패치
        hist_bins3 = st.slider("막대 개수 및 두께 조절 (왼쪽: 두껍고 적게 / 오른쪽: 얇고 많게)", min_value=5, max_value=100, value=30, key="t3_bins")
        
        default_xmin, default_xmax = -10.0, 10.0
        default_ymin, default_ymax = 0.0, 1.0
        if "정규분포" in dist_choice3:
            default_xmin = slider_params["평균 (μ)"] - 4 * slider_params["표준편차 (σ)"]
            default_xmax = slider_params["평균 (μ)"] + 4 * slider_params["표준편차 (σ)"]
        elif "이항분포" in dist_choice3:
            default_xmin, default_xmax = -1.0, float(slider_params["시행 횟수 (n)"] + 1)
        elif "포아송" in dist_choice3:
            default_xmin, default_xmax = -1.0, float(max(20, slider_params["발생률 (λ)"] * 2.5))
        elif "지수분포" in dist_choice3:
            default_xmin, default_xmax = 0.0, float(5 / slider_params["비율 모수 (λ)"])
        elif "t-분포" in dist_choice3:
            default_xmin, default_xmax = -5.0, 5.0
            default_ymin, default_ymax = 0.0, 0.5
        elif "카이제곱" in dist_choice3:
            default_xmin, default_xmax = 0.0, float(max(20, slider_params["자유도 (df)"] * 2.5))
            default_ymin, default_ymax = 0.0, 0.25

        custom_axis3 = st.checkbox("축 범위 직접 입력하기", key="t3_custom_ax")
        xmin3, xmax3, ymin3, ymax3 = default_xmin, default_xmax, default_ymin, default_ymax
        if custom_axis3:
            c3_1, c3_2 = st.columns(2)
            xmin3 = c3_1.number_input("X-Axis Min", value=float(default_xmin))
            xmax3 = c3_2.number_input("X-Axis Max", value=float(default_xmax))
            ymin3 = c3_1.number_input("Y-Axis Min", value=float(default_ymin), step=0.05)
            ymax3 = c3_2.number_input("Y-Axis Max", value=float(default_ymax), step=0.05)
            
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        eval_xmin = xmin3 if custom_axis3 else default_xmin
        eval_xmax = xmax3 if custom_axis3 else default_xmax
        x = np.linspace(eval_xmin, eval_xmax, 1000)
        
        # 실시간 데이터 시뮬레이션을 생성하여 슬라이더 조절 시 막대바 개수와 두께가 동적으로 변화하는 모습을 가시화
        sample_size = 5000
        np.random.seed(42)
        
        if "정규분포" in dist_choice3:
            samples = stats.norm.rvs(slider_params["평균 (μ)"], slider_params["표준편차 (σ)"], size=sample_size)
            sns.histplot(samples, bins=hist_bins3, stat="density", color='purple', alpha=0.25, ax=ax, label="Sample Histogram")
            
            y = stats.norm.pdf(x, slider_params["평균 (μ)"], slider_params["표준편차 (σ)"])
            ax.plot(x, y, color='purple', lw=2, label="Normal PDF")
            
        elif "지수분포" in dist_choice3:
            samples = stats.expon.rvs(scale=1/slider_params["비율 모수 (λ)"], size=sample_size)
            sns.histplot(samples, bins=hist_bins3, stat="density", color='teal', alpha=0.25, ax=ax, label="Sample Histogram")
            
            y = stats.expon.pdf(x, scale=1/slider_params["비율 모수 (λ)"])
            ax.plot(x, y, color='teal', lw=2, label="Exponential PDF")
            
        elif "이항분포" in dist_choice3:
            samples = stats.binom.rvs(slider_params["시행 횟수 (n)"], slider_params["성공 확률 (p)"], size=sample_size)
            sns.histplot(samples, bins=hist_bins3, stat="density", color='blue', alpha=0.25, ax=ax, label="Sample Histogram")
            
            x_b = np.arange(max(0, int(eval_xmin)), min(int(eval_xmax)+1, slider_params["시행 횟수 (n)"]+1))
            y_b = stats.binom.pmf(x_b, slider_params["시행 횟수 (n)"], slider_params["성공 확률 (p)"])
            ax.vlines(x_b, 0, y_b, colors='darkblue', lw=2, alpha=0.7, label="Theoretical PMF")
            ax.plot(x_b, y_b, 'bo', ms=4)
            
        elif "포아송" in dist_choice3:
            samples = stats.poisson.rvs(slider_params["발생률 (λ)"], size=sample_size)
            sns.histplot(samples, bins=hist_bins3, stat="density", color='green', alpha=0.25, ax=ax, label="Sample Histogram")
            
            x_p = np.arange(max(0, int(eval_xmin)), int(eval_xmax)+1)
            y_p = stats.poisson.pmf(x_p, slider_params["발생률 (λ)"])
            ax.vlines(x_p, 0, y_p, colors='darkgreen', lw=2, alpha=0.7, label="Poisson PMF")
            ax.plot(x_p, y_p, 'go', ms=4)
            
        elif "t-분포" in dist_choice3:
            samples = stats.t.rvs(slider_params["자유도 (df)"], size=sample_size)
            sns.histplot(samples, bins=hist_bins3, stat="density", color='red', alpha=0.25, ax=ax, label="Sample Histogram")
            
            y = stats.t.pdf(x, slider_params["자유도 (df)"])
            y_norm = stats.norm.pdf(x, 0, 1)
            ax.plot(x, y, color='red', lw=2, label=f"t-dist (df={slider_params['자유도 (df)']})")
            ax.plot(x, y_norm, color='gray', linestyle='--', label="Normal (0,1)")
            
        elif "카이제곱" in dist_choice3:
            samples = stats.chi2.rvs(slider_params["자유도 (df)"], size=sample_size)
            sns.histplot(samples, bins=hist_bins3, stat="density", color='orange', alpha=0.25, ax=ax, label="Sample Histogram")
            
            y_c = stats.chi2.pdf(x, slider_params["자유도 (df)"])
            ax.plot(x, y_c, color='orange', lw=2, label="Chi-Square PDF")
            
        ax.set_title(f"Theoretical & Sample Profile", fontsize=14)
        ax.set_xlabel("Random Variable (X)")
        ax.set_ylabel("Probability Density / Mass")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if custom_axis3:
            ax.set_xlim(xmin3, xmax3)
            ax.set_ylim(ymin3, ymax3)
        else:
            ax.set_xlim(default_xmin, default_xmax)
            ax.set_ylim(default_ymin, default_ymax)
            
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
                
                if "모평균" in analysis_type:
                    se = np.sqrt(sample_var / n)
                    t_stat = (sample_mean - h0_val) / se
                    df = n - 1
                    
                    if test_type == "같지 않다 (양측)":
                        p_val = stats.t.sf(np.abs(t_stat), df) * 2
                        cv_lower, cv_upper = stats.t.ppf(alpha/2, df), stats.t.isf(alpha/2, df)
                    elif test_type == "크다 (우측)":
                        p_val = stats.t.sf(t_stat, df)
                        cv_upper = stats.t.isf(alpha, df)
                        cv_lower = -np.inf
                    else:
                        p_val = stats.t.cdf(t_stat, df)
                        cv_lower = stats.t.ppf(alpha, df)
                        cv_upper = np.inf
                        
                    margin = stats.t.isf(alpha/2, df) * se
                    ci_lower, ci_upper = sample_mean - margin, sample_mean + margin
                    
                    st.success(f"**[평균 검정 결과]** t = {t_stat:.4f}, p-value = {p_val:.4f}")
                    if p_val < alpha: st.error("🚨 p-value < alpha: 귀무가설(H0)을 기각합니다.")
                    else: st.info("✅ p-value >= alpha: 귀무가설(H0)을 채택합니다.")
                    st.write(f"**[평균 추정]** {conf_level}% 신뢰구간: ({ci_lower:.4f}, {ci_upper:.4f})")
                    
                    x = np.linspace(-5, 5, 1000)
                    y = stats.t.pdf(x, df)
                    ax.plot(x, y, label="t-distribution")
                    if test_type == "같지 않다 (양측)":
                        ax.fill_between(x, y, where=(x < cv_lower) | (x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    elif test_type == "크다 (우측)":
                        ax.fill_between(x, y, where=(x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    else:
                        ax.fill_between(x, y, where=(x < cv_lower), color='red', alpha=0.3, label='Rejection Region')
                    ax.axvline(t_stat, color='green', lw=2, label=f'Test Stat: {t_stat:.2f}')
                    ax.set_title("Hypothesis Testing - Mean (t-Test)")
                
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
                    else:
                        p_val = stats.chi2.cdf(chi2_stat, df)
                        cv_lower = stats.chi2.ppf(alpha, df)
                        cv_upper = np.inf
                        
                    ci_lower = (df * sample_var) / stats.chi2.isf(alpha/2, df)
                    ci_upper = (df * sample_var) / stats.chi2.ppf(alpha/2, df)
                    
                    st.success(f"**[분산 검정 결과]** X² = {chi2_stat:.4f}, p-value = {p_val:.4f}")
                    if p_val < alpha: st.error("🚨 p-value < alpha: 귀무가설(H0)을 기각합니다.")
                    else: st.info("✅ p-value >= alpha: 귀무가설(H0)을 채택합니다.")
                    st.write(f"**[분산 추정]** {conf_level}% 신뢰구간: ({ci_lower:.4f}, {ci_upper:.4f})")
                    
                    x = np.linspace(0, max(chi2_stat * 1.5, df * 2.5), 1000)
                    y = stats.chi2.pdf(x, df)
                    ax.plot(x, y, label="Chi2-distribution")
                    if test_type == "같지 않다 (양측)":
                        ax.fill_between(x, y, where=(x < cv_lower) | (x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    elif test_type == "크다 (우측)":
                        ax.fill_between(x, y, where=(x > cv_upper), color='red', alpha=0.3, label='Rejection Region')
                    else:
                        ax.fill_between(x, y, where=(x < cv_lower), color='red', alpha=0.3, label='Rejection Region')
                    ax.axvline(chi2_stat, color='green', lw=2, label=f'Test Stat: {chi2_stat:.2f}')
                    ax.set_title("Hypothesis Testing - Variance (Chi-Square Test)")
                
                ax.set_xlabel("Test Statistic Value")
                ax.set_ylabel("Density")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("데이터가 부족하거나 형식이 잘못되었습니다.")

# =========================================================================
# 5단계: 상관분석 및 회귀분석 탭 (기존 파일 맨 아래에 그대로 이어 붙이세요)
# =========================================================================
with tab5:
    st.header("📈 5단계: 상관분석 및 회귀분석")
    st.markdown("""
    이 단계에서는 사용자가 설정한 가상의 모집단 관계($Y = aX + b$)에 정규분포를 따르는 무작위 오차항을 더해 가상 데이터를 생성하고, 
    이를 바탕으로 상관분석 및 선형회귀분석을 수행합니다.
    """)

    # ------------------------------------------
    # 1. 데이터 생성 섹션
    # ------------------------------------------
    st.subheader("1. 데이터 생성 (Data Generation)")
    
    # 세션 상태 초기화 (탭을 이동해도 데이터가 증발하지 않도록 유지)
    if 'reg_df' not in st.session_state:
        st.session_state.reg_df = None
    if 'reg_x_label' not in st.session_state:
        st.session_state.reg_x_label = "독립변수(X)"
    if 'reg_y_label' not in st.session_state:
        st.session_state.reg_y_label = "종속변수(Y)"

    # 입력 레이아웃 분할 (3열 구조)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        a_true = st.number_input("참 기울기 (a)", value=2.0, step=0.1, key="reg_a_true", help="y = ax + b 에서의 기울기")
        b_true = st.number_input("참 절편 (b)", value=5.0, step=0.1, key="reg_b_true", help="y = ax + b 에서의 Y절편")
    
    with col2:
        sigma = st.number_input("오차항 표준편차 (σ)", value=3.0, min_value=0.1, step=0.1, key="reg_sigma", help="값이 클수록 데이터가 회귀선에서 멀리 흩어집니다.")
        n_samples = st.number_input("데이터 쌍 개수 (n)", value=50, min_value=10, max_value=500, step=10, key="reg_n_samples")
        
    with col3:
        x_min = st.number_input("X의 최소값", value=0.0, step=1.0, key="reg_x_min")
        x_max = st.number_input("X의 최대값", value=10.0, step=1.0, key="reg_x_max")

    # 변수 이름 커스텀 입력 (가로축/세로축 명칭 직접 설정)
    st.markdown("##### 🏷️ 변수 이름 설정 (그래프 및 결과 레이블에 반영)")
    c_name1, c_name2 = st.columns(2)
    with c_name1:
        x_input_name = st.text_input("가로축 (X) 변수명 입력", value="공부 시간 (시간)", key="reg_x_name")
    with c_name2:
        y_input_name = st.text_input("세로축 (Y) 변수명 입력", value="시험 점수 (점)", key="reg_y_name")

    # 데이터 생성 버튼
    if st.button("데이터 생성하기", type="primary", key="reg_gen_btn"):
        if x_min >= x_max:
            st.error("X의 최소값은 최대값보다 작아야 합니다.")
        else:
            # 1) X 데이터 무작위 생성 (균등분포)
            x_val = np.random.uniform(x_min, x_max, int(n_samples))
            # 2) 오차항 생성 (평균 0, 표준편차 sigma인 정규분포)
            error = np.random.normal(0, sigma, int(n_samples))
            # 3) Y 데이터 생성 (y = ax + b + noise)
            y_val = a_true * x_val + b_true + error
            
            # 생성 데이터를 데이터프레임 구조로 변환 후 세션에 바인딩
            st.session_state.reg_df = pd.DataFrame({'X': x_val, 'Y': y_val})
            st.session_state.reg_x_label = x_input_name
            st.session_state.reg_y_label = y_input_name
            st.success(f"🎯 참 식 Y = {a_true}X + {b_true} 규칙을 따르는 {n_samples}개의 데이터 쌍이 생성되었습니다!")

    # 데이터가 정상적으로 생성되어 세션에 보관 중일 때만 시각화 및 분석창 오픈
    if st.session_state.reg_df is not None:
        df = st.session_state.reg_df
        xl = st.session_state.reg_x_label
        yl = st.session_state.reg_y_label
        
        st.markdown("▼ **생성된 데이터 미리보기 (상위 5개 행)**")
        preview_df = df.copy()
        preview_df.columns = [xl, yl]
        st.dataframe(preview_df.head(), use_container_width=True)

        # ------------------------------------------
        # 2. 상관 & 회귀 분석 섹션
        # ------------------------------------------
        st.write("---")
        st.subheader("2. 상관 & 회귀 분석 결과 (Correlation & Regression)")
        
        if st.button("📊 상관 & 회귀 분석 실행", key="reg_analysis_btn"):
            x = df['X']
            y = df['Y']
            n = len(x)
            
            # SciPy 연산 모듈로 선형 회귀 추정
            res = stats.linregress(x, y)
            r_coef = res.rvalue
            r_squared = r_coef ** 2
            a_est = res.slope
            b_est = res.intercept
            a_p = res.pvalue
            a_stderr = res.stderr
            
            # 신뢰구간 및 검정통계량 수동 산출 (scipy 표준 연산)
            dof = n - 2
            t_crit = stats.t.ppf(0.975, dof)
            
            # 기울기(a) 95% 신뢰구간
            a_ci_lower = a_est - t_crit * a_stderr
            a_ci_upper = a_est + t_crit * a_stderr
            a_t = a_est / a_stderr if a_stderr != 0 else 0
            
            # 절편(b) 표준오차 및 신뢰구간 구하기
            if hasattr(res, 'intercept_stderr') and res.intercept_stderr is not None:
                b_stderr = res.intercept_stderr
            else:
                b_stderr = a_stderr * np.sqrt(np.sum(x**2) / n)
                
            b_ci_lower = b_est - t_crit * b_stderr
            b_ci_upper = b_est + t_crit * b_stderr
            b_t = b_est / b_stderr if b_stderr != 0 else 0
            b_p = 2 * (1 - stats.t.cdf(abs(b_t), dof))

            # 레이아웃 배치 (좌측: 시각화 차트, 우측: 계량 지표 테이블)
            result_col1, result_col2 = st.columns([1.1, 0.9])
            
            with result_col1:
                st.markdown("#### 📉 산점도 및 추정 회귀선")
                
                fig, ax = plt.subplots(figsize=(6, 5))
                # 관측치 산점도 플로팅
                ax.scatter(x, y, color='#1f77b4', alpha=0.7, edgecolors='none', label='생성 데이터')
                
                # 최소제곱법 기반 추정 회귀선 계산 및 플로팅
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = a_est * x_line + b_est
                ax.plot(x_line, y_line, color='#d62728', linewidth=2.5, 
                        label=f'회귀선: Ŷ = {a_est:.2f}X + {b_est:.2f}')
                
                # 축 이름 커스텀 연동 및 꾸미기
                ax.set_xlabel(xl, fontsize=11)
                ax.set_ylabel(yl, fontsize=11)
                ax.set_title(f"[{xl}]과 [{yl}]의 회귀분석 결과", fontsize=13, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper left')
                
                st.pyplot(fig)
                
            with result_col2:
                st.markdown("#### 📋 주요 통계 분석 지표")
                
                # 주요 지표 상단 위젯 배치
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(label="상관계수 (Pearson r)", value=f"{r_coef:.4f}")
                with m2:
                    st.metric(label="결정계수 (R²)", value=f"{r_squared:.4f}")
                
                st.markdown("**✏️ 최종 추정 회귀식**")
                st.info(f"**{yl} = {a_est:.4f} × [{xl}] + {b_est:.4f}**")
                
                # 대학교재 표준 형태의 통계량 요약 요약표(Summary Table) 생성
                st.markdown("**🔬 회귀계수 검정 및 95% 신뢰구간**")
                
                summary_data = {
                    "추정치 (Coef)": [b_est, a_est],
                    "표준오차 (Std Err)": [b_stderr, a_stderr],
                    "t-값 (t-stat)": [b_t, a_t],
                    "p-값 (P>|t|)": [b_p, a_p],
                    "95% 하한 (Lower)": [b_ci_lower, a_ci_lower],
                    "95% 상한 (Upper)": [b_ci_upper, a_ci_upper]
                }
                
                summary_df = pd.DataFrame(summary_data, index=["절편 (b)", f"기울기 (a, {xl})"])
                st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)
                
                # 유의수준 5% 기준 가설검정 해석 가이드 자동 텍스트 출력
                st.markdown("**💡 통계적 해석 가이드**")
                if a_p < 0.05:
                    st.success(f"기울기(a)의 p-값이 **{a_p:.4f}**로 0.05보다 작으므로, 유의수준 5%에서 **{xl}은 {yl}에 통계적으로 유의미한 영향**을 미친다고 결론 내릴 수 있습니다.")
                else:
                    st.warning(f"기울기(a)의 p-값이 **{a_p:.4f}**로 0.05보다 크므로, 유의수준 5%에서 **{xl}이 {yl}에 미치는 영향은 통계적으로 유의미하지 않습니다**.")
