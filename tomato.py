import streamlit as st
import pandas as pd
import joblib
import os

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="Tomato Smart Farm AI",
    page_icon="🍅",
    layout="wide"
)

# 2. 커스텀 CSS로 디자인 입히기
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 모델 로드 함수
@st.cache_resource
def load_model():
    model_path = "tomato_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

rf_model = load_model()

# --- 메인 화면 구성 ---
st.title("🍅 AI 기반 토마토 착과율 예측 시스템")
st.markdown("현재 온실의 환경 데이터를 입력하면 인공지능이 예상 착과율을 계산합니다.")
st.divider()

# 레이아웃 나누기 (왼쪽: 입력, 오른쪽: 결과)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 환경 데이터 입력")
    with st.container(border=True):
        temp = st.slider("🌡️ 내부 온도 (°C)", 0.0, 50.0, 24.0, 0.5)
        humidity = st.slider("💧 내부 습도 (%)", 0.0, 100.0, 60.0, 1.0)
        co2 = st.number_input("☁️ 내부 CO2 (ppm)", value=450, step=10)
        
        predict_btn = st.button("🚀 착과율 분석 시작")

with col2:
    st.subheader("📊 분석 리포트")
    
    if predict_btn:
        if rf_model is not None:
            # 데이터 변환
            input_data = pd.DataFrame([[temp, humidity, co2]], 
                                      columns=['내부온도', '내부습도', '내부CO2'])
            
            # 예측
            prediction = rf_model.predict(input_data)[0]
            
            # 결과 표시 카드 디자인
            st.markdown(f"""
                <div class="result-card">
                    <h3>예상 착과율</h3>
                    <h1 style="color: #ff4b4b; font-size: 60px;">{prediction:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("") # 간격 띄우기
            
            # 상태 요약 및 시각적 피드백
            sub_col1, sub_col2, sub_col3 = st.columns(3)
            
            with sub_col1:
                st.metric("현재 온도", f"{temp}°C")
            with sub_col2:
                st.metric("현재 습도", f"{humidity}%")
            with sub_col3:
                status = "우수" if prediction >= 70 else "보통" if prediction >= 40 else "주의"
                st.metric("진단 상태", status)

            # 결과 차트나 추가 메시지
            if prediction >= 70:
                st.success("✅ 현재 생육 환경이 매우 최적입니다. 현 상태를 유지하세요!")
            elif prediction >= 40:
                st.warning("⚠️ 생육 환경 개선이 필요할 수 있습니다. CO2 농도와 환기를 체크하세요.")
            else:
                st.error("🚨 경고: 착과율이 매우 낮습니다. 환경 제어 시스템을 점검하십시오.")
        else:
            st.error("모델 파일을 로드할 수 없습니다. 파일 이름을 확인해주세요.")
    else:
        # 버튼을 누르기 전 가이드 화면
        st.info("왼쪽 대시보드에서 데이터를 조정한 후 분석 버튼을 눌러주세요.")

# 하단 정보
st.divider()
st.caption("© 2026 SmartFarm AI Solutions - Data Driven Agriculture")