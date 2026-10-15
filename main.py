import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 바탕으로 연도별 기온을 예측합니다.")


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자형으로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 사용할 수 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 2025년까지 + 관측일 300일 이상인 연도만 사용
# --------------------------------------------------
yearly = (
    df[df["연도"] <= 2025]
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일수가 300일 미만인 해 제외
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly["연평균기온"] = yearly["연평균기온"].round(2)

yearly = yearly.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 회귀 직선 계산
# --------------------------------------------------
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# --------------------------------------------------
# 1900~2100년 예측값 계산
# --------------------------------------------------
prediction_years = np.arange(1900, 2101)
prediction_temps = slope * prediction_years + intercept


def predict_temperature(year):
    return slope * year + intercept


# --------------------------------------------------
# 회귀 직선 정보
# --------------------------------------------------
st.subheader("회귀 분석 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "회귀 직선을 만든 해의 개수",
        f"{len(yearly)}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(yearly['연도'].min())}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{int(yearly['연도'].max())}년"
    )


st.info(
    f"사용한 데이터는 2025년까지이며, "
    f"각 연도의 관측일수가 300일 이상인 해만 포함했습니다."
)


# --------------------------------------------------
# 연도별 평균기온 + 회귀 직선 산점도
# --------------------------------------------------
st.subheader("서울 연평균기온과 회귀 직선")

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=[
            f"{year}년"
            for year in yearly["연도"]
        ],
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
        marker=dict(size=7)
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀 직선",
        hovertemplate=(
            "%{x}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 상관계수
# --------------------------------------------------
st.subheader("상관계수")

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{correlation:.3f}"
)

if correlation > 0:
    st.write(
        "상관계수가 양수이므로, 전체적으로 연도가 높아질수록 "
        "연평균기온이 높아지는 경향이 있습니다."
    )
elif correlation < 0:
    st.write(
        "상관계수가 음수이므로, 전체적으로 연도가 높아질수록 "
        "연평균기온이 낮아지는 경향이 있습니다."
    )
else:
    st.write(
        "연도와 연평균기온 사이에 뚜렷한 선형 상관관계가 나타나지 않습니다."
    )


# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------
st.subheader("연도를 선택해 기온 예측하기")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = predict_temperature(selected_year)

st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{predicted_temp:.2f} ℃"
)


# --------------------------------------------------
# 선택한 연도의 위치를 보여주는 그래프
# --------------------------------------------------
selected_fig = go.Figure()

selected_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temps,
        mode="lines",
        name="회귀 직선",
        hovertemplate=(
            "%{x}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

selected_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 연평균기온: {predicted_temp:.2f}℃"
            "<extra></extra>"
        )
    )
)

selected_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="예상 연평균기온 (℃)",
    height=450
)

st.plotly_chart(
    selected_fig,
    use_container_width=True
)


# --------------------------------------------------
# 데이터 확인
# --------------------------------------------------
with st.expander("회귀에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )
