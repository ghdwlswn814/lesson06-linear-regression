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
st.write("서울의 연평균기온 데이터를 바탕으로 장기적인 기온 변화를 살펴봅니다.")


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# --------------------------------------------------
# 연도별 평균기온 계산
# 2025년 이후 제외
# 관측일수 300일 미만 제외
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

yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

yearly["연평균기온"] = yearly["연평균기온"].round(2)


# --------------------------------------------------
# 전체 기간 회귀
# --------------------------------------------------
x_all = yearly["연도"].to_numpy(dtype=float)
y_all = yearly["연평균기온"].to_numpy(dtype=float)

slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)

correlation = np.corrcoef(x_all, y_all)[0, 1]

# 100년당 변화량
change_100_all = slope_all * 100


# --------------------------------------------------
# 최근 20년 회귀
# --------------------------------------------------
latest_year = int(yearly["연도"].max())
recent_start_year = latest_year - 19

recent20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()

x_recent = recent20["연도"].to_numpy(dtype=float)
y_recent = recent20["연평균기온"].to_numpy(dtype=float)

slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1
)

change_100_recent = slope_recent * 100


# --------------------------------------------------
# 회귀 직선 정보
# --------------------------------------------------
st.subheader("📈 기온 상승 속도")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{change_100_all:+.2f} ℃ / 100년"
    )
    st.caption(
        f"{int(yearly['연도'].min())}~{latest_year}년"
    )

with col2:
    st.metric(
        "최근 20년",
        f"{change_100_recent:+.2f} ℃ / 100년"
    )
    st.caption(
        f"{recent_start_year}~{latest_year}년"
    )


st.info(
    "기울기는 연평균기온의 회귀 직선에서 계산했습니다. "
    "예를 들어 +1.20℃/100년은 현재의 추세가 계속된다고 가정할 때 "
    "100년에 약 1.20℃ 높아지는 추세라는 뜻입니다."
)


# --------------------------------------------------
# 전체 기간과 최근 20년 비교
# --------------------------------------------------
st.subheader("전체 기간과 최근 20년의 상승 속도 비교")

comparison = pd.DataFrame({
    "기간": [
        "전체 기간",
        "최근 20년"
    ],
    "100년당 기온 변화": [
        change_100_all,
        change_100_recent
    ]
})

comparison_fig = go.Figure()

comparison_fig.add_trace(
    go.Bar(
        x=comparison["기간"],
        y=comparison["100년당 기온 변화"],
        text=[
            f"{value:+.2f}℃"
            for value in comparison["100년당 기온 변화"]
        ],
        textposition="outside",
        hovertemplate=(
            "%{x}<br>"
            "100년당 기온 변화: %{y:+.2f}℃"
            "<extra></extra>"
        )
    )
)

comparison_fig.update_layout(
    xaxis_title="분석 기간",
    yaxis_title="100년당 기온 변화 (℃)",
    height=450
)

st.plotly_chart(
    comparison_fig,
    use_container_width=True
)


# --------------------------------------------------
# 산점도 + 전체 기간 회귀 직선
# --------------------------------------------------
st.subheader("서울 연평균기온과 회귀 직선")

prediction_years = np.arange(1900, 2101)

prediction_all = (
    slope_all * prediction_years
    + intercept_all
)

fig = go.Figure()

# 실제 데이터
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
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

# 전체 기간 회귀 직선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_all,
        mode="lines",
        name="전체 기간 회귀 직선",
        hovertemplate=(
            "%{x}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선
recent_prediction = (
    slope_recent * prediction_years
    + intercept_recent
)

fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=recent_prediction,
        mode="lines",
        name="최근 20년 회귀 직선",
        hovertemplate=(
            "%{x}년<br>"
            "최근 20년 추세 예상: %{y:.2f}℃"
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
st.subheader("🔗 연도와 연평균기온의 상관관계")

st.metric(
    "상관계수",
    f"{correlation:.3f}"
)

if correlation > 0:
    st.write(
        "상관계수가 양수이므로 전체적으로 연도가 높아질수록 "
        "연평균기온도 높아지는 경향이 있습니다."
    )
elif correlation < 0:
    st.write(
        "상관계수가 음수이므로 전체적으로 연도가 높아질수록 "
        "연평균기온이 낮아지는 경향이 있습니다."
    )
else:
    st.write(
        "연도와 연평균기온 사이에 뚜렷한 선형 상관관계가 나타나지 않습니다."
    )


# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------
st.subheader("🌡️ 연도를 선택해 기온 예측하기")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = (
    slope_all * selected_year
    + intercept_all
)

st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{predicted_temp:.2f} ℃"
)


# --------------------------------------------------
# 선택한 연도 표시
# --------------------------------------------------
selected_fig = go.Figure()

selected_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_all,
        mode="lines",
        name="전체 기간 회귀 직선",
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
        marker=dict(size=16),
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
# 회귀에 사용한 데이터
# --------------------------------------------------
with st.expander("회귀에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )
