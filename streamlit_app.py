import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
        }
        section[data-testid="stSidebar"] {
            background: #102a43;
        }
        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        .dashboard-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: #102a43;
            margin-bottom: 0.25rem;
        }
        .dashboard-subtitle {
            color: #486581;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: white;
            border-radius: 18px;
            padding: 1rem 1.25rem;
            box-shadow: 0 10px 30px rgba(16, 42, 67, 0.08);
            border: 1px solid rgba(16, 42, 67, 0.06);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "train.csv"
    df = pd.read_csv(data_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order Week"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def monthly_sales_frame(data: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        data.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"]
        .sum()
        .reset_index()
        .sort_values("Order Date")
    )
    return monthly


def build_monthly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    frame = monthly.copy().set_index("Order Date")
    frame["Lag_1"] = frame["Sales"].shift(1)
    frame["Lag_2"] = frame["Sales"].shift(2)
    frame["Lag_3"] = frame["Sales"].shift(3)
    frame["Rolling_Mean_3"] = frame["Sales"].rolling(3).mean().shift(1)
    frame["Month"] = frame.index.month
    frame["Quarter"] = frame.index.quarter
    frame["Season"] = frame["Month"].apply(get_season)
    frame = pd.get_dummies(frame, columns=["Season"], drop_first=True)
    return frame.dropna().reset_index()


def _feature_columns(feature_frame: pd.DataFrame) -> list[str]:
    return [column for column in feature_frame.columns if column not in {"Order Date", "Sales"}]


def train_xgb(monthly: pd.DataFrame):
    feature_frame = build_monthly_features(monthly)
    if len(feature_frame) < 4:
        return None, None, None

    feature_cols = _feature_columns(feature_frame)
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(feature_frame[feature_cols], feature_frame["Sales"])
    return model, feature_cols, feature_frame


def recursive_forecast(
    model: XGBRegressor,
    feature_cols: list[str],
    monthly: pd.DataFrame,
    periods: int,
) -> pd.Series:
    history = monthly.copy().sort_values("Order Date").reset_index(drop=True)
    future_predictions: list[float] = []
    future_dates: list[pd.Timestamp] = []

    for _ in range(periods):
        if len(history) < 3:
            break

        next_date = history.iloc[-1]["Order Date"] + pd.offsets.MonthBegin(1)
        new_row = {
            "Lag_1": float(history.iloc[-1]["Sales"]),
            "Lag_2": float(history.iloc[-2]["Sales"]),
            "Lag_3": float(history.iloc[-3]["Sales"]),
            "Rolling_Mean_3": float(history.tail(3)["Sales"].mean()),
            "Month": int(next_date.month),
            "Quarter": int(next_date.quarter),
            "Season_Spring": int(get_season(next_date.month) == "Spring"),
            "Season_Summer": int(get_season(next_date.month) == "Summer"),
            "Season_Winter": int(get_season(next_date.month) == "Winter"),
        }

        row = pd.DataFrame([new_row]).reindex(columns=feature_cols, fill_value=0)
        prediction = float(model.predict(row)[0])

        future_predictions.append(prediction)
        future_dates.append(next_date)
        history = pd.concat(
            [history, pd.DataFrame({"Order Date": [next_date], "Sales": [prediction]})],
            ignore_index=True,
        )

    return pd.Series(future_predictions, index=future_dates)


def evaluate_model(monthly: pd.DataFrame) -> tuple[float, float]:
    monthly = monthly.copy().sort_values("Order Date").reset_index(drop=True)
    if len(monthly) < 6:
        return float("nan"), float("nan")

    test_size = min(3, max(1, len(monthly) // 4))
    if len(monthly) - test_size < 4:
        test_size = 1

    train_months = monthly.iloc[:-test_size].copy()
    test_months = monthly.iloc[-test_size:].copy()

    model, feature_cols, _ = train_xgb(train_months)
    if model is None or feature_cols is None:
        return float("nan"), float("nan")

    predictions = recursive_forecast(model, feature_cols, train_months, test_size)
    if len(predictions) != len(test_months):
        return float("nan"), float("nan")

    actual = test_months["Sales"].to_numpy(dtype=float)
    predicted = predictions.to_numpy(dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return float(mae), float(rmse)


@st.cache_data(show_spinner=False)
def get_forecast_bundle(scope_type: str, scope_value: str, horizon: int):
    df = load_data()
    subset = df[df[scope_type] == scope_value].copy()
    monthly = monthly_sales_frame(subset)
    model, feature_cols, _ = train_xgb(monthly)

    if model is None or feature_cols is None:
        return monthly, pd.Series(dtype=float), float("nan"), float("nan")

    forecast = recursive_forecast(model, feature_cols, monthly, horizon)
    mae, rmse = evaluate_model(monthly)
    return monthly, forecast, mae, rmse


@st.cache_data(show_spinner=False)
def get_anomaly_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data()
    weekly_sales = (
        df.set_index("Order Date")["Sales"]
        .resample("W")
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "Week"})
    )

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales["Anomaly"] = iso.fit_predict(weekly_sales[["Sales"]])
    anomalies = weekly_sales[weekly_sales["Anomaly"] == -1].copy()
    return weekly_sales, anomalies


def get_cluster_label(cluster_row: pd.Series) -> str:
    if cluster_row["Growth Rate"] >= cluster_row["Growth Rate Median"] and cluster_row["Total Sales Rank"] >= 2:
        return "Growing Demand"
    if cluster_row["Total Sales Rank"] == 0:
        return "Low Volume, Stable Demand"
    if cluster_row["Volatility Rank"] == 3 or cluster_row["Growth Rate"] < 0:
        return "Declining Demand"
    return "High Volume, Stable Demand"


@st.cache_data(show_spinner=False)
def get_cluster_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data().copy()

    total_sales = df.groupby("Sub-Category")["Sales"].sum()
    avg_order = df.groupby("Sub-Category")["Sales"].mean()
    volatility = df.groupby("Sub-Category")["Sales"].std().fillna(0)

    yearly_sales = (
        df.assign(Year=df["Order Date"].dt.year)
        .groupby(["Sub-Category", "Year"])["Sales"]
        .sum()
        .reset_index()
    )
    pivot = yearly_sales.pivot(index="Sub-Category", columns="Year", values="Sales").fillna(0)
    growth_rate = pivot.pct_change(axis=1).replace([np.inf, -np.inf], np.nan).mean(axis=1).fillna(0)

    features = pd.DataFrame(
        {
            "Total Sales": total_sales,
            "Growth Rate": growth_rate,
            "Volatility": volatility,
            "Average Order Value": avg_order,
        }
    ).fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    cluster_frame = features.copy()
    cluster_frame["Cluster"] = clusters

    cluster_summary = cluster_frame.groupby("Cluster").agg(
        {
            "Total Sales": "mean",
            "Growth Rate": "mean",
            "Volatility": "mean",
            "Average Order Value": "mean",
        }
    )

    cluster_summary["Total Sales Rank"] = cluster_summary["Total Sales"].rank(method="first", ascending=True).astype(int) - 1
    cluster_summary["Volatility Rank"] = cluster_summary["Volatility"].rank(method="first", ascending=True).astype(int) - 1
    cluster_summary["Growth Rate Median"] = float(cluster_summary["Growth Rate"].median())
    labels = {
        cluster_id: get_cluster_label(cluster_summary.loc[cluster_id])
        for cluster_id in cluster_summary.index
    }

    cluster_frame["Demand Segment"] = cluster_frame["Cluster"].map(labels)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    cluster_frame["PC1"] = components[:, 0]
    cluster_frame["PC2"] = components[:, 1]
    return cluster_frame.reset_index().rename(columns={"index": "Sub-Category"}), cluster_summary


def render_header() -> None:
    st.markdown('<div class="dashboard-title">Sales Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Interactive overview, forecast explorer, anomaly detection, and demand segmentation built from the training data.</div>',
        unsafe_allow_html=True,
    )


def render_page_1(df: pd.DataFrame) -> None:
    st.subheader("Sales Overview Dashboard")

    regions = sorted(df["Region"].dropna().unique().tolist())
    categories = sorted(df["Category"].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        selected_regions = st.multiselect("Filter by region", regions, default=regions)
    with col2:
        selected_categories = st.multiselect("Filter by category", categories, default=categories)

    filtered = df[df["Region"].isin(selected_regions) & df["Category"].isin(selected_categories)].copy()

    if filtered.empty:
        st.warning("No records match the selected filters.")
        return

    yearly_sales = filtered.groupby(filtered["Order Date"].dt.year)["Sales"].sum()
    monthly_sales = filtered.groupby(pd.Grouper(key="Order Date", freq="MS"))["Sales"].sum()

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Filtered sales", f"${filtered['Sales'].sum():,.0f}")
    metric_b.metric("Orders", f"{len(filtered):,}")
    metric_c.metric("Average shipping days", f"{filtered['Shipping Days'].mean():.1f}")

    col_left, col_right = st.columns(2)
    with col_left:
        fig, ax = plt.subplots(figsize=(8, 4))
        yearly_sales.plot(kind="bar", ax=ax, color="#1f77b4")
        ax.set_title("Total Sales by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sales")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, clear_figure=True)

    with col_right:
        fig, ax = plt.subplots(figsize=(8, 4))
        monthly_sales.plot(ax=ax, color="#ff7f0e", linewidth=2)
        ax.set_title("Monthly Sales Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Sales by Region and Category")
    region_category = pd.pivot_table(
        filtered,
        values="Sales",
        index="Region",
        columns="Category",
        aggfunc="sum",
        fill_value=0,
    )

    left, right = st.columns([1.15, 1])
    with left:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        sns.heatmap(region_category, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title("Sales Heatmap")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.dataframe(region_category.style.format("${:,.0f}"), use_container_width=True)


def render_page_2(df: pd.DataFrame) -> None:
    st.subheader("Forecast Explorer")

    scope_type = st.selectbox("Forecast by", ["Category", "Region"])
    options = sorted(df[scope_type].dropna().unique().tolist())
    scope_value = st.selectbox(f"Select {scope_type.lower()}", options)
    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3, step=1)

    monthly, forecast, mae, rmse = get_forecast_bundle(scope_type, scope_value, horizon)

    if monthly.empty:
        st.warning("Not enough monthly data to build a forecast for this selection.")
        return

    history = monthly.copy().set_index("Order Date")
    forecast_frame = forecast.rename("Forecast")

    fig, ax = plt.subplots(figsize=(12, 5))
    history["Sales"].plot(ax=ax, label="Historical sales", color="#1f77b4", linewidth=2)
    if not forecast_frame.empty:
        forecast_frame.plot(ax=ax, label="Forecast", color="#d62728", linewidth=2, marker="o")
    ax.set_title(f"{scope_type} Forecast for {scope_value}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    metric_left, metric_right = st.columns(2)
    metric_left.metric("MAE", "N/A" if np.isnan(mae) else f"{mae:,.2f}")
    metric_right.metric("RMSE", "N/A" if np.isnan(rmse) else f"{rmse:,.2f}")

    if forecast_frame.empty:
        st.info("Forecast could not be generated for this selection.")
    else:
        st.dataframe(forecast_frame.reset_index().rename(columns={"index": "Forecast Month"}), use_container_width=True)


def render_page_3() -> None:
    st.subheader("Anomaly Report")

    weekly_sales, anomalies = get_anomaly_data()
    st.metric("Detected anomalies", len(anomalies))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly_sales["Week"], weekly_sales["Sales"], color="#1f77b4", linewidth=2, label="Weekly sales")
    ax.scatter(anomalies["Week"], anomalies["Sales"], color="#d62728", s=60, label="Anomalies")
    ax.set_title("Isolation Forest - Weekly Sales Anomalies")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    report = anomalies[["Week", "Sales"]].copy().sort_values("Week")
    report["Week"] = report["Week"].dt.date
    st.dataframe(report, use_container_width=True)


def render_page_4() -> None:
    st.subheader("Product Demand Segments")

    cluster_frame, _ = get_cluster_data()

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("deep", n_colors=cluster_frame["Cluster"].nunique())
    for cluster_id, color in zip(sorted(cluster_frame["Cluster"].unique()), palette):
        subset = cluster_frame[cluster_frame["Cluster"] == cluster_id]
        ax.scatter(subset["PC1"], subset["PC2"], s=120, color=color, label=f"Cluster {cluster_id}")
        for _, row in subset.iterrows():
            ax.text(row["PC1"], row["PC2"], row["Sub-Category"], fontsize=8, alpha=0.85)
    ax.set_title("Product Demand Segmentation using K-Means")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Sub-category Cluster Assignments")
    table = cluster_frame[["Sub-Category", "Cluster", "Demand Segment", "Total Sales", "Growth Rate", "Volatility", "Average Order Value"]].copy()
    table = table.sort_values(["Cluster", "Total Sales"], ascending=[True, False])
    st.dataframe(
        table.style.format(
            {
                "Total Sales": "${:,.0f}",
                "Growth Rate": "{:.2%}",
                "Volatility": "{:,.2f}",
                "Average Order Value": "${:,.2f}",
            }
        ),
        use_container_width=True,
    )


def main() -> None:
    df = load_data()

    with st.sidebar:
        st.markdown("## Sales Dashboard")
        page = st.radio(
            "Navigate",
            [
                "Sales Overview Dashboard",
                "Forecast Explorer",
                "Anomaly Report",
                "Product Demand Segments",
            ],
        )
        st.markdown("---")
        st.caption("Built from train.csv")

    render_header()

    if page == "Sales Overview Dashboard":
        render_page_1(df)
    elif page == "Forecast Explorer":
        render_page_2(df)
    elif page == "Anomaly Report":
        render_page_3()
    else:
        render_page_4()


if __name__ == "__main__":
    main()