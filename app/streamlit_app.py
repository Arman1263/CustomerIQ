from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import os



PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_intelligence.parquet"
)

PROJECTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_projections.parquet"
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

# API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CustomerIQ",
    page_icon="📊",
    layout="wide"
)


# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("CustomerIQ")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Customer Intelligence",
        "Anomalies",
        "ML Visualization"
    ]
)


# -------------------------
# Load Data
# -------------------------

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


customers = load_data()


# -------------------------
# API Helper
# -------------------------

def get_api_data(endpoint):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=5
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.RequestException:
        return None


# -------------------------
# Header
# -------------------------

st.title("CustomerIQ")
st.caption("Unsupervised Customer Intelligence Platform")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Business Overview")

    # KPIs

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{len(customers):,}"
    )

    col2.metric(
        "Revenue",
        f"₹{customers['monetary'].sum():,.0f}"
    )

    col3.metric(
        "Segments",
        customers["segment_name"].nunique()
    )

    col4.metric(
        "Anomalies",
        f"{customers['is_anomaly'].sum():,}"
    )

    st.divider()

    # Segment Summary

    st.subheader("Customer Segments")

    segment_summary = (
        customers
        .groupby("segment_name")
        .agg(
            customers=("Customer ID", "count"),
            revenue=("monetary", "sum")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    segment_summary["revenue"] = (
        segment_summary["revenue"].round(2)
    )

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Segment Distribution

    st.subheader("Segment Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(
            segment_summary.set_index("segment_name")["customers"]
        )

    with col2:
        st.bar_chart(
            segment_summary.set_index("segment_name")["revenue"]
        )


# =========================================================
# CUSTOMER INTELLIGENCE
# =========================================================

elif page == "Customer Intelligence":

    st.header("Customer Intelligence")

    customer_ids = sorted(
        customers["Customer ID"].astype(int).unique()
    )

    default_customer = customer_ids.index(12347)

    customer_id = st.selectbox(
        "Select Customer ID",
        customer_ids,
        index=default_customer
    )

    # Fetch customer intelligence from FastAPI

    customer = get_api_data(
        f"/customers/{customer_id}"
    )

    if customer is None:

        st.error(
            "Unable to connect to CustomerIQ API. "
            "Make sure FastAPI is running."
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Segment",
            customer["segment_name"]
        )

        col2.metric(
            "Revenue",
            f"₹{customer['monetary']:,.2f}"
        )

        col3.metric(
            "Orders",
            int(customer["frequency"])
        )

        st.write(
            f"**Customer ID:** "
            f"{int(customer['Customer ID'])}"
        )

        st.write(
            f"**Recommended Action:** "
            f"{customer['recommended_action']}"
        )

        if customer["is_anomaly"]:

            st.warning(
                "⚠️ This customer is flagged as anomalous."
            )

        else:

            st.success(
                "Normal customer behavior."
            )

        customer_metrics = pd.DataFrame({
            "Metric": [
                "Recency (days)",
                "Frequency",
                "Monetary",
                "Average Order Value",
                "Unique Products",
                "Return Order Share",
                "Return Value Share",
                "DBSCAN Cluster",
                "GMM Cluster"
            ],
            "Value": [
                f"{customer['recency_days']:.2f}",
                f"{int(customer['frequency'])}",
                f"₹{customer['monetary']:,.2f}",
                f"₹{customer['average_order_value']:,.2f}",
                f"{int(customer['unique_products'])}",
                f"{customer['return_order_share']:.1%}",
                f"{customer['return_value_share']:.1%}",
                f"{int(customer['dbscan_cluster'])}",
                f"{int(customer['gmm_cluster'])}"
            ]
    })

    st.dataframe(
        customer_metrics,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ANOMALIES
# =========================================================

elif page == "Anomalies":

    st.header("Anomaly Detection")

    anomaly_data = get_api_data(
        "/anomalies"
    )

    if anomaly_data is None:

        st.error(
            "Unable to connect to CustomerIQ API. "
            "Make sure FastAPI is running."
        )

    else:

        anomaly_data = pd.DataFrame(
            anomaly_data
        )

        st.metric(
            "Flagged Customers",
            f"{len(anomaly_data):,}"
        )

        st.dataframe(
            anomaly_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ML VISUALIZATION
# =========================================================

elif page == "ML Visualization":

    st.header("ML Visualization")

    projections = pd.read_parquet(
        PROJECTION_PATH
    )

    projections["cluster"] = (
        projections["cluster"].astype(str)
    )

    # -------------------------
    # PCA
    # -------------------------

    st.subheader(
        "Customer Behavior — PCA"
    )

    fig_pca = px.scatter(
        projections,
        x="pca_1",
        y="pca_2",
        color="cluster",
        hover_data=["Customer ID"],
        labels={
            "pca_1": "PC1",
            "pca_2": "PC2",
            "cluster": "Segment"
        }
    )

    fig_pca.update_layout(
        legend_title="K-Means Segment",
        height=550
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )

    # -------------------------
    # UMAP
    # -------------------------

    st.subheader(
        "Customer Behavior — UMAP"
    )

    fig_umap = px.scatter(
        projections,
        x="umap_1",
        y="umap_2",
        color="cluster",
        hover_data=["Customer ID"],
        labels={
            "umap_1": "UMAP1",
            "umap_2": "UMAP2",
            "cluster": "Segment"
        }
    )

    fig_umap.update_layout(
        legend_title="K-Means Segment",
        height=550
    )

    st.plotly_chart(
        fig_umap,
        use_container_width=True
    )