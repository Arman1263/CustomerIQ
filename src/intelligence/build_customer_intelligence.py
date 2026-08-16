from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


customer_features = pd.read_parquet(
    PROCESSED_DIR / "customer_features.parquet"
)

kmeans = pd.read_parquet(
    PROCESSED_DIR / "customer_segments_kmeans.parquet"
)

dbscan = pd.read_parquet(
    PROCESSED_DIR / "customer_segments_dbscan.parquet"
)

gmm = pd.read_parquet(
    PROCESSED_DIR / "customer_segments_gmm.parquet"
)

anomalies = pd.read_parquet(
    PROCESSED_DIR / "customer_anomalies.parquet"
)


intelligence = customer_features.copy()


intelligence = intelligence.merge(
    kmeans[["Customer ID", "cluster"]],
    on="Customer ID",
    how="left"
)

intelligence = intelligence.merge(
    dbscan[["Customer ID", "dbscan_cluster"]],
    on="Customer ID",
    how="left"
)

intelligence = intelligence.merge(
    gmm[["Customer ID", "gmm_cluster"]],
    on="Customer ID",
    how="left"
)

intelligence = intelligence.merge(
    anomalies[
        ["Customer ID", "anomaly_score", "is_anomaly"]
    ],
    on="Customer ID",
    how="left"
)


OUTPUT_PATH = (
    PROCESSED_DIR /
    "customer_intelligence.parquet"
)

intelligence.to_parquet(
    OUTPUT_PATH,
    index=False
)


print("Customer Intelligence created.")
print("Shape:", intelligence.shape)
print("Saved:", OUTPUT_PATH)
print("\nColumns:")
print(intelligence.columns.tolist())



# Business segment labels

# Business segment labels

segment_names = {
    0: "Dormant / One-Time",
    1: "Return-Prone",
    2: "Recent Growing",
    3: "High-Value Loyal",
    4: "Return-Heavy High-AOV"
}

segment_actions = {
    0: "Run targeted reactivation campaigns.",
    1: "Investigate return drivers and improve retention.",
    2: "Use cross-sell and loyalty offers to increase value.",
    3: "Prioritize retention, VIP treatment, and personalized offers.",
    4: "Investigate unusual return behavior before increasing incentives."
}

intelligence["segment_name"] = intelligence["cluster"].map(segment_names)

intelligence["recommended_action"] = intelligence["cluster"].map(
    segment_actions
)


# SAVE — this must come AFTER the mapping above

OUTPUT_PATH = (
    PROCESSED_DIR /
    "customer_intelligence.parquet"
)

intelligence.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Customer Intelligence created.")
print("Shape:", intelligence.shape)
print("Saved:", OUTPUT_PATH)
print("\nColumns:")
print(intelligence.columns.tolist())