from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


features_df = pd.read_parquet(
    PROCESSED_DIR / "customer_features_ml.parquet"
)

kmeans_df = pd.read_parquet(
    PROCESSED_DIR / "customer_segments_kmeans.parquet"
)

feature_columns = [
    "recency_days",
    "log_frequency",
    "log_monetary",
    "log_average_order_value",
    "log_avg_items_per_order",
    "log_unique_products",
    "purchase_span_days",
    "return_order_share",
    "return_value_share",
]

X = features_df[feature_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA
pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)


# UMAP
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

X_umap = umap_model.fit_transform(X_scaled)


projections = pd.DataFrame({
    "Customer ID": features_df["Customer ID"],
    "pca_1": X_pca[:, 0],
    "pca_2": X_pca[:, 1],
    "umap_1": X_umap[:, 0],
    "umap_2": X_umap[:, 1],
    "cluster": kmeans_df["cluster"].values
})


OUTPUT_PATH = (
    PROCESSED_DIR /
    "customer_projections.parquet"
)

projections.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Projection data created.")
print("Shape:", projections.shape)
print("PCA variance:", pca.explained_variance_ratio_)
print("Saved:", OUTPUT_PATH)