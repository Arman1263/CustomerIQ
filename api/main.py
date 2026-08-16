from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_intelligence.parquet"
)

app = FastAPI(
    title="CustomerIQ API",
    description="Customer intelligence API powered by unsupervised ML.",
    version="1.0.0"
)

customers = pd.read_parquet(DATA_PATH)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "customers_loaded": len(customers)
    }


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    result = customers[
        customers["Customer ID"] == customer_id
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer not found"
        )

    return result.iloc[0].to_dict()


@app.get("/segments")
def get_segments():
    summary = (
        customers
        .groupby(["cluster", "segment_name"])
        .agg(
            customers=("Customer ID", "count"),
            revenue=("monetary", "sum"),
            average_revenue=("monetary", "mean")
        )
        .reset_index()
    )

    return summary.to_dict(orient="records")


@app.get("/anomalies")
def get_anomalies():
    result = customers[
        customers["is_anomaly"] == True
    ].copy()

    result = result[
        [
            "Customer ID",
            "monetary",
            "frequency",
            "recency_days",
            "return_order_share",
            "return_value_share",
            "anomaly_score",
            "segment_name"
        ]
    ]

    return result.to_dict(orient="records")



@app.get("/customers")
def search_customers(
    segment: str | None = None,
    anomaly: bool | None = None
):
    result = customers.copy()

    if segment:
        result = result[
            result["segment_name"].str.lower() == segment.lower()
        ]

    if anomaly is not None:
        result = result[
            result["is_anomaly"] == anomaly
        ]

    return result.to_dict(orient="records")