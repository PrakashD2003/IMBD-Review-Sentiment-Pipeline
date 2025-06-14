import os
import logging
import uvicorn
import multiprocessing
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from dask.distributed import Client, LocalCluster

from src.pipelines.unified_prediction_pipeline import UnifiedPredictionPipeline
from src.logger import configure_logger
from src.constants import DASK_SCHEDULER_ADDRESS

module_name = Path(__file__).stem
logger = configure_logger(
    logger_name=module_name,
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name=module_name,
)

def start_client() -> Client:
    """
    Instantiate a Dask client, either connecting to a remote scheduler
    (e.g. on EKS) or spinning up a local in-process cluster.

    Returns
    -------
    Client
        The Dask distributed client that will service all Dask calls.
    """
    if DASK_SCHEDULER_ADDRESS:
        logger.info("Connecting to remote Dask scheduler at %s", DASK_SCHEDULER_ADDRESS)
        return Client(DASK_SCHEDULER_ADDRESS)
    else:
        logger.info("Starting local Dask cluster")
        cluster = LocalCluster(n_workers=8, threads_per_worker=2, memory_limit="4GB")
        return Client(cluster)

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="IMDB Sentiment Prediction API")

pipeline: UnifiedPredictionPipeline
client: Client | None = None
REQUEST_COUNT = None
REQUEST_LATENCY = None
INFERENCE_TIME = None

@app.on_event("startup")
def register_metrics():
    # Only run this in the main UVicorn process,
    # not in each Dask worker that imports the module.
    if multiprocessing.current_process().name != "MainProcess":
        return
    global REQUEST_COUNT, REQUEST_LATENCY, INFERENCE_TIME, pipeline, client
    client = start_client()
    from prometheus_client import Counter, Histogram
    # Create your “unified” pipeline only once.
    pipeline = UnifiedPredictionPipeline()
# ─── Prometheus metrics ─────────────────────────────────────────────────────────
    REQUEST_COUNT    = Counter("app_request_count",    "Total HTTP requests",    ["endpoint","method","http_status"])
    REQUEST_LATENCY  = Histogram("app_request_latency_seconds", "Request latency", ["endpoint","method"])
    INFERENCE_TIME   = Histogram("inference_time_seconds", "Time spent in model inference", ["mode"])

# Close the Dask client gracefully when the server stops
@app.on_event("shutdown")
def shutdown_event():
    if client:
        client.close()

# ─── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Payload for inference endpoints: list of raw review texts."""
    reviews: List[str]

class PredictResponse(BaseModel):
    """Response: list of predicted labels (0 or 1)."""
    predictions: List[int]

# ─── Metrics middleware ─────────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method
    with REQUEST_LATENCY.labels(endpoint=endpoint, method=method).time():
        response = await call_next(request)
    REQUEST_COUNT.labels(
        endpoint=endpoint, method=method, http_status=str(response.status_code)
    ).inc()
    return response

# ─── Single‐record inference ───────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Perform low-latency inference on a small batch of reviews.

    This handler preprocesses client-supplied text, vectorizes it,
    runs the model, and returns integer labels.
    """
    try:
        with INFERENCE_TIME.labels(mode="single").time():
            import pandas as pd
            df = pd.DataFrame({"review": req.reviews})
            result_df = pipeline.predict_single(df)
            preds = result_df["prediction"].tolist()
        return PredictResponse(predictions=preds)
    except Exception as e:
        logging.exception("Single prediction error")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Bulk (Dask-powered) inference ──────────────────────────────────────────────
@app.post("/batch_predict", response_model=PredictResponse)
async def batch_predict(req: PredictRequest):
    """
    Perform high-throughput batch inference using Dask.

    Splits input into partitions and runs them in parallel,
    then gathers and returns all predictions.
    """
    try:
        with INFERENCE_TIME.labels(mode="batch").time():
            import pandas as pd
            df = pd.DataFrame({"review": req.reviews})
            result_df = pipeline.predict_batch(df)
            preds = result_df["prediction"].tolist()
        return PredictResponse(predictions=preds)
    except Exception as e:
        logging.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Prometheus scrape endpoint ────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics for scraping.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    uvicorn.run(
        "fast_api_app.app:app", 
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
