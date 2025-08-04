"""FastAPI service for IMDB review sentiment prediction.

This module defines the web application that exposes endpoints for
running the sentiment analysis pipeline, serves a React frontend and
exports Prometheus metrics.  It also manages a Dask client to enable
distributed batch inference.
"""

import os
import uvicorn
import pandas as pd      
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from dask.distributed import Client, LocalCluster

from services.prediction.pipelines.unified_prediction_pipeline import UnifiedPredictionPipeline
from common.logger import configure_logger
from common.constants import DASK_SCHEDULER_ADDRESS

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
    Initialize and return a Dask distributed client.

    This function attempts to connect to a remote Dask scheduler if 
    `DASK_SCHEDULER_ADDRESS` is defined. If not, it will start a local 
    Dask cluster with multiple workers and threads for parallel execution.

    Returns
    -------
    Client
        The initialized Dask distributed client instance.

    Notes
    -----
    - For production environments on Kubernetes/EKS, set 
      `DASK_SCHEDULER_ADDRESS` to the scheduler service address.
    - For local development or fallback, it will spin up a 
      LocalCluster with `n_workers=8` and `threads_per_worker=2`.
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods, or specify ['POST', 'GET']
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Serve the bundled React frontend if it exists
frontend_build = Path(__file__).parent / "frontend" / "build"
if frontend_build.exists():
    app.mount("/frontend", StaticFiles(directory=frontend_build, html=True), name="frontend")
    static_dir = frontend_build / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

pipeline: UnifiedPredictionPipeline = None
client: Client | None = None
REQUEST_COUNT = None
REQUEST_LATENCY = None
INFERENCE_TIME = None

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.

    - Initializes the Dask client (local or remote).
    - Creates the unified prediction pipeline instance.
    - Configures Prometheus metrics counters and histograms.

    Notes
    -----
    Ensures that the initialization only happens in the main Uvicorn 
    process to avoid duplicate Dask clusters in worker subprocesses.
    """
    global REQUEST_COUNT, REQUEST_LATENCY, INFERENCE_TIME, pipeline, client

    # Avoid re-initializing if already done (e.g., in weird reloader situations)
    if client is None:
        client = start_client()

    if pipeline is None:
        pipeline = UnifiedPredictionPipeline()

    if REQUEST_COUNT is None:
        REQUEST_COUNT = Counter(
            "app_request_count", "Total HTTP requests", ["endpoint", "method", "http_status"]
        )
    if REQUEST_LATENCY is None:
        REQUEST_LATENCY = Histogram(
            "app_request_latency_seconds", "Request latency", ["endpoint", "method"]
        )
    if INFERENCE_TIME is None:
        INFERENCE_TIME = Histogram(
            "inference_time_seconds", "Time spent in model inference", ["mode"]
        )


# Close the Dask client gracefully when the server stops
@app.on_event("shutdown")
def shutdown_event():
    """
    FastAPI shutdown event handler.

    Closes the Dask client gracefully to ensure all workers
    are terminated and resources are released.
    """
    if client:
        client.close()


# ─── Home page ───
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the application home page.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    HTMLResponse
        The rendered HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})



# ─── Request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Request schema for prediction endpoints.

    Attributes
    ----------
    reviews : List[str]
        A list of raw text reviews to perform sentiment prediction on.
    """
    reviews: List[str]

class PredictResponse(BaseModel):
    """
    Response schema for prediction endpoints.

    Attributes
    ----------
    predictions : List[int]
        List of predicted labels (e.g., 0 = negative, 1 = positive).
    """
    predictions: List[int]

# ─── Metrics middleware ─────────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware to record Prometheus metrics for each HTTP request.

    Parameters
    ----------
    request : Request
        Incoming HTTP request.
    call_next : Callable
        Next middleware or endpoint handler in the chain.

    Returns
    -------
    Response
        The HTTP response after processing the request.
    """
    endpoint = request.url.path
    method = request.method

    # Safe fallback if metrics aren't initialized yet
    if REQUEST_LATENCY is None or REQUEST_COUNT is None:
        response = await call_next(request)
        return response

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

    Parameters
    ----------
    req : PredictRequest
        Request payload containing a list of reviews.

    Returns
    -------
    PredictResponse
        Predictions for each input review.

    Raises
    ------
    HTTPException
        - 503: If pipeline is not initialized.
        - 500: For any runtime errors during inference.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline unavailable")
    try:
        with INFERENCE_TIME.labels(mode="single").time():
            df = pd.DataFrame({"review": req.reviews})
            result_df = pipeline.predict_single(df)
            preds = result_df["prediction"].tolist()
        return PredictResponse(predictions=preds)
    except Exception as e:
        logger.exception("Single prediction error")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Bulk (Dask-powered) inference ──────────────────────────────────────────────
@app.post("/batch_predict", response_model=PredictResponse)
async def batch_predict(req: PredictRequest):
    """
    Perform high-throughput batch inference using Dask.

    Parameters
    ----------
    req : PredictRequest
        Request payload containing a list of reviews.

    Returns
    -------
    PredictResponse
        Predictions for each input review.

    Raises
    ------
    HTTPException
        - 503: If pipeline is not initialized.
        - 500: For any runtime errors during inference.
    """ 
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline unavailable")
    try:
        with INFERENCE_TIME.labels(mode="batch").time():
            import pandas as pd
            df = pd.DataFrame({"review": req.reviews})
            result_df = pipeline.predict_batch(df)
            preds = result_df["prediction"].tolist()
        return PredictResponse(predictions=preds)
    except Exception as e:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Prometheus scrape endpoint ────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics for scraping.

    Returns
    -------
    Response
        Plain-text metrics in Prometheus exposition format.
    """

    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    """
    Entrypoint for running the FastAPI app using Uvicorn.

    Environment Variables
    ---------------------
    PORT : int, optional
        Port to bind the server to (default: 8000).
    """
    uvicorn.run(
        "services.prediction.api_end_point.prediction_fastapi_app:app", 
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
