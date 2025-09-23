import io
import subprocess
import json 
import logging

import uvicorn
from fastapi.responses import StreamingResponse
from contextlib import redirect_stdout, redirect_stderr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, StreamingResponse

from dask.distributed import Client, LocalCluster

from services.training.scripts.dvc_traning_management_script import ProductionDVCTrainingManager
from common.logger import configure_logger
from common.constants import DASK_SCHEDULER_ADDRESS

configure_logger(
    logger_name="training-service",
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name="training-service.log",
)

logger = logging.getLogger("training-service")

def start_client() -> Client:
    """
    Initialize and return a Dask distributed client.

    Attempts to connect to a remote scheduler if DASK_SCHEDULER_ADDRESS is set;
    otherwise spins up a local cluster for development/fallback.

    Returns
    -------
    Client
        The initialized Dask distributed client instance.
    """
    if DASK_SCHEDULER_ADDRESS:
        logger.info("Connecting to remote Dask scheduler at %s", DASK_SCHEDULER_ADDRESS)
        return Client(DASK_SCHEDULER_ADDRESS)
    else:
        logger.info("Starting local Dask cluster")
        cluster = LocalCluster(n_workers=8, threads_per_worker=2, memory_limit="4GB")
        return Client(cluster)


# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="IMDB Training Service")

# CORS: wildcard allowed for development; tighten in production to specific origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with front-end origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state / metrics
# Initialize global training manager
training_manager: ProductionDVCTrainingManager | None = None
client: Client | None = None
TRAIN_REQUEST_COUNT = None
TRAIN_REQUEST_LATENCY = None
TRAINING_DURATION = None
DVC_PUSH_DURATION = None
DVC_STATUS_DURATION = None


@app.on_event("startup")
def startup_event():
    """
    FastAPI startup event handler.

    Initializes the Dask client and Prometheus metric objects in an idempotent way so
    repeated invocations (e.g., due to reload) do not recreate them redundantly.
    """
    global client, training_manager, TRAIN_REQUEST_COUNT, TRAIN_REQUEST_LATENCY, TRAINING_DURATION, DVC_PUSH_DURATION, DVC_STATUS_DURATION

    if client is None:
        client = start_client()

    if training_manager is None:
        training_manager = ProductionDVCTrainingManager()

    if TRAIN_REQUEST_COUNT is None:
        TRAIN_REQUEST_COUNT = Counter(
            "train_request_count", "Total train-related HTTP requests", ["endpoint", "http_status"]
        )
    if TRAIN_REQUEST_LATENCY is None:
        TRAIN_REQUEST_LATENCY = Histogram(
            "train_request_latency_seconds", "Request latency for training endpoints", ["endpoint"]
        )
    if TRAINING_DURATION is None:
        TRAINING_DURATION = Histogram(
            "training_duration_seconds", "Duration of DVC training runs (dvc repro)"
        )
    if DVC_PUSH_DURATION is None:
        DVC_PUSH_DURATION = Histogram(
            "dvc_push_duration_seconds", "Duration of DVC push commands"
        )
    if DVC_STATUS_DURATION is None:
        DVC_STATUS_DURATION = Histogram(
            "dvc_status_duration_seconds", "Duration of DVC status commands"
        )


@app.on_event("shutdown")
def shutdown_event():
    """
    FastAPI shutdown event handler.

    Closes the Dask client gracefully to release resources.
    """
    global client
    if client:
        client.close()

def get_repo_root() -> str:
    # Align with manager’s workspace
    return str(training_manager.workspace)

# ─── Metrics middleware ────────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware to record Prometheus metrics for every HTTP request.

    Safely skips metric recording if initialization hasn't completed yet.
    """
    # Prefer the route *template* (e.g., "/reproduce/{training_id}") to avoid high-cardinality labels
    route = request.scope.get("route")
    endpoint = getattr(route, "path", request.url.path)

    # Skip /metrics to reduce noise
    if endpoint == "/metrics" or TRAIN_REQUEST_LATENCY is None or TRAIN_REQUEST_COUNT is None:
        return await call_next(request)

    with TRAIN_REQUEST_LATENCY.labels(endpoint=endpoint).time():
        response = await call_next(request)

    TRAIN_REQUEST_COUNT.labels(endpoint=endpoint, http_status=str(response.status_code)).inc()
    return response




# ─── Prometheus scrape endpoint ────────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics for scraping.

    Returns
    -------
    Response
        Plain-text metrics in Prometheus exposition format.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ─── Training and DVC endpoints ───────────────────────────────────────────────
@app.post("/train")
def trigger_dvc_training():
    """Trigger DVC-managed training with complete reproducibility."""
    try:
        with TRAINING_DURATION.time():
             result = training_manager.run_training_with_dvc()
        return {
            "status": "training_completed",
            "training_id": result["post_training"]["training_id"],
            "fingerprint": result["post_training"],
            "reproduction_info": result["reproduction_commands"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/train_stream")
def stream_training_logs():
    """
    Stream real-time logs and progress from run_training_with_dvc as Server-Sent Events (SSE).
    """
    def event_stream():
        try:
            for event in training_manager.run_training_with_dvc():
                event_type = event.get('type', 'log')
                data = event.get('data', '')

                if event_type == 'progress':
                    # Send a 'progress' event with JSON data
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
                else: # Handles 'log' and 'error' types
                    # Send a standard 'message' event with text data
                    yield f"data: {data}\n\n"
            
            yield "event: end\ndata: done\n\n"
        except Exception as e:
            error_message = str(e).replace('\n', ' ')
            yield f"data: error: Training pipeline failed: {error_message}\n\n"
            yield "event: end\ndata: done\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/reproduce/{training_id}")
def reproduce_training(training_id: str):
    """Reproduce a specific training run."""
    try:
        result = training_manager.reproduce_training(training_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dvc_status")
def get_dvc_status():
    """Get current DVC pipeline status."""
    try:
        with DVC_STATUS_DURATION.time():
          result = subprocess.run(
              ["dvc", "status", "--json"], 
              capture_output=True, text=True, cwd=training_manager.workspace
        )
        
        if result.returncode == 0:
            status = json.loads(result.stdout) if result.stdout.strip() else {"status": "up_to_date"}
        else:
            status = {"error": result.stderr}
        
        return {
            "dvc_status": status,
            "workspace": str(training_manager.workspace),
            "pipeline_version": training_manager.pipeline_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training_history")
def get_training_history():
    """Get history of all training runs."""
    try:
        # List all experiments from S3
        response = training_manager.s3_client.list_objects_v2(
            Bucket=training_manager.s3_bucket,
            Prefix="experiments/",
            Delimiter="/"
        )
        
        training_runs = []
        for prefix in response.get('CommonPrefixes', []):
            training_id = prefix['Prefix'].rstrip('/').split('/')[-1]
            training_runs.append(training_id)
        
        return {
            "training_runs": sorted(training_runs, reverse=True),
            "total_count": len(training_runs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline_info")
def get_pipeline_info():
    """Get information about the current DVC pipeline."""
    try:
        # Get DVC pipeline DAG
        result = subprocess.run(
            ["dvc", "dag", "--json"], 
            capture_output=True, text=True, cwd=training_manager.workspace
        )
        
        pipeline_info = {
            "pipeline_version": training_manager.pipeline_version,
            "workspace": str(training_manager.workspace),
            "s3_bucket": training_manager.s3_bucket,
            "dvc_remote": f"s3://{training_manager.s3_bucket}/dvc-storage"
        }
        
        if result.returncode == 0 and result.stdout.strip():
            pipeline_info["dag"] = json.loads(result.stdout)
        
        return pipeline_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    """
    Entrypoint for running the training FastAPI app directly.
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")