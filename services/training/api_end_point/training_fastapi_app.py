import os
import subprocess
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, StreamingResponse

from dask.distributed import Client, LocalCluster

from common.logger import configure_logger
from common.constants import DASK_SCHEDULER_ADDRESS

logger = configure_logger(
    logger_name="training_service",
    level="DEBUG",
    to_console=True,
    to_file=True,
    log_file_name="training_service",
)


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
    global client, TRAIN_REQUEST_COUNT, TRAIN_REQUEST_LATENCY, TRAINING_DURATION, DVC_PUSH_DURATION, DVC_STATUS_DURATION

    if client is None:
        client = start_client()

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


# ─── Metrics middleware ────────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware to record Prometheus metrics for every HTTP request.

    Safely skips metric recording if initialization hasn't completed yet.
    """
    endpoint = request.url.path

    # Defensive: if metrics aren't ready, proceed without instrumentation
    if TRAIN_REQUEST_LATENCY is None or TRAIN_REQUEST_COUNT is None:
        response = await call_next(request)
        return response

    with TRAIN_REQUEST_LATENCY.labels(endpoint=endpoint).time():
        response = await call_next(request)

    TRAIN_REQUEST_COUNT.labels(
        endpoint=endpoint, http_status=str(response.status_code)
    ).inc()
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


# ─── Debug endpoint for DVC availability ───────────────────────────────────────
@app.get("/debug_dvc")
def debug_dvc():
    """
    Internal helper to verify that the `dvc` executable is on PATH and retrieve its version.

    Returns
    -------
    dict
        Contains the resolved path, version (or error), and current PATH for diagnostics.
    """
    dvc_path = shutil.which("dvc")
    version = None
    if dvc_path:
        try:
            result = subprocess.run(
                ["dvc", "--version"], capture_output=True, text=True, check=True
            )
            version = result.stdout.strip()
        except Exception as e:
            version = f"error getting version: {e}"
    return {
        "dvc_path": dvc_path,
        "version": version,
        "PATH": os.environ.get("PATH"),
    }


def get_repo_root() -> Path:
    """
    Resolve and return the repository root directory based on this file's location.

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[3]  # IMBD-Review-Sentiment-Pipeline root


# ─── Training and DVC endpoints ───────────────────────────────────────────────
@app.post("/train")
def trigger_training():
    """
    Trigger the DVC training pipeline by running `dvc repro -f`.

    Blocks until the pipeline completes and returns status. Errors are propagated as HTTP 500.
    """
    repo_root = get_repo_root()
    try:
        with TRAINING_DURATION.time():
            result = subprocess.run(
                ["dvc", "repro", "-f"],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            logger.error("Training failed: %s", result.stderr)
            raise HTTPException(status_code=500, detail=result.stderr.strip())
        logger.info("Training succeeded: %s", result.stdout.strip())
        return {"detail": "Training pipeline triggered"}
    except FileNotFoundError:
        logger.error("dvc executable not found")
        raise HTTPException(status_code=500, detail="dvc is not installed")


@app.get("/train_stream")
def stream_training_logs():
    """
    Stream real-time logs from the DVC training process as Server-Sent Events (SSE).

    Returns
    -------
    StreamingResponse
        A text/event-stream with incremental training output.
    """
    repo_root = get_repo_root()

    def event_stream():
        try:
            with TRAINING_DURATION.time():
                process = subprocess.Popen(
                    ["dvc", "repro", "-f"],
                    cwd=repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env={"PYTHONUNBUFFERED": "1", **os.environ},
                )
                for line in iter(process.stdout.readline, ""):
                    yield f"data: {line.rstrip()}\n\n"
                process.wait()
                yield "event: end\ndata: done\n\n"
        except FileNotFoundError:
            yield "data: dvc executable not found\n\n"
        except Exception as e:
            logger.exception("Error while streaming training logs")
            yield f"data: error: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/dvc_push")
def trigger_dvc_push():
    """
    Trigger a DVC data push and Git commit after training.

    This endpoint runs the `post_train_commit.sh` script located in
    `services/training/scripts/`. The script is responsible for pushing
    updated data artifacts to the configured DVC remote and committing
    the changes to Git.

    Workflow:
        1. Determine the repository root using `get_repo_root()`.
        2. Execute `post_train_commit.sh` with Bash from the repo root.
        3. Capture stdout/stderr for logging and debugging.

    Returns:
        dict: A JSON response with a `detail` message indicating success.

    Raises:
        HTTPException (500):
            - If the script returns a non-zero exit code (subprocess error).
            - If the script file is not found at the expected location.

    Notes:
        - Requires `bash` to be installed in the environment.
        - The script must handle both `dvc push` and `git commit` internally.
    """
    repo_root = get_repo_root()
    try:
        result = subprocess.run(
            ["bash", "services/training/scripts/post_train_commit.sh"],  # script handles dvc push & git commit
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Post-train script: %s", result.stdout.strip())
        return {"detail": "Push & commit completed"}
    except subprocess.CalledProcessError as e:
        logger.error("Post-train script failed: %s", e.stderr)
        raise HTTPException(status_code=500, detail=e.stderr.strip())
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="script not found")


@app.get("/dvc_status")
def dvc_status():
    """
    Get the current DVC workspace status via `dvc status`.

    Returns
    -------
    dict
        Contains the output of `dvc status` or a clean workspace message.

    Raises
    ------
    HTTPException
        If `dvc` is missing or the status command fails.
    """
    repo_root = get_repo_root()
    try:
        with DVC_STATUS_DURATION.time():
            result = subprocess.run(
                ["dvc", "status"],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            logger.error("DVC status failed: %s", result.stderr)
            raise HTTPException(status_code=500, detail=result.stderr.strip())
        output = result.stdout.strip() or "Workspace clean"
        logger.info("DVC status: %s", output)
        return {"detail": output}
    except FileNotFoundError:
        logger.error("dvc executable not found")
        raise HTTPException(status_code=500, detail="dvc is not installed")


if __name__ == "__main__":
    """
    Entrypoint for running the training FastAPI app directly.

    Environment Variables
    ---------------------
    PORT : int, optional
        Port to bind the server to (default: 8001).
    """
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)), log_level="info")
