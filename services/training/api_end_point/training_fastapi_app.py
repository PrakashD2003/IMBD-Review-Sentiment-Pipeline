import os
import subprocess
import shutil
import datetime
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

    This endpoint performs the same operations as the bash script but
    directly from Python, making it cross-platform compatible.
    It also auto-configures DVC remote if not already set.

    Workflow:
        1. Check and configure DVC remote if needed.
        2. Run `dvc push` to upload artifacts to remote storage.
        3. Stage DVC pipeline files (dvc.lock, dvc.yaml).
        4. Commit changes if there are any.
        5. Push to Git remote if upstream branch exists.

    Environment Variables:
        DVC_REMOTE_URL: URL for DVC remote storage (e.g., s3://bucket/path)

    Returns:
        dict: A JSON response with a `detail` message indicating success.

    Raises:
        HTTPException (500):
            - If DVC is not installed or not found in PATH.
            - If any command fails during execution.
    """
    repo_root = get_repo_root()
    
    # Check if DVC is available
    if not shutil.which("dvc"):
        logger.error("DVC executable not found in PATH")
        raise HTTPException(
            status_code=500, 
            detail="DVC is not installed or not found in PATH. Please install DVC first."
        )
    
    # Check if git is available
    if not shutil.which("git"):
        logger.error("Git executable not found in PATH")
        raise HTTPException(
            status_code=500, 
            detail="Git is not installed or not found in PATH."
        )
    
    try:
        # Step 0: Initialize DVC if not already done
        dvc_dir = repo_root / ".dvc"
        if not dvc_dir.exists():
            logger.info("Initializing DVC repository...")
            subprocess.run(
                ["dvc", "init", "--no-scm"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("DVC repository initialized.")
        
        # Step 0.5: Configure DVC remote if needed
        dvc_remote_url = os.getenv("DVC_REMOTE_URL")
        if dvc_remote_url:
            logger.info("Configuring DVC remote from environment variable...")
            
            # Check if remote already exists
            remote_list_result = subprocess.run(
                ["dvc", "remote", "list"],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            
            if "storage" in remote_list_result.stdout:
                # Remote exists, modify it
                subprocess.run(
                    ["dvc", "remote", "modify", "storage", "url", dvc_remote_url],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info("Updated existing DVC remote 'storage' to: %s", dvc_remote_url)
            else:
                # Remote doesn't exist, add it
                subprocess.run(
                    ["dvc", "remote", "add", "-d", "storage", dvc_remote_url],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info("Added DVC remote 'storage': %s", dvc_remote_url)
        else:
            # Check if any remote is configured
            remote_list_result = subprocess.run(
                ["dvc", "remote", "list"],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
            
            if not remote_list_result.stdout.strip():
                raise HTTPException(
                    status_code=500,
                    detail="No DVC remote configured. Please set DVC_REMOTE_URL environment variable or configure a remote with: dvc remote add -d storage <remote-url>"
                )
        
        # Step 1: DVC push
        logger.info("Starting DVC push...")
        with DVC_PUSH_DURATION.time():
            dvc_result = subprocess.run(
                ["dvc", "push"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        logger.info("DVC push completed: %s", dvc_result.stdout.strip())
        
        # Step 2: Stage DVC files
        logger.info("Staging DVC files...")
        subprocess.run(
            ["git", "add", "-A", "dvc.lock", "dvc.yaml", ".dvc/config"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        
        # Step 3: Check if there are changes to commit
        git_diff_result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        
        if git_diff_result.returncode != 0:
            # There are changes to commit
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            commit_message = f"Post-training commit {timestamp}"
            
            logger.info("Committing changes...")
            git_commit_result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Git commit completed: %s", git_commit_result.stdout.strip())
            
            # Step 4: Try to push to remote if upstream exists
            try:
                # Check current branch
                branch_result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                current_branch = branch_result.stdout.strip()
                
                # Check if upstream exists
                upstream_result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                )
                
                if upstream_result.returncode == 0:
                    # Upstream exists, push
                    logger.info("Pushing to origin/%s...", current_branch)
                    git_push_result = subprocess.run(
                        ["git", "push", "origin", current_branch],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    logger.info("Git push completed: %s", git_push_result.stdout.strip())
                else:
                    logger.info("No upstream set for branch %s. Skipping push.", current_branch)
                    
            except subprocess.CalledProcessError as e:
                logger.warning("Git push failed, but continuing: %s", e.stderr)
                # Don't fail the whole operation if push fails
                
        else:
            logger.info("No changes to commit.")
        
        return {"detail": "Push & commit completed successfully"}
        
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with return code %d: %s", e.returncode, e.stderr)
        raise HTTPException(
            status_code=500, 
            detail=f"Command failed: {e.stderr.strip() if e.stderr else 'Unknown error'}"
        )
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error: {str(e)}"
        )


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