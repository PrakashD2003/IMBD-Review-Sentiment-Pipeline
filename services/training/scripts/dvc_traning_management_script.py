"""
Production DVC implementation for training service.
Focuses on reproducibility and artifact management without Git dependency.
"""

import os
import json
import hashlib
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any

import boto3

from common.logger import configure_logger

logger = configure_logger(
    logger_name="production_dvc_training",
    level="INFO",
    to_console=True,
    to_file=True,
    log_file_name="production_training"
)

class ProductionDVCTrainingManager:
    """
    Manages DVC-based training in production without Git dependency.
    Provides complete reproducibility and artifact tracking.
    """
    
    def __init__(self):
        self.workspace = Path(os.getenv("WORKSPACE_DIR", Path.cwd()))
        self.s3_bucket = os.getenv("DVC_S3_BUCKET")
        self.no_scm = os.getenv("DVC_NO_SCM", "true").lower() in ("1", "true", "yes")
        self.pipeline_version = os.getenv("PIPELINE_VERSION", "latest")
        self.s3_client = boto3.client('s3')
        
        # Ensure workspace exists
        self.workspace.mkdir(exist_ok=True)
        os.chdir(self.workspace)
        
        # Initialize DVC environment
        
        self.setup_dvc_environment()
    
    def setup_dvc_environment(self):
        """Initialize DVC environment for production training."""
        try:
            logger.info("Setting up DVC environment for production training...")
            
            # 1) Init DVC {no-scm(without Git) only if requested}
            if not (self.workspace / ".dvc").exists():
                cmd = ["dvc", "init"]
                if self.no_scm:
                    cmd.append("--no-scm")
                subprocess.run(cmd, check=True, cwd=self.workspace)
            
            # 2. Configure S3 remote storage
            if self.s3_bucket:
                dvc_remote_url = f"s3://{self.s3_bucket}/dvc-storage"
                
                # Remove existing storage remote if any
                subprocess.run(["dvc", "remote", "remove", "storage"], 
                             capture_output=True, cwd=self.workspace)
                
                # Add production storage remote
                subprocess.run([
                    "dvc", "remote", "add", "-d", "storage", dvc_remote_url
                ], check=True, cwd=self.workspace)
                
                logger.info(f"DVC remote configured: {dvc_remote_url}")
            else:
                raise ValueError("DVC_S3_BUCKET environment variable not set")
            
            # 3. Download pipeline configuration for this version
            self.download_pipeline_configuration()
            
        except Exception as e:
            logger.error(f"Failed to setup DVC environment: {e}")
            raise
    
    def download_pipeline_configuration(self):
        """Download DVC pipeline configuration from S3."""
        try:
            config_prefix = f"pipeline-configs/{self.pipeline_version}/"
            
            # Download dvc.yaml, dvc.lock, params.yaml
            config_files = ["dvc.yaml", "params.yaml"]
            
            for config_file in config_files:
                try:
                    s3_key = f"{config_prefix}{config_file}"
                    local_path = self.workspace / config_file
                    
                    self.s3_client.download_file(
                        self.s3_bucket, s3_key, str(local_path)
                    )
                    logger.info(f"Downloaded {config_file} from S3")
                except Exception as e:
                    logger.warning(f"Could not download {config_file}: {e}")
                    # In production, you might want to fail here
                    # For now, we'll continue with local versions
            
            # Try to download existing dvc.lock if available
            try:
                s3_key = f"{config_prefix}dvc.lock"
                local_path = self.workspace / "dvc.lock"
                self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
                logger.info("Downloaded existing dvc.lock from S3")
            except:
                logger.info("No existing dvc.lock found in S3 - will create new one")
                
        except Exception as e:
            logger.error(f"Failed to download pipeline configuration: {e}")
            raise
    
    def upload_pipeline_configuration(self):
        """Upload current pipeline state to S3."""
        try:
            config_prefix = f"pipeline-configs/{self.pipeline_version}/"
            
            config_files = ["dvc.yaml", "dvc.lock", "params.yaml"]
            
            for config_file in config_files:
                local_path = self.workspace / config_file
                if local_path.exists():
                    s3_key = f"{config_prefix}{config_file}"
                    self.s3_client.upload_file(
                        str(local_path), self.s3_bucket, s3_key
                    )
                    logger.info(f"Uploaded {config_file} to S3")
                    
        except Exception as e:
            logger.error(f"Failed to upload pipeline configuration: {e}")
            raise
    
    def generate_training_fingerprint(self) -> Dict[str, Any]:
        """Generate comprehensive fingerprint for reproducibility."""
        try:
            # Get DVC lock content for exact artifact versions
            dvc_lock_path = self.workspace / "dvc.lock"
            dvc_lock_content = ""
            if dvc_lock_path.exists():
                with open(dvc_lock_path) as f:
                    dvc_lock_content = f.read()
            
            # Get parameters hash
            params_path = self.workspace / "params.yaml"
            params_content = ""
            if params_path.exists():
                with open(params_path) as f:
                    params_content = f.read()
            
            # Get data fingerprints from DVC
            data_fingerprints = self.get_dvc_data_fingerprints()
            
            fingerprint = {
                "training_id": f"train_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "pipeline_version": self.pipeline_version,
                "commit_sha": os.getenv("COMMIT_SHA", "unknown"),
                "build_id": os.getenv("BUILD_ID", "unknown"),
                "container_image": os.getenv("CONTAINER_IMAGE_TAG", "unknown"),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                
                # DVC-specific reproducibility data
                "dvc_lock_hash": hashlib.md5(dvc_lock_content.encode()).hexdigest(),
                "params_hash": hashlib.md5(params_content.encode()).hexdigest(),
                "data_fingerprints": data_fingerprints,
                "dvc_remote": f"s3://{self.s3_bucket}/dvc-storage",
                
                # Infrastructure metadata
                "kubernetes_namespace": os.getenv("POD_NAMESPACE", "unknown"),
                "kubernetes_pod": os.getenv("HOSTNAME", "unknown"),
                "aws_region": os.getenv("AWS_REGION", "unknown")
            }
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Failed to generate training fingerprint: {e}")
            raise
    
    def get_dvc_data_fingerprints(self) -> Dict[str, str]:
        """Get fingerprints of all data artifacts managed by DVC."""
        try:
            # Get DVC status to understand current state
            result = subprocess.run(
                ["dvc", "status", "--json"], 
                capture_output=True, text=True, cwd=self.workspace
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            else:
                return {"status": "up_to_date"}
                
        except Exception as e:
            logger.warning(f"Could not get DVC fingerprints: {e}")
            return {"error": str(e)}
    
    def run_training_with_dvc(self):
            """
            Execute DVC training pipeline, stream its output, and store metadata.
            This is a generator function that yields log lines.
            """
            try:
                logger.info("Starting DVC-managed training pipeline...")
                
                pre_fingerprint = self.generate_training_fingerprint()
                logger.info(f"Training ID: {pre_fingerprint['training_id']}")
                yield f"Training ID: {pre_fingerprint['training_id']}"
                
                logger.info("Executing DVC training pipeline...")
                yield "Executing DVC training pipeline..."

                # 1. Run DVC repro and stream/capture output
                dvc_output_lines = []
                process = subprocess.Popen(
                    ["dvc", "repro", "-f"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.workspace,
                    bufsize=1
                )
                for line in iter(process.stdout.readline, ''):
                    clean_line = line.strip()
                    dvc_output_lines.append(clean_line) # Capture line
                    yield clean_line                    # Stream line

                process.wait()
                if process.returncode != 0:
                    raise Exception(f"DVC pipeline failed with exit code {process.returncode}")

                yield "DVC pipeline completed successfully."
                
                # 2. Run DVC push and stream/capture output
                yield "Pushing artifacts to DVC remote..."
                push_process = subprocess.Popen(
                    ["dvc", "push"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.workspace,
                    bufsize=1
                )
                for line in iter(push_process.stdout.readline, ''):
                    yield line.strip()

                push_process.wait()
                if push_process.returncode != 0:
                    raise Exception(f"DVC push failed with exit code {push_process.returncode}")
                
                yield "Artifacts pushed to remote storage."

                # 3. Perform post-run tasks silently
                post_fingerprint = self.generate_training_fingerprint()
                
                training_output_str = "\n".join(dvc_output_lines)
                training_metadata = {
                    "pre_training": pre_fingerprint,
                    "post_training": post_fingerprint,
                    "training_output": training_output_str, # Use the captured output
                    "dvc_pipeline_status": self.get_dvc_data_fingerprints(),
                    "reproduction_commands": self.generate_reproduction_commands(post_fingerprint)
                }
                
                self.store_training_metadata(training_metadata)
                self.upload_pipeline_configuration()
                
                # As a generator, this function does not return a value.
                # Its job is to yield logs and perform actions.

            except Exception as e:
                error_msg = f"Training with DVC failed: {e}"
                logger.error(error_msg)
                yield error_msg
                raise
    
    def generate_reproduction_commands(self, fingerprint: Dict[str, Any]) -> Dict[str, str]:
        """Generate exact commands needed to reproduce this training."""
        training_id = fingerprint["training_id"]
        
        return {
            "description": "Commands to reproduce this exact training run",
            "setup": f"dvc remote add -d storage s3://{self.s3_bucket}/dvc-storage",
            "restore_pipeline": f"aws s3 cp s3://{self.s3_bucket}/experiments/{training_id}/dvc.lock ./",
            "restore_data": "dvc checkout",
            "pull_artifacts": "dvc pull", 
            "reproduce": "dvc repro -f",
            "docker_command": f"docker run -e DVC_S3_BUCKET={self.s3_bucket} -e PIPELINE_VERSION={self.pipeline_version} your-registry/training-service:latest"
        }
    
    def store_training_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store complete training metadata and exact pipeline files in S3."""
        try:
            training_id = metadata["post_training"]["training_id"]
        except KeyError as e:
            raise ValueError("metadata['post_training']['training_id'] is required") from e

        base_prefix = f"experiments/{training_id}/"

        # Store full metadata JSON
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=base_prefix + "complete_metadata.json",
            Body=json.dumps(metadata, indent=2, default=str),
            ContentType="application/json",
        )

        # Upload exact pipeline state for this run
        for name in ("dvc.yaml", "dvc.lock", "params.yaml"):
            file_path = self.workspace / name
            if file_path.exists():
                self.s3_client.upload_file(str(file_path), self.s3_bucket, base_prefix + name)
            else:
                logger.warning("Expected file missing during upload: %s", file_path)

        logger.info("Training metadata stored at s3://%s/%s", self.s3_bucket, base_prefix)


    def reproduce_training(self, training_id: str) -> Dict[str, Any]:
        """Reproduce a specific training run using stored DVC state."""
        try:
            logger.info("Reproducing training run: %s", training_id)
            base_prefix = f"experiments/{training_id}/"

            # Restore exact pipeline files
            for name in ("dvc.yaml", "params.yaml", "dvc.lock"):
                s3_key = base_prefix + name
                local_path = self.workspace / name
                self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            logger.info("Restored pipeline files (dvc.yaml, params.yaml, dvc.lock)")

            # Ensure DVC workspace exists (no-scm)
            if not (self.workspace / ".dvc").exists():
                subprocess.run(["dvc", "init", "--no-scm"], check=True, cwd=self.workspace)

            # Pull blobs first, then checkout workspace
            subprocess.run(["dvc", "pull"], check=True, cwd=self.workspace)
            logger.info("Pulled artifacts from remote")
            subprocess.run(["dvc", "checkout"], check=True, cwd=self.workspace)
            logger.info("Checked out exact data versions")

            # Reproduce the pipeline
            result = subprocess.run(
                ["dvc", "repro", "-f"],
                capture_output=True, text=True, cwd=self.workspace
            )
            if result.returncode != 0:
                raise RuntimeError(f"Reproduction failed: {result.stderr}")

            logger.info("Training reproduction completed successfully")

            return {
                "status": "reproduction_successful",
                "original_training_id": training_id,
                "reproduction_timestamp": datetime.datetime.utcnow().isoformat(),
                "output": result.stdout,
            }

        except Exception as e:
            logger.error("Failed to reproduce training: %s", e)
            raise


