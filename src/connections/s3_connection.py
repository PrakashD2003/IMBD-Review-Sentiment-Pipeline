import os
import boto3
import logging 
import dask.dataframe as ddf
import pandas as pd
from io import StringIO
from typing import Optional, Union, Optional

from mypy_boto3_s3 import S3Client
from logger import configure_logger
from exception import DetailedException
from constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

class S3Connection:
    """
    Wrapper around a boto3 S3 client, providing convenience methods for
    fetching CSV data directly into pandas DataFrames.

    Uses a class‐level singleton client so that multiple instances reuse
    the same boto3 client.
    """

    s3_client: Optional[S3Client] = None

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the S3Connection, configuring AWS credentials and region
        from environment variables (falling back to boto3 defaults).

        :param logger: Optional Logger; if not provided, one is configured.
        :raises DetailedException: On any failure configuring the client.
        """
        try:
            self.logger = logger or configure_logger(logger_name=__name__, 
                                                level="DEBUG",
                                                to_console=True,
                                                to_file=True,
                                                log_file_name=__name__)
            if S3Connection.s3_client == None:
                self.logger.debug("Getting AWS Credentials from Environment Variable(if present)...")
                # build kwargs only if creds exist
                creds = {}
                aws_access_key_id = os.getenv(AWS_ACCESS_KEY_ID)
                aws_secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY)
                aws_region = os.getenv(AWS_REGION, "ap-south-1" ) 
                if aws_access_key_id and aws_secret_access_key:
                    self.logger.debug("Found AWS env credentials, using them.")
                    creds["aws_access_key_id"] = aws_access_key_id
                    creds["aws_secret_access_key"] = aws_secret_access_key
                else:
                    self.logger.debug("No AWS env creds, boto3 will use default chain.")
                
                creds["region_name"] = aws_region

                self.logger.debug("Setting up AWS S3 client...")
                S3Connection.s3_client = boto3.client("s3", **creds)
                self.logger.info("AWS resource and client configured Successfully.")

            self.s3_client = S3Connection.s3_client
        except Exception as e:
            raise DetailedException(exc=e, logger=self.logger) from e   
    
    def load_csv_from_s3(self, bucket_name:str, file_key:str, as_ddf:bool = True)-> Union[pd.DataFrame, ddf.DataFrame]:
        """
        Fetches a CSV file from S3 and loads it into either a Dask DataFrame or a pandas DataFrame.

        :param bucket_name: Name of the S3 bucket.
        :param file_key: The key (path) to the CSV file within the bucket.
        :param as_ddf: If True, return a Dask DataFrame; otherwise return a pandas DataFrame.
                    Defaults to True.
        :return: A Dask DataFrame (if as_ddf=True) or a pandas DataFrame (if as_ddf=False).
        :raises DetailedException: On any error during S3 access or CSV parsing.
        """
        try:
            self.logger.info("Entered 'load_csv_from_s3' method of 'S3Connection' class.")
            self.logger.debug(f"Retrieving specified S3 file: '{file_key}' from '{bucket_name}'...")

            obj = self.s3_client.get_object(Bucket=bucket_name, Key=file_key)
            self.logger.info(f"file object '{file_key}' retrieved successfully from bucket '{bucket_name}'.")

            body_bytes = obj["Body"].read()
            body_str = body_bytes.decode("utf-8")
            buffer = StringIO(body_str)
            
            if as_ddf:
                self.logger.debug("Converting fetched s3 object into dask dataframe...")
                df = ddf.read_csv(buffer, assume_missing=True)
                self.logger.info("Successfully concerted s3 object into dask dataframe.")
            else:
                self.logger.debug("Converting fetched s3 object into dataframe...")
                df = pd.read_csv(buffer)
                self.logger.info("Successfully concerted s3 object into dataframe.")
            self.logger.debug("Exited 'load_csv_from_s3' method of S3Connection' class.")
            return df
        except Exception as e:
            raise DetailedException(exc=e, logger=self.logger)
    
  



        
        