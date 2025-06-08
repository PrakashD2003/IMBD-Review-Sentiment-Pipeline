terraform {
  required_version = ">= 1.11.4"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.33.0"
    }
  }
  backend "s3" {
    bucket         = "s3-for-terraform-state31" # Name of the S3 bucket where the Terraform state file will be stored (must be lowercase & globally unique)
    key            = "terraform.tfstate"        # The path and name of the state file inside the S3 bucket.
    region         = "ap-south-1"               # The AWS region where the S3 bucket and DynamoDB table are located.
    dynamodb_table = "terraform-state-locks"    # Name of the DynamoDB table used for state locking.
    encrypt        = true                       # Ensures that the state file is encrypted using AES-256 encryption.
  }
}