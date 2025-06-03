variable "aws_region" {
  description = "value"
}

variable "billing_mode" {
  description = "The billing mode for the DynamoDB table (e.g., PROVISIONED or PAY_PER_REQUEST)"
  type = string
  default = "PAY_PER_REQUEST"
}

variable "aws_dynamodb_table_name" {
  description = "The name of the DynamoDB table for state locking"
  type        = string
  default     = "terraform-state-locks" # Default name for the DynamoDB table
  
}

variable "aws_dynamodb_table_hash_key" {
  description = "The hash key for the DynamoDB table"
  type        = string
  default     = "LockID" # Default hash key for the DynamoDB table
}

variable "bucket_name" {
  description = "The name of the S3 bucket for storing Terraform state files"
  type        = string
  default     = "s3-for-terraform-state31" # Default name for the S3 bucket
  
}

variable "sse_algorithm" {
  description = "The server-side encryption algorithm for the S3 bucket (e.g., AES256 or aws:kms)"
  type        = string
  default     = "AES256" # Default server-side encryption algorithm for the S3 bucket
  
}

variable "versioning_configuration" {
  description = "The versioning configuration for the S3 bucket (e.g., Enabled or Suspended)"
  type        = string
  default     = "Enabled" # Default versioning configuration for the S3 bucket
}