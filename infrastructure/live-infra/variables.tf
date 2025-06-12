variable "aws_region" {
  description = "The AWS region to deploy the infrastructure"
  type        = string
  default     = "ap-south-1"
}



### S3 Variables ###
variable "s3_buckets" {
  description = "Map of configuration for each S3 bucket"
  type = map(object({
    bucket_name       = string
    sse_algorithm     = string
    force_destroy     = bool
    enable_versioning = bool
    enable_sse        = bool
    s3_tags           = map(string)
    s3_output_file    = string
  }))
}
