output "bucket_name" {
  description = "Name of the bucket"
  value = aws_s3_bucket.this.bucket
}

output "bucket_id" {
  description = "ID of the created S3 bucket"
  value       = aws_s3_bucket.this.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.this.arn
}

output "encryption_enabled" {
  description = "Whether SSE configuration is applied"
  value       = length(aws_s3_bucket_server_side_encryption_configuration.this) > 0
}

output "versioning_enabled" {
  description = "Whether versioning is enabled"
  value       = length(aws_s3_bucket_versioning.this) > 0
}

resource "local_file" "s3_metadata" {
  content = jsonencode({
    bucket_name = aws_s3_bucket.this.bucket
    bucket_id = aws_s3_bucket.this.id
    bucket_arn = aws_s3_bucket.this.arn
    enable_versioning = length(aws_s3_bucket_server_side_encryption_configuration.this) > 0
    versioning_enabled = length(aws_s3_bucket_versioning.this) > 0
})
  
  filename = var.s3_output_file
}
