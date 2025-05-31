locals {
  default_tags = merge(
    {
        Name = var.bucket_name
        Environment = "Dev"
    },
    var.s3_tags

  )
}

resource "aws_s3_bucket" "this" {
    bucket = var.bucket_name
    force_destroy = var.force_destroy
    tags = local.default_tags
}

resource "aws_s3_bucket_versioning" "this" {
  count = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.this

  versioning_configuration {
    status = var.enable_versioning ? "Enable" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  count = var.enable_sse ? 1 : 0
  bucket = aws_s3_bucket.this

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = var.sse_algorithm
    }
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  bucket = aws_s3_bucket.this
  
  block_public_acls = true
  block_public_policy = true
  ignore_public_acls = true
  restrict_public_buckets = true
}
