module "s3" {
  source            = "../modules/s3"
  bucket_name       = var.bucket_name
  enable_sse        = var.enable_sse
  sse_algorithm     = var.sse_algorithm
  enable_versioning = var.enable_versioning
  force_destroy     = var.force_destroy
  s3_tags           = var.s3_tags
}