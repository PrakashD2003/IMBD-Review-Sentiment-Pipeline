module "s3" {
  source   = "../modules/s3"
  for_each = var.s3_buckets

  bucket_name       = each.value.bucket_name
  enable_sse        = each.value.enable_sse
  sse_algorithm     = each.value.sse_algorithm
  enable_versioning = each.value.enable_versioning
  force_destroy     = each.value.force_destroy
  s3_tags           = each.value.s3_tags
}

