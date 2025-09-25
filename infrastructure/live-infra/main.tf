### VPC Module ###
module "vpc" {
  source = "../modules/vpc"
  for_each            = var.vpc

  vpc_name             = each.value.vpc_name
  cidr                 = each.value.vpc_cidr_block
  availability_zones   = each.value.availability_zones
  public_subnet_cidrs  = each.value.public_sibnet_cidrs
  public_subnet_names  = each.value.public_subnet_names
  private_subnet_cidrs = concat(each.value.private_CICD_subnet_cidrs, each.value.private_app_subnet_cidrs)
  private_subnet_names = each.value.private_subnet_names
  enable_nat_gateway   = each.value.enable_nat_gateway
  enable_dns_hostnames = each.value.enable_dns_hostnames
  extra_private_routes = each.value.extra_private_routes
  tags                 = each.value.tags
  vpc_output_file      = each.value.vpc_output_file
} 

### S3 Module ###
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

