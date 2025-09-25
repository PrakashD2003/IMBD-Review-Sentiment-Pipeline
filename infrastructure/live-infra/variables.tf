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

### VPC Variables ###
variable "vpc" {
  description = "Map of configuration for the VPC"
  type = map(object({
    vpc_name = string
    vpc_cidr_block = string
    availability_zones = list(string)
    public_subnet_cidrs = list(string)
    public_subnet_names = list(string)
    private_CICD_subnet_cidrs = list(string)
    private_app_subnet_cidrs = list(string)
    private_subnet_names = list(string)
    enable_nat_gateway = bool
    enable_dns_hostnames = bool
    extra_private_routes = optional(list(object({
      cidr_block                  = string
      # --- CHOOSE ONE TARGET ---
      vpc_peering_connection_id   = optional(string)
      transit_gateway_id          = optional(string)
      nat_gateway_id              = optional(string)
      network_interface_id        = optional(string)
    })))
    tags = map(string)
    vpc_output_file = string
  }))
}