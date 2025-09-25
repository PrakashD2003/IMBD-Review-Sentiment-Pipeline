variable "vpc_name" {
  description = "The name for the VPC and its associated resources"
  type        = string
  default     = "imdb-app-vpc"
}

variable "vpc_cidr_block" {
  description = "The CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "A list of Availability Zones to deploy subnets into"
  type        = list(string)
  default     = ["ap-south-1a", "ap-south-1b"]
}

variable "public_subnet_cidrs" {
  description = "A list of CIDR blocks for the public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "public_subnet_names" {
  description = "A list of descriptive names for the public subnets."
  type        = list(string)
  default     = []
}

variable "private_subnet_cidrs" {
  description = "A list of CIDR blocks for the private subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

variable "private_subnet_names" {
  description = "A list of descriptive names for the private subnets."
  type        = list(string)
  default     = []
}

variable "enable_nat_gateway" {
  description = "Set to true to create a NAT Gateway for outbound traffic from private subnets"
  type        = bool
  default     = true
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames in the VPC"
  type        = bool
  default     = true
}

variable "extra_private_routes"{
  description = "A list of extra, custom routing rules to add to the privarte route tables"
  type = list(object({
    cidr_block                  = string
    # --- CHOOSE ONE TARGET ---
    vpc_peering_connection_id   = optional(string)
    transit_gateway_id          = optional(string)
    nat_gateway_id              = optional(string)
    network_interface_id        = optional(string)
  }))
  default = []
}

variable "tags" {
  description = "A map of tags to assign to all resources"
  type        = map(string)
  default     = {}
}

variable "vpc_output_file" {
  description = "Path to write out VPC metadata, following the pattern from the S3 module"
  type        = string
  default     = "./infrastructure/outputs/vpc_metadata.json"
}