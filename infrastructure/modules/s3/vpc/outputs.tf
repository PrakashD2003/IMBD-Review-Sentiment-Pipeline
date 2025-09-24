output "vpc_id" {
  description = "The ID of the created VPC"
  value       = aws_vpc.this.id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = aws_vpc.this.cidr_block
}

output "public_subnet_ids" {
  description = "A list of IDs for the public subnets"
  value       = [for s in aws_subnet.public : s.id]
}

output "private_subnet_ids" {
  description = "A list of IDs for the private subnets"
  value       = [for s in aws_subnet.private : s.id]
}

output "nat_gateway_public_ip" {
  description = "The public IP address of the NAT Gateway"
  value       = var.enable_nat_gateway ? aws_eip.nat[0].public_ip : null
}

# Replicating the pattern from the s3 module to output metadata to a file
resource "local_file" "vpc_metadata" {
  content = jsonencode({
    vpc_id             = aws_vpc.this.id
    vpc_cidr_block     = aws_vpc.this.cidr_block
    public_subnet_ids  = [for s in aws_subnet.public : s.id]
    private_subnet_ids = [for s in aws_subnet.private : s.id]
  })

  filename = var.vpc_output_file
}