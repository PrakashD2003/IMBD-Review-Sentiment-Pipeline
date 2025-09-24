locals {
  # Merge default tags with any user-provided tags
  default_tags = merge(
    {
      "Name"        = var.vpc_name,
      "Project"     = "IMDB-Sentiment-Pipeline",
      "ManagedBy"   = "Terraform"
    },
    var.tags
  )
}

# 1. Create the Virtual Private Cloud (VPC)
resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr_block
  enable_dns_hostnames = var.enable_dns_hostnames
  
  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-vpc"
  })
}

# 2. Create Public Subnets
resource "aws_subnet" "public" {
  for_each          = { for i, cidr in var.public_subnet_cidrs : i => cidr }
  vpc_id            = aws_vpc.this.id
  cidr_block        = each.value
  availability_zone = var.availability_zones[each.key]
  map_public_ip_on_launch = true

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-public-subnet-${each.key + 1}"
  })
}

# 3. Create Private Subnets
resource "aws_subnet" "private" {
  for_each          = { for i, cidr in var.private_subnet_cidrs : i => cidr }
  vpc_id            = aws_vpc.this.id
  cidr_block        = each.value
  availability_zone = var.availability_zones[each.key]

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-private-subnet-${each.key + 1}"
  })
}

# 4. Create Internet Gateway for the Public Subnets
resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-igw"
  })
}

# 5. Create NAT Gateway for the Private Subnets (Conditional)
resource "aws_eip" "nat" {
  count = var.enable_nat_gateway ? 1 : 0
  domain   = "vpc"

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-nat-eip"
  })
}

resource "aws_nat_gateway" "this" {
  count         = var.enable_nat_gateway ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = values(aws_subnet.public)[0].id # Place NAT in the first public subnet

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-nat-gw"
  })

  depends_on = [aws_internet_gateway.this]
}

# 6. Configure Routing
# Public Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-public-rt"
  })
}

# Private Route Table (with conditional route to NAT Gateway)
resource "aws_route_table" "private" {
  count  = var.enable_nat_gateway ? 1 : 0
  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this[0].id
  }

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-private-rt"
  })
}

# 7. Associate Route Tables with Subnets
resource "aws_route_table_association" "public" {
  for_each       = aws_subnet.public
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = var.enable_nat_gateway ? length(var.private_subnet_cidrs) : 0
  subnet_id      = values(aws_subnet.private)[count.index].id
  route_table_id = aws_route_table.private[0].id
}