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
    Name = var.public_subnet_names[each.key]
  })
}

# 3. Create Private Subnets
resource "aws_subnet" "private" {
  for_each          = { for i, cidr in var.private_subnet_cidrs : i => cidr }
  vpc_id            = aws_vpc.this.id
  cidr_block        = each.value
  availability_zone = var.availability_zones[each.key]

  tags = merge(local.default_tags, {
    Name = var.private_subnet_names[each.key]
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
  count  = var.enable_nat_gateway ? length(var.public_subnet_cidrs) : 0
  domain = "vpc"

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-nat-eip-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "this" {
  # Create one NAT Gateway for each public subnet if NAT is enabled
  count         = var.enable_nat_gateway ? length(var.public_subnet_cidrs) : 0
  allocation_id = aws_eip.nat[count.index].id
  # Place each NAT Gateway in its corresponding public subnet
  subnet_id     = values(aws_subnet.public)[count.index].id

  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-nat-gw-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.this]
}

# 6. Configure Routing
# Public Route Table(one for all public subnets)
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

# Private Route Table (with conditional route to NAT Gateway)(one for EACH Availability Zone)
resource "aws_route_table" "private" {
  count  = var.enable_nat_gateway ? length(var.availability_zones): 0
  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = values(aws_nat_gateway.this)[count.index].id
  }

  # This will loop through the `extra_private_routes` variable
  # and create a new route for each item in the list.
  dynamic "route" {
    for_each = var.extra_private_routes
    content{
      cidr_block = route.value.cidr_block
      # The `lookup` function safely gets a value if it exists, otherwise returns null.
      # This allows you to specify only one target per route.
      vpc_peering_connection_id = lookup(route.value, "vpc_peering_connection_id", null)
      transit_gateway_id        = lookup(route.value, "transit_gateway_id", null)
      nat_gateway_id            = lookup(route.value, "nat_gateway_id", null)
      network_interface_id      = lookup(route.value, "network_interface_id", null)
    }
  }
  tags = merge(local.default_tags, {
    Name = "${var.vpc_name}-private-rt-${var.availability_zones[count.index]}"
  })
}

# 7. Associate Route Tables with Subnets

# Associate the single public route table with all public subnets
resource "aws_route_table_association" "public" {
  for_each       = aws_subnet.public
  subnet_id      = each.value.id
  route_table_id = aws_route_table.public.id
}

# Associate private subnets with the correct private route table for their AZ
resource "aws_route_table_association" "private" {
  for_each       = aws_subnet.private
  subnet_id      = each.value.id
  route_table_id = aws_route_table.private[index(var.availability_zones, each.value.availability_zone)].id
}
