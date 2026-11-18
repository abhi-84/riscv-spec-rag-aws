terraform {
  required_version = ">= 1.15.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.27" }
  }
  backend "s3" {
    bucket       = "riscv-spec-tfstate-51e315"  
    key          = "riscv-spec-rag/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = { project = "riscv-spec-rag", managed_by = "terraform" }
  }
}

data "aws_caller_identity" "current" {}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}
