variable "billing_account_id" {
  description = "GCP Billing Account ID"
  type        = string
}

variable "org_id" {
  description = "GCP Organization ID (optional, leave empty for no org)"
  type        = string
  default     = ""
}

variable "project_id" {
  description = "GCP Project ID for parameter golf"
  type        = string
  default     = "pgolf-lmxlab"
}

variable "region" {
  description = "GCP region for A100 availability"
  type        = string
  default     = "us-central1"  # Good A100 spot availability
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-tesla-a100"  # 40GB A100
}

variable "machine_type" {
  description = "GCP machine type"
  type        = string
  default     = "a2-highgpu-1g"  # 1x A100, 12 vCPUs, 85GB RAM
}

variable "spot" {
  description = "Use spot/preemptible instance for cost savings"
  type        = bool
  default     = true
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 200  # Need space for FineWeb data (~45GB) + tokenization
}
