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
  default     = "nvidia-l4"  # L4 24GB — available NOW, $0.70/hr, for validation only
}

variable "machine_type" {
  description = "GCP machine type"
  type        = string
  default     = "g2-standard-8"  # 1x L4, 8 vCPUs, 32GB RAM, ~$0.70/hr
}

variable "spot" {
  description = "Use spot/preemptible instance for cost savings (risk: interruption every 1-4 hrs)"
  type        = bool
  default     = false  # On-demand is safer for 15-min experiments (~$3.67/hr vs $0.35)
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 100  # Validation only — just sp1024 data (~20GB) + code
}
