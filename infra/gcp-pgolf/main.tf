terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# --- Project ---
resource "google_project" "pgolf" {
  name            = "Parameter Golf"
  project_id      = var.project_id
  billing_account = var.billing_account_id
  org_id          = var.org_id != "" ? var.org_id : null

  lifecycle {
    prevent_destroy = true
  }
}

# --- Enable required APIs ---
resource "google_project_service" "compute" {
  project = google_project.pgolf.project_id
  service = "compute.googleapis.com"
}

resource "google_project_service" "iap" {
  project = google_project.pgolf.project_id
  service = "iap.googleapis.com"
}

# --- VPC Network ---
resource "google_compute_network" "pgolf_net" {
  name                    = "pgolf-network"
  project                 = google_project.pgolf.project_id
  auto_create_subnetworks = true  # Auto-creates subnets in all regions

  depends_on = [google_project_service.compute]
}

# --- Firewall: allow SSH via IAP tunnel ---
resource "google_compute_firewall" "allow_iap_ssh" {
  name    = "pgolf-allow-iap-ssh"
  network = google_compute_network.pgolf_net.name
  project = google_project.pgolf.project_id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]  # Google IAP IP range
  target_tags   = ["pgolf-gpu"]
}

# --- Cloud Router + NAT (for outbound internet without external IP) ---
resource "google_compute_router" "pgolf_router" {
  name    = "pgolf-router"
  network = google_compute_network.pgolf_net.name
  project = google_project.pgolf.project_id
  region  = var.region
}

resource "google_compute_router_nat" "pgolf_nat" {
  name                               = "pgolf-nat"
  router                             = google_compute_router.pgolf_router.name
  project                            = google_project.pgolf.project_id
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# --- GPU VM ---
# g2 (L4) has GPU built into machine type — no guest_accelerator needed
# a2 (A100) / a3 (H100) need guest_accelerator
locals {
  needs_guest_accelerator = startswith(var.machine_type, "a2-") || startswith(var.machine_type, "a3-")
  vm_name                 = "pgolf-gpu"
}

resource "google_compute_instance" "pgolf_gpu" {
  name         = local.vm_name
  machine_type = var.machine_type
  zone         = var.zone
  project      = google_project.pgolf.project_id

  tags = ["pgolf-gpu"]

  scheduling {
    preemptible                 = var.spot
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
    provisioning_model          = var.spot ? "SPOT" : "STANDARD"
    instance_termination_action = var.spot ? "STOP" : null
  }

  dynamic "guest_accelerator" {
    for_each = local.needs_guest_accelerator ? [1] : []
    content {
      type  = var.gpu_type
      count = 1
    }
  }

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
      size  = var.disk_size_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network    = google_compute_network.pgolf_net.name
    subnetwork = "projects/${google_project.pgolf.project_id}/regions/${var.region}/subnetworks/pgolf-network"
    # No access_config = no external IP (use IAP tunnel for SSH instead)
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    echo "=== pgolf GPU instance starting ==="
    pip install -q sentencepiece 2>/dev/null || true
    echo "Startup complete at $(date)"
  EOT

  depends_on = [google_project_service.compute]

  lifecycle {
    ignore_changes = [metadata_startup_script]
  }
}

output "ssh_command" {
  value       = "gcloud compute ssh ${local.vm_name} --project=${var.project_id} --zone=${var.zone} --tunnel-through-iap"
  description = "SSH command to connect via IAP tunnel (no external IP)"
}

output "delete_command" {
  value       = "gcloud compute instances delete ${local.vm_name} --project=${var.project_id} --zone=${var.zone} --quiet"
  description = "DELETE VM to stop ALL charges (including disk)"
}

output "estimated_cost" {
  value       = var.spot ? "~$0.30-0.40/hr (spot A100 40GB)" : "~$3.67/hr (on-demand A100 40GB)"
  description = "Estimated hourly cost"
}
