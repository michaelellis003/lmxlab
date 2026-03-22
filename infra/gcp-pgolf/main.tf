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

# --- Firewall: allow SSH ---
resource "google_compute_firewall" "allow_ssh" {
  name    = "pgolf-allow-ssh"
  network = "default"
  project = google_project.pgolf.project_id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["pgolf-gpu"]

  depends_on = [google_project_service.compute]
}

# --- GPU VM ---
resource "google_compute_instance" "pgolf_gpu" {
  name         = "pgolf-a100"
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

  guest_accelerator {
    type  = var.gpu_type
    count = 1
  }

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/pytorch-latest-gpu"
      size  = var.disk_size_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
    access_config {} # Public IP for SSH
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    echo "=== pgolf GPU instance starting ==="
    # Install sentencepiece for tokenizer training
    pip install -q sentencepiece 2>/dev/null || true
    echo "Startup complete at $(date)"
  EOT

  depends_on = [google_project_service.compute]

  lifecycle {
    ignore_changes = [metadata_startup_script]
  }
}

output "instance_ip" {
  value       = google_compute_instance.pgolf_gpu.network_interface[0].access_config[0].nat_ip
  description = "Public IP of the GPU instance"
}

output "ssh_command" {
  value       = "gcloud compute ssh pgolf-a100 --project=${var.project_id} --zone=${var.zone}"
  description = "SSH command to connect"
}

output "estimated_cost" {
  value       = var.spot ? "~$0.30-0.40/hr (spot A100 40GB)" : "~$3.67/hr (on-demand A100 40GB)"
  description = "Estimated hourly cost"
}
