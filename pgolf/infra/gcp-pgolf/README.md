# Parameter Golf GCP Infrastructure

Terraform config for a dedicated GCP project with A100 GPU for prototyping.

## Setup

```bash
# 1. Copy and edit variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your billing account ID

# 2. Initialize Terraform
terraform init

# 3. Review what will be created
terraform plan

# 4. Create the project + VM
terraform apply

# 5. SSH into the instance
gcloud compute ssh pgolf-a100 --project=pgolf-lmxlab --zone=us-central1-a

# 6. On the VM: clone repo and run
git clone https://github.com/<your-fork>/parameter-golf.git
cd parameter-golf
bash records/track_10min_16mb/2026-03-21_XSA_VR_ZLoss_sp2048/quick_prototype.sh baseline
```

## Cost Control

- **Spot instance**: ~$0.30-0.40/hr (vs $3.67 on-demand) = 90% savings
- **Stop when not in use**: `gcloud compute instances stop pgolf-a100 --project=pgolf-lmxlab --zone=us-central1-a`
- **Destroy when done**: `terraform destroy`
- **Budget alert**: Set up at https://console.cloud.google.com/billing/budgets

## Estimated Costs

| Task | Time | Cost (spot) |
|------|------|-------------|
| 20 experiments (600s each) | 5 hours | ~$1.50 |
| sp2048 tokenization | 1.5 hours | ~$0.50 |
| Full prototype session | 8 hours | ~$2.50 |
