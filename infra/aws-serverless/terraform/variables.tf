variable "aws_region" {
  default = "us-east-1"
}

variable "account_id" {
  description = "395249043027 — shared across all 3 projects"
  type        = string
  default     = "395249043027"
}

variable "project_name" {
  default = "ser-inference"
}

variable "lambda_memory_mb" {
  description = "Torch + WavLM cold start needs headroom; tune after measuring actual cold start."
  default     = 4096
}

variable "lambda_timeout_s" {
  default = 60
}

variable "reserved_concurrency" {
  description = "HARD budget backstop. This is the one public, user-facing endpoint across all 3 projects — cap it low."
  default     = 2
}

variable "alert_webhook_url" {
  description = "Slack/Discord webhook URL, same one alerts/webhook.py already posts to. Passed as a Lambda env var / SSM param, not hardcoded."
  type        = string
  sensitive   = true
}

variable "admin_ui_secret" {
  description = "Shared secret for the solo-labeler /admin page. Store as SSM SecureString, not plaintext tfvars."
  type        = string
  sensitive   = true
}

variable "budget_limit_usd" {
  default = 4  # alarm at $4, not the full $5 ceiling
}

variable "budget_alert_email" {
  type = string
}

variable "ecr_image_retention_count" {
  description = "Keep only the last N images — weekly retrains otherwise accumulate ECR storage cost."
  default     = 3
}
