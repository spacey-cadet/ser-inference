terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# ECR
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "ser_inference" {
  name                 = "${var.project_name}"
  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_lifecycle_policy" "keep_recent_only" {
  repository = aws_ecr_repository.ser_inference.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep only last N images — weekly retrains push new images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = var.ecr_image_retention_count
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# IAM — Lambda execution role
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "basic_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_data_access" {
  name = "${var.project_name}-lambda-data-access"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AudioAndFeatureLogBucket"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject"]
        Resource = [
          "${aws_s3_bucket.ser_data.arn}/audio-samples/*",
          "${aws_s3_bucket.ser_data.arn}/feature-logs/*",
        ]
      },
      {
        Sid    = "ReviewQueueTable"
        Effect = "Allow"
        Action = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query", "dynamodb:UpdateItem", "dynamodb:Scan"]
        Resource = [
          aws_dynamodb_table.review_queue.arn,
          "${aws_dynamodb_table.review_queue.arn}/index/*",
        ]
      },
      {
        Sid      = "SessionStateTable"
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:UpdateItem"]
        Resource = aws_dynamodb_table.session_state.arn
      },
      {
        Sid      = "PresignAudioForAdminUI"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.ser_data.arn}/audio-samples/*"
      },
      {
        Sid      = "AdminSecret"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = aws_ssm_parameter.admin_ui_secret.arn
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------
resource "aws_lambda_function" "ser_inference" {
  function_name = var.project_name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.ser_inference.repository_url}:latest"

  memory_size                   = var.lambda_memory_mb
  timeout                       = var.lambda_timeout_s
  reserved_concurrent_executions = var.reserved_concurrency

  environment {
    variables = {
      STORAGE_BACKEND      = "aws"
      AUDIO_BUCKET         = aws_s3_bucket.ser_data.bucket
      FEATURE_LOG_PREFIX   = "feature-logs/"
      AUDIO_SAMPLE_PREFIX  = "audio-samples/"
      REVIEW_QUEUE_TABLE   = aws_dynamodb_table.review_queue.name
      SESSION_STATE_TABLE  = aws_dynamodb_table.session_state.name
      ALERT_WEBHOOK_URL    = var.alert_webhook_url
      ADMIN_UI_SECRET_PARAM = aws_ssm_parameter.admin_ui_secret.name
    }
  }
}

resource "aws_lambda_function_url" "ser_inference_url" {
  function_name      = aws_lambda_function.ser_inference.function_name
  authorization_type = "NONE"
}

resource "aws_ssm_parameter" "admin_ui_secret" {
  name  = "/${var.project_name}/admin-ui-secret"
  type  = "SecureString"
  value = var.admin_ui_secret
}

output "function_url" {
  value = aws_lambda_function_url.ser_inference_url.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.ser_inference.repository_url
}
