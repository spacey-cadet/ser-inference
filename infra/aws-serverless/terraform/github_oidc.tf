# ---------------------------------------------------------------------------
# GitHub OIDC — build once, reuse across all 3 projects (Project 1 needs
# this exact same setup and hasn't built it yet — copy this file there
# too instead of duplicating a second role).
# ---------------------------------------------------------------------------

data "aws_iam_openid_connect_provider" "github" {
  # If this doesn't exist yet in the account:
  # resource "aws_iam_openid_connect_provider" "github" {
  #   url             = "https://token.actions.githubusercontent.com"
  #   client_id_list  = ["sts.amazonaws.com"]
  #   thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
  # }
  url = "https://token.actions.githubusercontent.com"
}

variable "github_repo" {
  description = "org/repo, e.g. spacey-cadet/ser-inference"
  type        = string
}

resource "aws_iam_role" "github_actions_deploy" {
  name = "github-actions-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = data.aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "github_actions_deploy_policy" {
  name = "github-actions-deploy-policy"
  role = aws_iam_role.github_actions_deploy.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRPushPull"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = "*"
      },
      {
        Sid      = "LambdaDeploy"
        Effect   = "Allow"
        Action   = ["lambda:UpdateFunctionCode", "lambda:GetFunction"]
        Resource = "arn:aws:lambda:*:*:function:ser-inference"
      },
      {
        Sid    = "ReadReviewQueueAndWatermark"
        Effect = "Allow"
        Action = ["dynamodb:Query", "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Scan"]
        Resource = [
          "arn:aws:dynamodb:*:*:table/ser-inference-review-queue",
          "arn:aws:dynamodb:*:*:table/ser-inference-review-queue/index/*",
          "arn:aws:dynamodb:*:*:table/ser-inference-retrain-state",
        ]
      },
      {
        Sid      = "ReadAudioForBatchAssembly"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "arn:aws:s3:::ser-inference-data-*/audio-samples/*"
      },
    ]
  })
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions_deploy.arn
  description = "Set this as the AWS_DEPLOY_ROLE_ARN GitHub secret"
}
