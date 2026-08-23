# ---------------------------------------------------------------------------
# CloudWatch alarms — kept inside the free 10-alarm tier.
# Notifications go to alerts/webhook.py's Slack/Discord webhook via a
# tiny forwarder Lambda, not SNS+email — cheaper and reuses the alert
# channel the app already posts to instead of adding a second one.
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.project_name}-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods   = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 3
  dimensions = {
    FunctionName = aws_lambda_function.ser_inference.function_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "lambda_throttles" {
  alarm_name          = "${var.project_name}-lambda-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods   = 1
  metric_name         = "Throttles"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  dimensions = {
    FunctionName = aws_lambda_function.ser_inference.function_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  # Throttling means reserved_concurrency=2 is actually being hit —
  # i.e. the budget backstop is engaging. Worth knowing immediately,
  # not just inferring it from user complaints.
}

resource "aws_cloudwatch_metric_alarm" "dynamodb_throttles" {
  alarm_name          = "${var.project_name}-dynamodb-throttles"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods   = 1
  metric_name         = "ThrottledRequests"
  namespace           = "AWS/DynamoDB"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# SNS exists only as the mechanical trigger for the webhook-forwarder
# Lambda below — not used for direct email/SMS, so it stays inside the
# always-free SNS tier (first 1M publishes/mo free).
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "webhook_forwarder" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.webhook_forwarder.arn
}

resource "aws_lambda_permission" "allow_sns" {
  statement_id  = "AllowSNSInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.webhook_forwarder.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.alerts.arn
}

resource "aws_iam_role" "webhook_forwarder_role" {
  name = "${var.project_name}-webhook-forwarder"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "webhook_forwarder_logs" {
  role       = aws_iam_role.webhook_forwarder_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Tiny zip-packaged Lambda (pure Python, no Docker) that takes the SNS
# alarm payload and posts to the same webhook URL alerts/webhook.py
# already uses. Source lives at ../lambda/webhook_forwarder/handler.py
# — see that file for the ~15-line implementation.
data "archive_file" "webhook_forwarder_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda/webhook_forwarder"
  output_path = "${path.module}/../lambda/webhook_forwarder.zip"
}

resource "aws_lambda_function" "webhook_forwarder" {
  function_name    = "${var.project_name}-webhook-forwarder"
  role             = aws_iam_role.webhook_forwarder_role.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.webhook_forwarder_zip.output_path
  source_code_hash = data.archive_file.webhook_forwarder_zip.output_base64sha256
  timeout          = 10
  environment {
    variables = { ALERT_WEBHOOK_URL = var.alert_webhook_url }
  }
}

# ---------------------------------------------------------------------------
# Budget guardrail
# ---------------------------------------------------------------------------
resource "aws_budgets_budget" "monthly_cap" {
  name         = "${var.project_name}-monthly-budget"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }
  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
