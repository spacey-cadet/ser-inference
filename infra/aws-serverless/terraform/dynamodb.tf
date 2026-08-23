# ---------------------------------------------------------------------------
# Review queue — human-in-the-loop labeling queue
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "review_queue" {
  name         = "${var.project_name}-review-queue"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "request_id"

  attribute {
    name = "request_id"
    type = "S"
  }

  # Needed so the admin UI and the retrain-policy check can list
  # pending / labeled items without a full table scan.
  attribute {
    name = "status"
    type = "S"
  }
  attribute {
    name = "created_at"
    type = "S"
  }

  global_secondary_index {
    name            = "status-created_at-index"
    hash_key        = "status"
    range_key       = "created_at"
    projection_type = "ALL"
  }

  # Optional: auto-expire un-labeled low-value items after N days so the
  # queue doesn't grow forever if nobody labels them. Set expires_at
  # only on items you're OK losing if unlabeled.
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# ---------------------------------------------------------------------------
# Session state — rolling emotion state per chat session
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "session_state" {
  name         = "${var.project_name}-session-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"

  attribute {
    name = "session_id"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# ---------------------------------------------------------------------------
# Retrain watermark — single-item table tracking "last successful
# training run" state for the policy-check script. A DynamoDB table is
# overkill for one item, but keeps this in the same
# console/IAM-permission surface as everything else rather than adding
# an SSM parameter with a different access pattern.
# ---------------------------------------------------------------------------
resource "aws_dynamodb_table" "retrain_state" {
  name         = "${var.project_name}-retrain-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "state_key"

  attribute {
    name = "state_key"
    type = "S"
  }
}
