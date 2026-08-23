resource "aws_s3_bucket" "ser_data" {
  bucket = "${var.project_name}-data-${var.account_id}"
}

resource "aws_s3_bucket_lifecycle_configuration" "ser_data_lifecycle" {
  bucket = aws_s3_bucket.ser_data.id

  rule {
    id     = "expire-old-audio-samples"
    status = "Enabled"
    filter { prefix = "audio-samples/" }
    # Audio is only useful until it's been labeled and folded into a
    # Kaggle training batch. 180 days is a generous backstop against
    # unbounded growth from a budget standpoint — tune to taste, but
    # don't leave this unset.
    expiration { days = 180 }
  }

  rule {
    id     = "expire-old-feature-logs"
    status = "Enabled"
    filter { prefix = "feature-logs/" }
    expiration { days = 90 }
  }
}

resource "aws_s3_bucket_public_access_block" "ser_data_block_public" {
  bucket                  = aws_s3_bucket.ser_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "data_bucket_name" {
  value = aws_s3_bucket.ser_data.bucket
}
