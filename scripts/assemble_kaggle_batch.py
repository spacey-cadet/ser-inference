#!/usr/bin/env python
"""
scripts/assemble_kaggle_batch.py

"Push mode" bridge between AWS and Kaggle. Runs in GitHub Actions,
where AWS credentials already live (via OIDC) — deliberately does NOT
give the Kaggle kernel direct AWS access, since handing a third-party
hosted-notebook service live credentials to your S3/DynamoDB is a
bigger blast radius than it needs to be.

Steps:
  1. Pull labeled items from DynamoDB (status=labeled, created_at after
     the last training watermark).
  2. Download the corresponding audio from S3.
  3. Package into a local directory: audio files + a manifest.csv
     mapping filename -> label -> request_id.
  4. Push as a new VERSION of a pinned Kaggle Dataset via the Kaggle
     API, so the training kernel always consumes a specific, citable
     dataset version rather than "whatever's newest."

Requires the `kaggle` CLI/API package and KAGGLE_USERNAME / KAGGLE_KEY
env vars (from GitHub Secrets, not Kaggle Secrets — this runs in GH
Actions, not on Kaggle).

Usage:
    python scripts/assemble_kaggle_batch.py \
        --review-queue-table ser-inference-review-queue \
        --audio-bucket ser-inference-data-395249043027 \
        --since <last-watermark-iso> \
        --kaggle-dataset your-username/ser-retrain-batch \
        --workdir /tmp/kaggle-batch
"""
import argparse
import csv
import json
import os
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review-queue-table", required=True)
    p.add_argument("--audio-bucket", required=True)
    p.add_argument("--since", required=True, help="ISO timestamp watermark")
    p.add_argument("--kaggle-dataset", required=True, help="username/dataset-slug")
    p.add_argument("--workdir", default="/tmp/kaggle-batch")
    p.add_argument("--out-manifest", default="batch_manifest.json",
                    help="Written locally too, so the GH Actions job can read back "
                         "which request_ids were included (needed to mark them "
                         "'deployed' after a successful promotion).")
    args = p.parse_args()

    workdir = Path(args.workdir)
    audio_dir = workdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dynamodb = boto3.resource("dynamodb")
    s3 = boto3.client("s3")
    table = dynamodb.Table(args.review_queue_table)

    resp = table.query(
        IndexName="status-created_at-index",
        KeyConditionExpression=Key("status").eq("labeled") & Key("created_at").gt(args.since),
    )
    items = resp["Items"]
    if not items:
        print("No new labeled items since watermark — nothing to assemble.")
        Path(args.out_manifest).write_text(json.dumps({"request_ids": []}))
        return

    manifest_rows = []
    request_ids = []
    for item in items:
        request_id = item["request_id"]
        s3_key = item["s3_audio_key"]
        label = item["label"]
        local_filename = f"{request_id}.wav"
        s3.download_file(args.audio_bucket, s3_key, str(audio_dir / local_filename))
        manifest_rows.append({"filename": local_filename, "label": label, "request_id": request_id})
        request_ids.append(request_id)

    with open(workdir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "request_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    dataset_metadata = {
        "title": "SER retrain batch",
        "id": args.kaggle_dataset,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (workdir / "dataset-metadata.json").write_text(json.dumps(dataset_metadata))

    # Requires `kaggle datasets init`-style metadata file to already exist
    # once (first run only); subsequent runs just version.
    os.system(f"kaggle datasets version -p {workdir} -m 'batch of {len(items)} labeled items'")

    Path(args.out_manifest).write_text(json.dumps({"request_ids": request_ids}))
    print(f"Assembled and pushed {len(items)} labeled items as a new dataset version.")


if __name__ == "__main__":
    main()
