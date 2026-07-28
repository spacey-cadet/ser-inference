"""
One-time, local Great Expectations run against your training/eval set,
before each retrain — null checks, expected label distribution, expected
duration range. Free, no AWS/warehouse needed. Explicitly NOT continuous
ingestion validation (there's no Gate 1/Gate 2 pipeline here) — this is a
pre-retrain sanity gate you run by hand or in CI right before kicking off
a training job.

Usage:
    python scripts/ge_validate_dataset.py --data-csv data/labels.csv
"""
import argparse
import sys

import great_expectations as gx
import pandas as pd

EXPECTED_LABELS = {"angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--label-col", default="emotion")
    ap.add_argument("--duration-col", default="duration_sec")
    ap.add_argument("--min-duration", type=float, default=0.3)
    ap.add_argument("--max-duration", type=float, default=15.0)
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("training_set")
    data_asset = data_source.add_dataframe_asset(name="labels")
    batch_def = data_asset.add_batch_definition_whole_dataframe("full")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    failures = []

    result = batch.validate(gx.expectations.ExpectColumnValuesToNotBeNull(column=args.label_col))
    if not result.success:
        failures.append(f"Nulls found in {args.label_col}")

    result = batch.validate(gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column=args.label_col, value_set=list(EXPECTED_LABELS)
    ))
    if not result.success:
        failures.append(f"Unexpected label values in {args.label_col} (expected subset of {EXPECTED_LABELS})")

    if args.duration_col in df.columns:
        result = batch.validate(gx.expectations.ExpectColumnValuesToBeBetween(
            column=args.duration_col, min_value=args.min_duration, max_value=args.max_duration
        ))
        if not result.success:
            failures.append(f"Durations outside [{args.min_duration}, {args.max_duration}]s")

    label_counts = df[args.label_col].value_counts()
    print("Label distribution:")
    print(label_counts)
    imbalance_ratio = label_counts.max() / label_counts.min()
    if imbalance_ratio > 5:
        print(f"WARNING: class imbalance ratio {imbalance_ratio:.1f}x — expected given "
              "the guide's noted Sad/Angry/Calm imbalance, but worth confirming it hasn't worsened.")

    if failures:
        print("FAILED checks:")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)

    print("All Great Expectations checks passed.")


if __name__ == "__main__":
    main()
