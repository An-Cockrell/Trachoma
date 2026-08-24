"""
Find subjects with conflicting TF grades (both 0 and 1) within the same source.

Groups by (source, id) and flags any group containing both a positive and
negative label.  Prints a summary and saves the conflicted rows to a CSV.

Usage:
    python find_conflicted_subjects.py \
        --csv /home/Trachoma/data/all_metadata.csv \
        --out_csv /home/Trachoma/data/conflicted_subjects.csv
"""

import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/Trachoma/data/all_metadata.csv")
    p.add_argument("--out_csv", default="/home/Trachoma/data/conflicted_subjects.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    # For each (source, id) group, check if it contains both label values
    group_labels = df.groupby(["source", "id"])["label"].agg(lambda x: set(x.unique()))
    conflicted = group_labels[group_labels.apply(lambda s: 0 in s and 1 in s)]

    n_conflicted_subjects = len(conflicted)
    conflicted_keys = conflicted.index.to_frame(index=False)  # DataFrame of (source, id)
    conflicted_rows = df.merge(conflicted_keys, on=["source", "id"], how="inner")
    n_conflicted_images = len(conflicted_rows)

    print(f"Conflicted subjects (mixed TF grade within source): {n_conflicted_subjects}")
    print(f"Total images from those subjects:                   {n_conflicted_images}")
    print()

    by_source = conflicted_keys.groupby("source").size().rename("conflicted_subjects")
    print("Breakdown by source:")
    print(by_source.to_string())
    print()

    conflicted_rows.to_csv(args.out_csv, index=False)
    print(f"Saved conflicted rows to: {args.out_csv}")

if __name__ == "__main__":
    main()
