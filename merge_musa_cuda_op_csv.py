import argparse
import csv
import os
from collections import defaultdict


def _to_float(v, default=0.0):
    if v is None or v == "":
        return default
    return float(v)


def _to_int(v, default=0):
    if v is None or v == "":
        return default
    return int(float(v))


def load_op_profile(csv_path):
    # key: op_type, value: {"time_us": float, "count": int}
    agg = defaultdict(lambda: {"time_us": 0.0, "count": 0})

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "op_type" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} missing required column: op_type")

        has_time_us = "time_us" in (reader.fieldnames or [])
        has_time_ms = "time_ms" in (reader.fieldnames or [])
        if not has_time_us and not has_time_ms:
            raise ValueError(f"{csv_path} missing required column: time_us or time_ms")
        if "count" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} missing required column: count")

        for row in reader:
            op_type = row["op_type"]
            if not op_type:
                continue

            if has_time_us:
                time_us = _to_float(row.get("time_us"), 0.0)
            else:
                time_us = _to_float(row.get("time_ms"), 0.0) * 1000.0

            count = _to_int(row.get("count"), 0)
            agg[op_type]["time_us"] += time_us
            agg[op_type]["count"] += count

    return agg


def merge_op_profiles(musa_csv, cuda_csv, out_csv, sort_by="abs_time_diff_desc"):
    musa = load_op_profile(musa_csv)
    cuda = load_op_profile(cuda_csv)

    all_ops = set(musa.keys()) | set(cuda.keys())
    rows = []

    for op_type in all_ops:
        musa_time_us = musa[op_type]["time_us"] if op_type in musa else 0.0
        cuda_time_us = cuda[op_type]["time_us"] if op_type in cuda else 0.0
        musa_count = musa[op_type]["count"] if op_type in musa else 0
        cuda_count = cuda[op_type]["count"] if op_type in cuda else 0

        rows.append(
            {
                "op_type": op_type,
                "musa_time_us": musa_time_us,
                "cuda_time_us": cuda_time_us,
                "time_us_diff_musa_minus_cuda": musa_time_us - cuda_time_us,
                "musa_count": musa_count,
                "cuda_count": cuda_count,
                "count_diff_musa_minus_cuda": musa_count - cuda_count,
            }
        )

    if sort_by == "abs_time_diff_desc":
        rows.sort(key=lambda r: abs(r["time_us_diff_musa_minus_cuda"]), reverse=True)
    elif sort_by == "time_diff_desc":
        rows.sort(key=lambda r: r["time_us_diff_musa_minus_cuda"], reverse=True)
    elif sort_by == "op_type":
        rows.sort(key=lambda r: r["op_type"])

    total_musa_time = sum(r["musa_time_us"] for r in rows)
    total_cuda_time = sum(r["cuda_time_us"] for r in rows)
    total_musa_count = sum(r["musa_count"] for r in rows)
    total_cuda_count = sum(r["cuda_count"] for r in rows)

    total_row = {
        "op_type": "__TOTAL__",
        "musa_time_us": total_musa_time,
        "cuda_time_us": total_cuda_time,
        "time_us_diff_musa_minus_cuda": total_musa_time - total_cuda_time,
        "musa_count": total_musa_count,
        "cuda_count": total_cuda_count,
        "count_diff_musa_minus_cuda": total_musa_count - total_cuda_count,
    }

    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "op_type",
            "musa_time_us",
            "cuda_time_us",
            "time_us_diff_musa_minus_cuda",
            "musa_count",
            "cuda_count",
            "count_diff_musa_minus_cuda",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(total_row)

    print(f"Merged CSV saved to: {out_csv}")
    print("Summary:")
    print(f"  musa_time_us_total: {total_musa_time:.3f}")
    print(f"  cuda_time_us_total: {total_cuda_time:.3f}")
    print(f"  time_us_diff(musa-cuda): {total_musa_time - total_cuda_time:.3f}")
    print(f"  musa_count_total: {total_musa_count}")
    print(f"  cuda_count_total: {total_cuda_count}")
    print(f"  count_diff(musa-cuda): {total_musa_count - total_cuda_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", default="meta_graph_1_frozen")
    parser.add_argument("--out-csv", default="musa_cuda_op_merged.csv")
    parser.add_argument(
        "--sort-by",
        default="time_diff_desc",
        choices=["abs_time_diff_desc", "time_diff_desc", "op_type"],
        help="Sort output rows.",
    )
    args = parser.parse_args()
    
    musa_csv = os.path.join(args.pb, "musa", "op_profile.csv")
    cuda_csv = os.path.join(args.pb, "cuda", "op_profile.csv")
    out_csv = os.path.join(args.pb, args.out_csv)
    merge_op_profiles(
        musa_csv=musa_csv,
        cuda_csv=cuda_csv,
        out_csv=out_csv,
        sort_by=args.sort_by,
    )
