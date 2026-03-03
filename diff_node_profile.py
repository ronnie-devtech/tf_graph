import argparse
import csv
import os
import re
from collections import Counter, defaultdict


def normalize_node_name(name):
    n = name.split(":")[0].lstrip("^")
    # TF runtime wrapper names
    n = re.sub(r"^_(arg|retval)_(.+)_\d+_\d+$", r"\2", n)
    # Grappler prefixes
    n = re.sub(r"^ConstantFolding/", "", n)
    n = re.sub(
        r"^ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_(float|half|double|int32|int64)_",
        "",
        n,
    )
    # Layout optimizer generated helper nodes
    n = re.sub(r"-\d+-0?-?TransposeNHWCToNCHW-LayoutOptimizer$", "", n)
    n = re.sub(r"-\d+-0?-?TransposeNCHWToNHWC-LayoutOptimizer$", "", n)
    n = re.sub(r"-\d+-0?-?PermConstNHWCToNCHW-LayoutOptimizer$", "", n)
    n = re.sub(r"-\d+-0?-?PermConstNCHWToNHWC-LayoutOptimizer$", "", n)
    # Common optimizer suffixes
    n = re.sub(r"_recip$", "", n)
    n = re.sub(r"_const_axis$", "", n)
    return n


def load_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            node_name = r.get("node_name", "")
            op_type = r.get("op_type", "Unknown")
            rows.append(
                {
                    "node_name": node_name,
                    "norm_name": normalize_node_name(node_name),
                    "op_type": op_type,
                    "device": r.get("device", ""),
                    "time_us": float(r.get("time_us", "0") or 0.0),
                }
            )
    return rows


def analyze(pb_dir, topk=20):
    musa_csv = os.path.join(pb_dir, "musa", "node_profile.csv")
    cuda_csv = os.path.join(pb_dir, "cuda", "node_profile.csv")

    musa_rows = load_rows(musa_csv)
    cuda_rows = load_rows(cuda_csv)

    musa_op_cnt = Counter(r["op_type"] for r in musa_rows)
    cuda_op_cnt = Counter(r["op_type"] for r in cuda_rows)
    all_ops = sorted(set(musa_op_cnt) | set(cuda_op_cnt))

    print(f"=== {pb_dir} ===")
    print(f"musa node count: {len(musa_rows)}")
    print(f"cuda node count: {len(cuda_rows)}")
    print(f"total diff (musa-cuda): {len(musa_rows) - len(cuda_rows)}\n")

    print("OpType count diff (musa-cuda, only non-zero):")
    for op in all_ops:
        d = musa_op_cnt[op] - cuda_op_cnt[op]
        if d != 0:
            print(f"  {op:24s} musa={musa_op_cnt[op]:5d} cuda={cuda_op_cnt[op]:5d} diff={d:4d}")
    print("")

    by_op_musa = defaultdict(Counter)
    by_op_cuda = defaultdict(Counter)
    for r in musa_rows:
        by_op_musa[r["op_type"]][r["norm_name"]] += 1
    for r in cuda_rows:
        by_op_cuda[r["op_type"]][r["norm_name"]] += 1

    print(f"Node-level mismatches by op_type (top {topk} each side):")
    for op in all_ops:
        d = musa_op_cnt[op] - cuda_op_cnt[op]
        if d == 0:
            continue
        print(f"\n[{op}]")
        only_musa = by_op_musa[op] - by_op_cuda[op]
        only_cuda = by_op_cuda[op] - by_op_musa[op]

        musa_items = only_musa.most_common(topk)
        cuda_items = only_cuda.most_common(topk)
        if musa_items:
            print("  only in MUSA:")
            for name, cnt in musa_items:
                print(f"    {name} x{cnt}")
        if cuda_items:
            print("  only in CUDA:")
            for name, cnt in cuda_items:
                print(f"    {name} x{cnt}")

    unknown_cuda = [r for r in cuda_rows if r["op_type"] == "Unknown"]
    unknown_musa = [r for r in musa_rows if r["op_type"] == "Unknown"]
    print("\nUnknown nodes:")
    print(f"  MUSA: {len(unknown_musa)}")
    print(f"  CUDA: {len(unknown_cuda)}")
    if unknown_cuda:
        print("  CUDA Unknown sample:")
        for r in unknown_cuda[:topk]:
            print(f"    {r['node_name']} | {r['device']} | {r['time_us']} us")
    if unknown_musa:
        print("  MUSA Unknown sample:")
        for r in unknown_musa[:topk]:
            print(f"    {r['node_name']} | {r['device']} | {r['time_us']} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb", required=True, help="Directory like meta_graph_2_frozen")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()
    analyze(args.pb, topk=args.topk)
