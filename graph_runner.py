import collections
import argparse
import csv
import os
import re
import time

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import timeline

tf.compat.v1.disable_eager_execution()

musa_plugin_path = "/home/workspace/ronnie/code/tensorflow_musa_extension/build/libmusa_plugin.so"


# ==========================================
# 1. 加载 MUSA 插件
# ==========================================
def load_musa_plugin():
    if os.path.exists(musa_plugin_path):
        try:
            tf.load_op_library(musa_plugin_path)
            print(f">>>> [MUSA] Plugin loaded successfully from: {musa_plugin_path}")
        except Exception as e:
            print(f"!!!! [MUSA] Failed to load plugin: {e}")
    else:
        print(f"!!!! [MUSA] Plugin not found at {musa_plugin_path}, assuming built-in.")


def load_graph_def(pb_path):
    graph_def = graph_pb2.GraphDef()
    with open(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        if node.device:
            node.device = ""
    return graph_def


def import_graph(graph_def):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph


def scan_placeholders(graph, spec_path, batch_size):
    input_names = []
    for op in graph.get_operations():
        if op.type == "Placeholder":
            input_names.append(op.outputs[0].name)

    placeholders = []
    meta_graph = tf.Graph()
    with tf.compat.v1.Session(graph=meta_graph):
        tf.compat.v1.train.import_meta_graph(spec_path, clear_devices=False)
        input_tensors = meta_graph.get_collection("input_spec")
        for input_tensor in input_tensors:
            if input_tensor.name in input_names:
                shape = input_tensor.shape.as_list()
                if len(shape) > 0 and shape[0] is None:
                    shape[0] = batch_size
                shape = [1 if dim is None else dim for dim in shape]
                placeholders.append(
                    {
                        "name": input_tensor.name,
                        "dtype": input_tensor.dtype,
                        "shape": shape,
                    }
                )
    return placeholders


def generate_random_input(name, dtype, shape, rng):
    if dtype == tf.float32:
        return rng.random(shape, dtype=np.float32)
    if dtype == tf.float16:
        return rng.random(shape, dtype=np.float32).astype(np.float16)
    if dtype == tf.int32:
        return rng.integers(0, 100, size=shape, dtype=np.int32)
    if dtype == tf.int64:
        return rng.integers(0, 100, size=shape, dtype=np.int64)
    if dtype == tf.bool:
        return rng.integers(0, 2, size=shape, dtype=np.int8).astype(bool)
    if dtype == tf.string:
        return np.array([b"aweme_dou_plus" for _ in range(np.prod(shape))]).reshape(shape)
    raise ValueError(f"Unsupported dtype {dtype} for placeholder {name}")


def save_named_arrays(npz_path, names, arrays):
    payload = {"names": np.array(names, dtype=object)}
    for i, arr in enumerate(arrays):
        payload[f"arr_{i}"] = arr
    np.savez(npz_path, **payload)
    print(f"Saved arrays to: {npz_path}")


def load_named_arrays(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    names = data["names"].tolist()
    arrays = [data[f"arr_{i}"] for i in range(len(names))]
    return names, arrays

def log_op_dtypes(graph, csv_path, op_name_to_test=None):
    if op_name_to_test is None:
        return 
    op_rows = []
    for op in graph.get_operations():
        if op.type != op_name_to_test:
            continue
        input_names = ";".join(t.name for t in op.inputs) or "<none>"
        input_dtypes = ";".join(t.dtype.name for t in op.inputs) or "<unknown>"
        output_names = ";".join(t.name for t in op.outputs) or "<none>"
        output_dtypes = ";".join(t.dtype.name for t in op.outputs) or "<unknown>"
        op_rows.append([op.name, input_names, input_dtypes, output_names, output_dtypes])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["op_node", "input_tensor", "input_dtype", "output_tensor", "output_dtype"])
        writer.writerows(op_rows)

    print(f"Logged {len(op_rows)} {op_name_to_test} node dtype pairs to: {csv_path}")

def get_model_output_dir(pb_path, device=None):
    pb_abs = os.path.abspath(pb_path)
    pb_dir = os.path.dirname(pb_abs)
    model_name = os.path.splitext(os.path.basename(pb_abs))[0]
    model_dir = os.path.join(pb_dir, model_name)
    if device:
        model_dir = os.path.join(model_dir, device)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def resolve_model_path(model_dir, user_path, default_name):
    if user_path is None:
        out = os.path.join(model_dir, default_name)
    elif os.path.isabs(user_path):
        out = user_path
    else:
        out = os.path.join(model_dir, user_path)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return out


def build_feed_dict(graph, placeholders, seed=1234, load_input_npz=None):
    feed_dict = {}
    if load_input_npz:
        names, arrays = load_named_arrays(load_input_npz)
        name_to_arr = {k: v for k, v in zip(names, arrays)}
        for ph in placeholders:
            if ph["name"] not in name_to_arr:
                raise KeyError(f"Input npz missing placeholder: {ph['name']}")
            tensor = graph.get_tensor_by_name(ph["name"])
            feed_dict[tensor] = name_to_arr[ph["name"]]
        print(f"Loaded fixed inputs from: {load_input_npz}")
        return feed_dict

    rng = np.random.default_rng(seed)
    for ph in placeholders:
        tensor = graph.get_tensor_by_name(ph["name"])
        feed_dict[tensor] = generate_random_input(ph["name"], ph["dtype"], ph["shape"], rng)
    return feed_dict


def get_root_upstream_op_types(op):
    visited = set()
    roots = set()

    def dfs(cur_op):
        if not cur_op.inputs:
            roots.add(cur_op.type)
            return
        for tensor in cur_op.inputs:
            src_op = tensor.op
            if src_op not in visited:
                visited.add(src_op)
                dfs(src_op)

    dfs(op)
    return roots


def find_output_tensors(graph):
    outputs = []
    for op in graph.get_operations():
        if op.type not in {"NoOp", "Assert", "Print"} and all(
            len(out.consumers()) == 0 for out in op.outputs
        ):
            root_op_set = get_root_upstream_op_types(op)
            if root_op_set - {"Const", "VariableV2"} == set():
                continue
            outputs.extend(op.outputs)
    return outputs


# =========================
# 2. 名称匹配增强（关键）
# =========================
TOKEN_RE = re.compile(r"[A-Za-z0-9]{8,}")


def build_op_indices(graph):
    op_type_map = {op.name: op.type for op in graph.get_operations()}
    token_to_types = collections.defaultdict(set)

    for op_name, op_type in op_type_map.items():
        # 全名
        for t in TOKEN_RE.findall(op_name):
            token_to_types[t].add(op_type)

        # basename
        base = op_name.split("/")[-1]
        for t in TOKEN_RE.findall(base):
            token_to_types[t].add(op_type)

        # 常见后缀剥离
        for suf in ["_recip", "_const_axis"]:
            if base.endswith(suf):
                core = base[: -len(suf)]
                for t in TOKEN_RE.findall(core):
                    token_to_types[t].add(op_type)

    return op_type_map, token_to_types


def extract_runtime_token(node_name):
    n = node_name.split(":")[0].lstrip("^")

    # _arg_xxx_0_123 / _retval_xxx_0_3
    m = re.match(r"^_(arg|retval)_(.+)_\d+_\d+$", n)
    if m:
        n = m.group(2)

    # 剥常见后缀
    for suf in ["_recip", "_const_axis"]:
        if n.endswith(suf):
            n = n[: -len(suf)]

    toks = TOKEN_RE.findall(n)
    return toks[-1] if toks else n


def resolve_op_type(node_name, op_type_map, token_to_types):
    n = node_name.split(":")[0].lstrip("^")

    # 1) 直接命中
    if n in op_type_map:
        return op_type_map[n]

    # 2) token 命中
    token = extract_runtime_token(node_name)
    cands = token_to_types.get(token, set())
    if len(cands) == 1:
        return next(iter(cands))

    # 3) 模糊兜底
    fuzzy = {v for k, v in op_type_map.items() if k.endswith(token) or ("/" + token + "_") in k}
    if len(fuzzy) == 1:
        return next(iter(fuzzy))

    return "Unknown"


# =========================
# 3. Profile + Warmup
# =========================
def profile_and_run(sess, graph, outputs, feed_dict,
                    trace_json="tf_timeline.json",
                    op_csv="op_profile.csv",
                    node_csv="node_profile.csv",
                    topk_nodes=200):
    run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()

    output_values = sess.run(
        outputs,
        feed_dict=feed_dict,
        options=run_options,
        run_metadata=run_metadata,
    )

    tl = timeline.Timeline(run_metadata.step_stats)
    with open(trace_json, "w") as f:
        f.write(tl.generate_chrome_trace_format())

    op_type_map, token_to_types = build_op_indices(graph)

    # key: (op_type, device), value: {"time_us":..., "count":...}
    agg = collections.defaultdict(lambda: {"time_us": 0.0, "count": 0})
    node_rows = []

    for dev_stat in run_metadata.step_stats.dev_stats:
        device = dev_stat.device
        for ns in dev_stat.node_stats:
            op_type = resolve_op_type(ns.node_name, op_type_map, token_to_types)
            dur_us = float(ns.op_end_rel_micros - ns.op_start_rel_micros)

            agg[(op_type, device)]["time_us"] += dur_us
            agg[(op_type, device)]["count"] += 1
            node_rows.append((ns.node_name, op_type, device, dur_us))

    print("\n===== Op Profile (op_type + device) =====")
    op_rows = sorted(agg.items(), key=lambda x: x[1]["time_us"], reverse=True)
    for (op_type, device), v in op_rows:
        avg_us = v["time_us"] / v["count"] if v["count"] else 0.0
        print(f"{op_type:30s} | {device:45s} | count={v['count']:6d} | total={v['time_us']/1000.0:10.3f} ms | avg={avg_us:10.3f} us")

    with open(op_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op_type", "device", "count", "time_us", "time_ms", "avg_us_per_call"])
        for (op_type, device), v in op_rows:
            avg_us = v["time_us"] / v["count"] if v["count"] else 0.0
            w.writerow([op_type, device, v["count"], v["time_us"], v["time_us"] / 1000.0, avg_us])

    node_rows_sorted = sorted(node_rows, key=lambda x: x[3], reverse=True)
    with open(node_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_name", "op_type", "device", "time_us", "time_ms"])
        for node_name, op_type, device, us in node_rows_sorted:
            w.writerow([node_name, op_type, device, us, us / 1000.0])

    return output_values

def benchmark_model_latency(sess, outputs, feed_dict, iters=50, csv_path="model_latency.csv"):
    lat_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(outputs, feed_dict=feed_dict)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(lat_ms, dtype=np.float64)
    summary = {
        "iters": int(iters),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iters", "mean_ms", "min_ms", "max_ms", "p50_ms", "p90_ms", "p99_ms"])
        w.writerow([summary["iters"], summary["mean_ms"], summary["min_ms"], summary["max_ms"],
                    summary["p50_ms"], summary["p90_ms"], summary["p99_ms"]])

    print("\n===== Model Latency Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved model latency: {csv_path}")

    return summary

def run_inference(
    pb_path,
    spec_path,
    batch_size=1,
    warmup_iters=5,
    strict_placement=False,
    bench_iters=50,
    seed=1234,
    save_input_npz=None,
    load_input_npz=None,
    save_output_npz=None,
    disable_grappler=False,
    dtype_to_log_for_profile=None,
    device=None,
):
    # load_musa_plugin()

    graph_def = load_graph_def(pb_path)
    graph = import_graph(graph_def)
    model_output_dir = get_model_output_dir(pb_path, device=device)
    print(f"\n===== Artifact Dir =====\n{model_output_dir}")

    input_npz_to_save = resolve_model_path(model_output_dir, save_input_npz, "input.npz") \
        if save_input_npz is not None else None
    output_npz_to_save = resolve_model_path(model_output_dir, save_output_npz, "output.npz") \
        if save_output_npz is not None else None
    input_npz_to_load = resolve_model_path(model_output_dir, load_input_npz, "input.npz") \
        if load_input_npz is not None else None
    model_latency_csv = resolve_model_path(model_output_dir, None, "model_latency.csv")
    timeline_json = resolve_model_path(model_output_dir, None, "tf_timeline.json")
    op_profile_csv = resolve_model_path(model_output_dir, None, "op_profile.csv")
    node_profile_csv = resolve_model_path(model_output_dir, None, "node_profile.csv")
    dtype_to_log_csv = resolve_model_path(model_output_dir, None, f"{dtype_to_log_for_profile}_dtypes.csv")

    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = (not strict_placement)
    config.log_device_placement = True
    config.gpu_options.allow_growth = True
    if disable_grappler:
        rew = config.graph_options.rewrite_options
        rew.disable_meta_optimizer = True
        rew.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
        rew.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
        rew.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
        rew.remapping = rewriter_config_pb2.RewriterConfig.OFF

    placeholders = scan_placeholders(graph, spec_path=spec_path, batch_size=batch_size)
    outputs = find_output_tensors(graph)
    log_op_dtypes(graph, dtype_to_log_csv, dtype_to_log_for_profile)

    print("\n===== Placeholders =====")
    for ph in placeholders:
        print(f"{ph['name']} | dtype={ph['dtype']} | shape={ph['shape']}")

    print("\n===== Output Tensors =====")
    for out in outputs:
        print(f"{out.name} | dtype={out.dtype} | shape={out.shape}")

    feed_dict = build_feed_dict(graph, placeholders, seed=seed, load_input_npz=input_npz_to_load)

    if input_npz_to_save:
        input_names = [ph["name"] for ph in placeholders]
        input_values = [feed_dict[graph.get_tensor_by_name(ph["name"])] for ph in placeholders]
        save_named_arrays(input_npz_to_save, input_names, input_values)

    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        for _ in range(warmup_iters):
            sess.run(outputs, feed_dict=feed_dict)
        print(f"\n===== Warmup Done ({warmup_iters} iters) =====")

        # 整模型耗时统计（不带 timeline，避免 tracing 开销干扰）
        benchmark_model_latency(sess, outputs, feed_dict, iters=bench_iters, csv_path=model_latency_csv)

        # 单次 profile（导出 timeline + op/node 明细）
        output_values = profile_and_run(
            sess, graph, outputs, feed_dict,
            trace_json=timeline_json,
            op_csv=op_profile_csv,
            node_csv=node_profile_csv,
            topk_nodes=200
        )

    print("\n===== Inference Done =====")
    for out, val in zip(outputs, output_values):
        print(f"{out.name} -> output shape: {val.shape}")

    if output_npz_to_save:
        output_names = [out.name for out in outputs]
        save_named_arrays(output_npz_to_save, output_names, output_values)

    return output_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-path", default="meta_graph_1.spec")
    parser.add_argument("--pb-path", default="meta_graph_1_frozen.pb")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--strict-placement", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--save-input-npz", default=None)
    parser.add_argument("--load-input-npz", default=None)
    parser.add_argument("--save-output-npz", default=None)
    parser.add_argument("--load-musa-plugin", action="store_true")
    parser.add_argument("--disable-grappler", action="store_true")
    parser.add_argument("--dtype_to_log_for_profile", default=None, help="If set, will log the input/output dtypes of all nodes of this op type to a csv for analysis.")
    args = parser.parse_args()

    if args.save_input_npz and args.load_input_npz:
        raise ValueError("save-input-npz and load-input-npz cannot be used together.")
    if args.load_musa_plugin:
        load_musa_plugin()
    
    run_inference(
        pb_path=args.pb_path,
        spec_path=args.spec_path,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        strict_placement=args.strict_placement,
        bench_iters=args.bench_iters,
        seed=args.seed,
        save_input_npz=args.save_input_npz,
        load_input_npz=args.load_input_npz,
        save_output_npz=args.save_output_npz,
        disable_grappler=args.disable_grappler,
        device="musa" if args.load_musa_plugin else "cuda",
        dtype_to_log_for_profile=args.dtype_to_log_for_profile
    )
    
