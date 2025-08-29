#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance client for Triton OCR ensemble:
- Parallel request flood with configurable --concurrency
- Optional batching (--batch-size) when the server supports it
- Repeats images if you ask for more parallelism/requests than images available
- Prints wall time, throughput, and latency percentiles

Assumptions (override with flags if your repo differs):
  model_name  : ensemble_model
  input_name  : input_image
  outputs     : rec_text, rec_score, dt_boxes

Note: True in-request batching requires same HxW across images (or server-side padding).
If shapes differ, this client falls back to single-image requests for those items.
"""
import argparse
import concurrent.futures as cf
import itertools
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def load_image(path: str, to_rgb: bool = True, as_fp32: bool = False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_fp32:
        arr = img.astype(np.float32) / 255.0
        dtype = "FP32"
    else:
        arr = np.ascontiguousarray(img, dtype=np.uint8)
        dtype = "UINT8"
    return arr, dtype


def make_infer_inputs(
    batch: List[np.ndarray],
    input_name: str,
    dtype: str,
    batched: bool,
):
    import tritonclient.grpc as grpcclient  # local import to avoid module reorder issues

    if batched:
        # All images must match H, W, C
        first_shape = batch[0].shape
        for i, arr in enumerate(batch):
            if arr.shape != first_shape:
                raise ValueError(f"Batch images have mismatched shapes: {first_shape} vs {arr.shape} (idx={i})")
        data = np.stack(batch, axis=0)  # [B,H,W,3]
    else:
        data = batch[0]  # [H,W,3]

    inp = grpcclient.InferInput(input_name, data.shape, dtype)
    inp.set_data_from_numpy(data)
    return [inp]


def decode_bytes_array(arr: np.ndarray) -> List[str]:
    # Triton returns dtype=object/bytes for TYPE_BYTES
    out = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8", errors="ignore"))
        elif isinstance(x, np.ndarray) and x.dtype.type is np.object_:
            # nested object case (unlikely here)
            out.append(bytes(x).decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def fetch_model_capabilities(client: grpcclient.InferenceServerClient, model_name: str):
    cfg = client.get_model_config(model_name=model_name, as_json=True)
    config = cfg["config"]
    max_bs = int(config.get("max_batch_size", 0))
    inputs = {i["name"]: i for i in config.get("input", [])}
    outputs = {o["name"]: o for o in config.get("output", [])}
    return max_bs, inputs, outputs


def worker_infer(
    server_url: str,
    model_name: str,
    input_name: str,
    output_names: List[str],
    images: List[str],
    as_fp32: bool,
    to_rgb: bool,
    batched: bool,
    timeout_s: float,
) -> Tuple[float, dict]:
    """
    Build request (possibly batched) and time it end-to-end on a fresh client.
    Returns (latency_seconds, minimal_result_info)
    """
    st = time.perf_counter()
    with grpcclient.InferenceServerClient(server_url, verbose=False) as client:
        # load + prepare payload
        arrays = []
        dtype_seen = None
        for p in images:
            arr, dtype = load_image(p, to_rgb=to_rgb, as_fp32=as_fp32)
            if dtype_seen is None:
                dtype_seen = dtype
            elif dtype_seen != dtype:
                raise ValueError(f"Mixed dtypes in batch: {dtype_seen} vs {dtype}")
            arrays.append(arr)

        inputs = make_infer_inputs(arrays, input_name=input_name, dtype=dtype_seen, batched=batched)
        requested_outputs = [grpcclient.InferRequestedOutput(n) for n in output_names]
        results = client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=requested_outputs,
            client_timeout=timeout_s if timeout_s and timeout_s > 0 else None,
        )

        # tiny summary (count/shape only, avoids big prints)
        info = {}
        for n in output_names:
            try:
                arr = results.as_numpy(n)
            except InferenceServerException as e:
                info[n] = f"ERROR: {e}"
                continue
            if arr is None:
                info[n] = "None"
                continue
            if arr.dtype == np.object_:
                # bytes/strings
                info[n] = f"len={len(arr)} dtype=bytes"
            else:
                info[n] = f"shape={tuple(arr.shape)} dtype={arr.dtype}"

    et = time.perf_counter()
    return (et - st), info


def main():
    ap = argparse.ArgumentParser(description="Triton OCR perf client (gRPC)")
    ap.add_argument("--server-url", default="localhost:8001", help="host:port for Triton gRPC")
    ap.add_argument("--model-name", default="ensemble_model")
    ap.add_argument("--input-name", default="input_image")
    ap.add_argument("--outputs", default="rec_text,rec_score,dt_boxes",
                    help="comma-separated output names to request")
    ap.add_argument("--fp32", action="store_true", help="send FP32 normalized [0,1] instead of UINT8")
    ap.add_argument("--bgr", action="store_true", help="send BGR instead of RGB (default RGB)")
    ap.add_argument("--batch-size", type=int, default=1, help="images per request (requires same HxW)")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel requests (threads)")
    ap.add_argument("--requests", type=int, default=0, help="total number of requests to send (0 = len(images))")
    ap.add_argument("--warmup", type=int, default=2, help="number of warmup requests (not timed)")
    ap.add_argument("--timeout", type=float, default=0.0, help="per-request timeout in seconds (0=none)")
    ap.add_argument("--print-every", type=int, default=0, help="print every Nth result summary (0=never)")
    ap.add_argument("images", nargs="+", help="list of image files")
    args = ap.parse_args()

    to_rgb = not args.bgr
    out_names = [s.strip() for s in args.outputs.split(",") if s.strip()]

    # Basic validation
    for p in args.images:
        if not os.path.isfile(p):
            print(f"ERROR: not a file: {p}", file=sys.stderr)
            sys.exit(2)

    # Inspect server capabilities
    with grpcclient.InferenceServerClient(args.server_url, verbose=False) as probe:
        max_bs, inputs, outputs = fetch_model_capabilities(probe, args.model_name)

    # Decide batching mode
    want_batch = args.batch_size > 1
    if want_batch and max_bs <= 0:
        print(f"[warn] Server reports max_batch_size={max_bs}. Disabling batching.", file=sys.stderr)
        want_batch = False

    if want_batch and args.batch_size > max_bs:
        print(f"[warn] Requested --batch-size={args.batch_size} > server max_batch_size={max_bs}. "
              f"Clamping to {max_bs}.", file=sys.stderr)
        args.batch_size = max_bs

    # Build request plan
    total_requests = args.requests if args.requests > 0 else len(args.images)
    if total_requests <= 0:
        print("ERROR: total requests computed as 0. Provide images or --requests > 0.", file=sys.stderr)
        sys.exit(2)

    # If concurrency exceeds num images (or batch groups), we cycle the list
    img_cycle = itertools.cycle(args.images)

    # Warmup (not timed)
    if args.warmup > 0:
        print(f"[warmup] Sending {args.warmup} request(s)...")
        for _ in range(args.warmup):
            try:
                batch_imgs = [next(img_cycle)]
                _ = worker_infer(
                    server_url=args.server_url,
                    model_name=args.model_name,
                    input_name=args.input_name,
                    output_names=out_names,
                    images=batch_imgs,
                    as_fp32=bool(args.fp32),
                    to_rgb=to_rgb,
                    batched=False,  # keep warmups simple (single)
                    timeout_s=args.timeout,
                )
            except Exception as e:
                print(f"[warmup] ERROR: {e}", file=sys.stderr)

    # Prepare jobs
    jobs = []
    # When batching, we try to build batch_size images per request, but they must share shape.
    # Because raw OCR pipeline usually accepts variable sizes, we only batch when *all* images in a batch
    # happen to share shape. Otherwise we fall back to single-image requests for that batch slot.
    if want_batch:
        # Preload dims to try to form shape-homogeneous batches cheaply
        preload = []
        for _ in range(args.batch_size * args.concurrency):
            p = next(img_cycle)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {p}")
            preload.append((p, img.shape))
        # Use the preloaded shapes as a seed; after that we still just attempt best-effort grouping
        seed_pairs = itertools.cycle(preload)

    start_wall = time.perf_counter()
    latencies = []
    printed = 0

    def submit_one(executor):
        if want_batch:
            # Try to pick batch_size images with same shape; if not, just use the first one
            first_p, first_shape = next(seed_pairs)
            batch_paths = [first_p]
            while len(batch_paths) < args.batch_size:
                p = next(img_cycle)
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is not None and img.shape == first_shape:
                    batch_paths.append(p)
                else:
                    # can't batch safely -> will fall back to single image
                    break
            batched_flag = len(batch_paths) == args.batch_size
        else:
            batch_paths = [next(img_cycle)]
            batched_flag = False

        return executor.submit(
            worker_infer,
            args.server_url,
            args.model_name,
            args.input_name,
            out_names,
            batch_paths,
            bool(args.fp32),
            to_rgb,
            batched_flag,
            args.timeout,
        )

    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        # Prime up to concurrency
        inflight = [submit_one(ex) for _ in range(min(args.concurrency, total_requests))]
        launched = len(inflight)
        completed = 0

        while completed < total_requests:
            done, pending = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)

            # keep the not-yet-done futures as a list
            inflight = list(pending)

            for fut in done:
                completed += 1
                try:
                    latency, info = fut.result()
                    latencies.append(latency)
                    if args.print_every > 0 and (completed % args.print_every == 0):
                        printed += 1
                        print(f"[{completed}] {latency*1000:.1f} ms :: {info}")
                except Exception as e:
                    latencies.append(float("inf"))
                    print(f"[{completed}] ERROR: {e}", file=sys.stderr)

                # launch a replacement to keep concurrency until we hit the quota
                if launched < total_requests:
                    inflight.append(submit_one(ex))
                    launched += 1


    end_wall = time.perf_counter()
    wall = end_wall - start_wall

    # Stats
    good_latencies = [x for x in latencies if np.isfinite(x)]
    n = len(good_latencies)
    if n == 0:
        print("No successful requests; cannot compute stats.", file=sys.stderr)
        sys.exit(1)

    p50 = float(np.percentile(good_latencies, 50))
    p90 = float(np.percentile(good_latencies, 90))
    p95 = float(np.percentile(good_latencies, 95))
    avg = float(np.mean(good_latencies))
    fps = n / wall

    print("\n=== Performance Summary ===")
    print(f"Requests total          : {total_requests}")
    print(f"Concurrency (threads)   : {args.concurrency}")
    print(f"Batch size (requested)  : {args.batch_size}  "
          f"(effective: {'batched where possible' if want_batch else '1'})")
    print(f"Successful responses    : {n} / {total_requests}")
    print(f"Wall time               : {wall:.3f} s")
    print(f"Throughput              : {fps:.2f} images/sec")
    print(f"Avg latency             : {avg*1000:.2f} ms")
    print(f"P50 / P90 / P95         : {p50*1000:.2f} / {p90*1000:.2f} / {p95*1000:.2f} ms")


if __name__ == "__main__":
    main()
