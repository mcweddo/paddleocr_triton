#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json

import cv2
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient


def _decode_utf8(arr: np.ndarray):
    """
    Triton returns TYPE_STRING as object/bytes. Decode to UTF-8.
    Returns a single string if size==1, else a list[str].
    """
    if arr is None:
        return None
    flat = arr.ravel()
    out = []
    for x in flat:
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out[0] if arr.size == 1 else out


def _output_names(result) -> list:
    try:
        return [o.name for o in result.get_response().outputs]
    except Exception:
        return []


def _valid_polygon(poly: np.ndarray, min_area: float = 5.0) -> bool:
    """
    poly: (4,2) float/int
    """
    p = poly.astype(np.float32)
    if not np.isfinite(p).all():
        return False
    if np.allclose(p, 0):
        return False
    area = cv2.contourArea(p.astype(np.float32))
    return area >= min_area


def _draw_and_save_boxes(img_bgr: np.ndarray,
                         boxes: np.ndarray,
                         out_path: str,
                         max_count: int | None = None) -> bool:
    """
    boxes expected as [N,4,2] or [1,N,4,2]. Draws polygons and saves image.
    max_count: if provided (e.g., from dt_counts), only draw first max_count boxes.
    """
    if boxes is None:
        return False

    b = boxes
    if b.ndim == 4 and b.shape[0] == 1:
        b = b[0]
    b = b.reshape(-1, 4, 2)

    if max_count is not None:
        b = b[: int(max_count)]

    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    drawn = 0
    for i, poly in enumerate(b):
        poly = np.clip(poly, [0, 0], [w - 1, h - 1]).astype(np.int32)
        if not _valid_polygon(poly):
            continue
        cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        p0 = poly[0]
        cv2.putText(
            vis, str(i + 1), (int(p0[0]), int(p0[1]) - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )
        drawn += 1

    ok = cv2.imwrite(out_path, vis)
    return bool(ok and drawn >= 0)


def infer_http(image_bgr: np.ndarray, args):
    client = httpclient.InferenceServerClient(url=args.url_http, verbose=args.verbose)

    # Triton ensemble expects HxWx3 UINT8 (no batch)
    inputs = []
    inp = httpclient.InferInput("input_image", image_bgr.shape, datatype="UINT8")
    inp.set_data_from_numpy(image_bgr)
    inputs.append(inp)

    t0 = time.time()
    result = client.infer(model_name=args.model_name, inputs=inputs)
    dt = time.time() - t0

    print(f"[HTTP] Time: {dt:.3f}s")

    # Decode UTF-8 text
    if "rec_text" in _output_names(result):
        rec = _decode_utf8(result.as_numpy("rec_text"))
        if isinstance(rec, list):
            print("Text(s):")
            for i, t in enumerate(rec, 1):
                print(f"[{i}] {t}")
        else:
            print(f"Text: {rec}")

    # Draw boxes and save
    out_path = _make_out_path(args.image, suffix="_bbox")
    names = _output_names(result)
    dt_boxes = result.as_numpy("dt_boxes") if "dt_boxes" in names else None
    dt_counts = result.as_numpy("dt_counts") if "dt_counts" in names else None
    max_count = int(dt_counts.ravel()[0]) if (dt_counts is not None and dt_counts.size > 0) else None

    if dt_boxes is not None:
        print("dt_boxes.shape:", dt_boxes.shape)
        if _draw_and_save_boxes(image_bgr, dt_boxes, out_path, max_count=max_count):
            print(f"Saved with boxes: {out_path}")
        else:
            print("WARN: failed to save boxed image")
    else:
        print("WARN: 'dt_boxes' not found in outputs")


def infer_grpc(image_bgr: np.ndarray, args):
    client = grpcclient.InferenceServerClient(url=args.url, verbose=args.verbose)

    inputs = []
    inp = grpcclient.InferInput("input_image", image_bgr.shape, datatype="UINT8")
    inp.set_data_from_numpy(image_bgr)
    inputs.append(inp)

    t0 = time.time()
    result = client.infer(model_name=args.model_name, inputs=inputs)
    dt = time.time() - t0

    try:
        stats = client.get_inference_statistics(model_name=args.model_name)
        if hasattr(stats, "model_stats"):
            print(stats)
    except Exception:
        pass

    print(f"[gRPC] Time: {dt:.3f}s")

    # Decode UTF-8 text
    if "rec_text" in _output_names(result):
        rec = _decode_utf8(result.as_numpy("rec_text"))
        if isinstance(rec, list):
            print("Text(s):")
            for i, t in enumerate(rec, 1):
                print(f"[{i}] {t}")
        else:
            print(f"Text: {rec}")

    # Draw boxes and save
    out_path = _make_out_path(args.image, suffix="_bbox")
    names = _output_names(result)
    dt_boxes = result.as_numpy("dt_boxes") if "dt_boxes" in names else None
    dt_counts = result.as_numpy("dt_counts") if "dt_counts" in names else None
    max_count = int(dt_counts.ravel()[0]) if (dt_counts is not None and dt_counts.size > 0) else None

    if dt_boxes is not None:
        print("dt_boxes.shape:", dt_boxes.shape)
        if _draw_and_save_boxes(image_bgr, dt_boxes, out_path, max_count=max_count):
            print(f"Saved with boxes: {out_path}")
        else:
            print("WARN: failed to save boxed image")
    else:
        print("WARN: 'dt_boxes' not found in outputs")


def _make_out_path(in_path: str, suffix: str = "_bbox") -> str:
    root, ext = os.path.splitext(in_path)
    ext = ext if ext else ".png"
    return f"{root}{suffix}{ext}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="ensemble_model", help="Model name")
    ap.add_argument("--image", type=str, required=True, help="Path to the image")
    ap.add_argument("--url", type=str, default="localhost:8001", help="gRPC URL (default localhost:8001)")
    ap.add_argument("--url_http", type=str, default="localhost:8000", help="HTTP URL (default localhost:8000)")
    ap.add_argument("--http", action="store_true", help="Use HTTP instead of gRPC")
    ap.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose client")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Read image in color and ensure contiguous UINT8 HxWx3
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    if args.http:
        infer_http(img, args)
    else:
        infer_grpc(img, args)


if __name__ == "__main__":
    main()
