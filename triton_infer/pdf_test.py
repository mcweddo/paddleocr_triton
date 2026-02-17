#!/usr/bin/env python3
"""
Benchmark: PyMuPDF (fitz) vs pypdfium2 PDF->image throughput

Usage:
  python benchmark_pdf_render.py --input /path/to/pdfs --dpi 200 --fmt png --workers 0 --engines pdfium pymupdf --repeats 1 --csv out.csv

Notes:
- Parallelizes ACROSS PDFs (not within a single PDF) for stable memory use.
- Measures wall-clock time, pages processed, PDFs processed, pages/sec, and PDFs/sec.
- Skips encrypted PDFs that cannot be opened without a password.
- Requires: pymupdf, pypdfium2, pillow (or pillow-simd). Installs are up to you.
"""

import argparse
import os
import sys
import time
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

# --------- Renderers ---------

def _render_pdf_pypdfium2(path: str, out_dir: str, dpi: int, fmt: str) -> Tuple[int, float]:
    import pypdfium2 as pdfium
    from PIL import Image  # noqa: F401
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    pages_done = 0
    scale = dpi / 72.0
    try:
        pdf = pdfium.PdfDocument(path)
    except Exception:
        return (0, time.perf_counter() - t0)
    try:
        n = len(pdf)
        for i in range(n):
            page = pdf[i]
            pil = page.render(scale=scale).to_pil()
            out_path = os.path.join(out_dir, f"{os.path.basename(path)}-p{i+1}.{fmt}")
            if fmt.lower() == "jpeg":
                pil.save(out_path, quality=90)
            else:
                pil.save(out_path)
            pages_done += 1
            pil.close()
            page.close()
    finally:
        try:
            pdf.close()
        except Exception:
            pass
    return (pages_done, time.perf_counter() - t0)


def _render_pdf_pymupdf(path: str, out_dir: str, dpi: int, fmt: str) -> Tuple[int, float]:
    import fitz  # PyMuPDF
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    pages_done = 0
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    try:
        doc = fitz.open(path)
    except Exception:
        return (0, time.perf_counter() - t0)
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=(fmt.lower() == "png"))
            out_path = os.path.join(out_dir, f"{os.path.basename(path)}-p{i}.{fmt}")
            pix.save(out_path)
            pages_done += 1
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return (pages_done, time.perf_counter() - t0)


RENDERERS = {
    "pdfium": _render_pdf_pypdfium2,
    "pymupdf": _render_pdf_pymupdf,
}

# --------- Worker ---------

def _worker_task(args: Tuple[str, str, str, int, str]) -> Tuple[str, int, float, Optional[str]]:
    engine, pdf_path, out_root, dpi, fmt = args
    out_dir = os.path.join(out_root, os.path.splitext(os.path.basename(pdf_path))[0])
    try:
        pages, seconds = RENDERERS[engine](pdf_path, out_dir, dpi, fmt)
        return (pdf_path, pages, seconds, None)
    except Exception as e:
        return (pdf_path, 0, 0.0, f"{type(e).__name__}: {e}")

@dataclass
class EngineResult:
    engine: str
    total_pdfs: int
    total_pages: int
    wall_time: float
    pages_per_sec: float
    pdfs_per_sec: float
    errors: int

def _run_engine(engine: str, pdfs: List[str], out_root: str, dpi: int, fmt: str, workers: int, repeats: int) -> EngineResult:
    if engine not in RENDERERS:
        raise ValueError(f"Unknown engine: {engine}")

    # Prepare tasks
    tasks = []
    for _ in range(repeats):
        for p in pdfs:
            tasks.append((engine, p, out_root, dpi, fmt))

    start = time.perf_counter()
    total_pages = 0
    total_pdfs = 0
    errors = 0

    if workers == 1:
        for t in tasks:
            _, pages, _, err = _worker_task(t)
            total_pages += pages
            total_pdfs += 1
            if err:
                errors += 1
    else:
        with mp.Pool(processes=workers, maxtasksperchild=10) as pool:
            for (_pdf_path, pages, _secs, err) in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                total_pages += pages
                total_pdfs += 1
                if err:
                    errors += 1

    wall = time.perf_counter() - start
    pps = (total_pages / wall) if wall > 0 else 0.0
    dps = (total_pdfs / wall) if wall > 0 else 0.0
    return EngineResult(engine=engine, total_pdfs=total_pdfs, total_pages=total_pages,
                        wall_time=wall, pages_per_sec=pps, pdfs_per_sec=dps, errors=errors)

def find_pdfs(input_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                out.append(os.path.join(root, f))
    out.sort()
    return out

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark PDF->image conversion engines")
    ap.add_argument("--input", required=True, help="Directory containing PDFs (recursive)")
    ap.add_argument("--out", default="benchmark_out", help="Output directory for rendered images")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200)")
    ap.add_argument("--fmt", choices=["png", "jpeg"], default="png", help="Output image format")
    ap.add_argument("--workers", type=int, default=0, help="Worker processes (0 = CPU cores)")
    ap.add_argument("--engines", nargs="+", default=["pdfium", "pymupdf"], choices=list(RENDERERS.keys()),
                    help="Engines to benchmark")
    ap.add_argument("--repeats", type=int, default=1, help="Repeat whole corpus N times to stabilize results")
    ap.add_argument("--csv", default=None, help="Optional CSV to append summary rows")
    ap.add_argument("--list-only", action="store_true", help="Just list found PDFs and exit")
    return ap.parse_args()

def main():
    args = parse_args()

    pdfs = find_pdfs(args.input)
    if not pdfs:
        print("No PDFs found in:", args.input, file=sys.stderr)
        sys.exit(2)

    if args.list_only:
        for p in pdfs:
            print(p)
        print(f"Found {len(pdfs)} PDFs.")
        return

    workers = args.workers if args.workers > 0 else max(1, mp.cpu_count())
    print(f"Using {workers} worker processes.")
    os.makedirs(args.out, exist_ok=True)

    # Print environment info
    print("== Benchmark config ==")
    print(f"PDFs: {len(pdfs)}  |  DPI: {args.dpi}  |  Format: {args.fmt}  |  Workers: {workers}  |  Repeats: {args.repeats}")
    print(f"Engines: {', '.join(args.engines)}")
    print(f"Output dir: {os.path.abspath(args.out)}")
    print()

    results: List[EngineResult] = []
    for engine in args.engines:
        print(f"--- {engine} ---")
        # Quick import check
        try:
            if engine == "pdfium":
                import pypdfium2  # noqa: F401
            elif engine == "pymupdf":
                import fitz  # noqa: F401
        except Exception as e:
            print(f"SKIP {engine}: not importable ({e})")
            continue

        out_root = os.path.join(args.out, engine)
        os.makedirs(out_root, exist_ok=True)

        res = _run_engine(engine, pdfs, out_root, args.dpi, args.fmt, workers, args.repeats)
        results.append(res)

        print(f"PDFs processed: {res.total_pdfs}  |  Pages: {res.total_pages}  |  Errors: {res.errors}")
        print(f"Wall time: {res.wall_time:.3f}s  |  Pages/sec: {res.pages_per_sec:.2f}  |  PDFs/sec: {res.pdfs_per_sec:.2f}")
        print()

    if not results:
        print("No engines were benchmarked (imports failed?).", file=sys.stderr)
        sys.exit(1)

    # CSV output (append-safe, add header if file doesn't exist)
    if args.csv:
        import csv
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "engine","total_pdfs","total_pages","wall_time","pages_per_sec","pdfs_per_sec","errors",
                "dpi","fmt","workers","repeats","input_dir","timestamp"
            ])
            if write_header:
                w.writeheader()
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            for r in results:
                row = asdict(r)
                row.update({
                    "dpi": args.dpi,
                    "fmt": args.fmt,
                    "workers": workers,
                    "repeats": args.repeats,
                    "input_dir": os.path.abspath(args.input),
                    "timestamp": ts,
                })
                w.writerow(row)

    print("== Summary ==")
    for r in results:
        print(f"{r.engine:8s} | pages/sec: {r.pages_per_sec:8.2f} | pdfs/sec: {r.pdfs_per_sec:6.2f} | pages: {r.total_pages:7d} | wall(s): {r.wall_time:8.2f} | errors: {r.errors}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
