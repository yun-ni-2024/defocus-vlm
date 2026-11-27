"""Estimate the disc radius of every blurred text photograph of a split once and store the results.

Training method (d) needs the estimated radius of every training photograph at every pass over the
data; the estimator takes a few seconds per photograph, so the radii are computed once here. The
estimator uses only the photograph and is deterministic, so a cached radius is the same number it
would give at that moment.

Usage (environment activated; CPU only, several processes):
  python scripts/estimate_radii.py train cache/radii_train_psf_backup.json --estimator psf_backup --workers 8
Writes the JSON file under DEFOCUS_ROOT:
  {"estimator": "psf_backup", "split": "train", "radius": {"<path under data/hdc2021>": r, ...}}
"""
import argparse
import importlib
import json
import pathlib
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.hdc import test_samples, train_samples  # noqa: E402
from src.data.image import load_tif  # noqa: E402
from src.frontend.pairs import cache_key  # noqa: E402
from src.utils.run import defocus_root  # noqa: E402

ESTIMATOR = None  # set in each worker process by init_worker()


def init_worker(estimator):
    """Import the estimator once in each worker process.

    Args:
        estimator: package under src/ that provides estimate_radius(a).
    """
    global ESTIMATOR
    # one thread per process, so the processes do not compete for the same cores
    import torch
    torch.set_num_threads(1)
    ESTIMATOR = importlib.import_module(f"src.{estimator}").estimate_radius


def estimate_one(path):
    """Estimate the radius of one blurred photograph.

    Args:
        path: path of the photograph.

    Returns:
        (cache key, estimated radius).
    """
    # the image exactly as the evaluation reads it (white_percentile does not change the radius)
    return cache_key(path), float(ESTIMATOR(load_tif(path)))


def main():
    """Estimate all radii of the split in parallel and write the cache file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("output", help="cache file, relative to DEFOCUS_ROOT")
    ap.add_argument("--estimator", default="psf_backup")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    samples = (train_samples if args.split == "train" else test_samples)()
    paths = [s.blurred for s in samples]
    print(f"{len(paths)} blurred photographs, estimator {args.estimator}, {args.workers} processes", flush=True)

    # estimate in parallel, reporting progress every 200 photographs
    radius, t0 = {}, time.time()
    with Pool(args.workers, initializer=init_worker, initargs=(args.estimator,)) as pool:
        for key, r in pool.imap_unordered(estimate_one, paths, chunksize=4):
            radius[key] = round(r, 3)
            if len(radius) % 200 == 0:
                print(f"{len(radius)}/{len(paths)}, {time.time() - t0:.0f} s", flush=True)

    # write the cache, sorted so the file is stable
    out = defocus_root() / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"estimator": args.estimator, "split": args.split,
                   "radius": dict(sorted(radius.items()))}, f, indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
