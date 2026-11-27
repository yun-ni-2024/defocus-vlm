"""Check which training loss agrees with reading accuracy, before training with it.

For a few saved filters of method (d) (e.g. the starting point and the end of the feature and pixel
runs), every validation photograph is restored and three losses are computed per image:
  pixel:   MSE to the sharp photograph after a gain and offset (src/frontend/train.py, pixel_loss)
  feature: MSE of the visual tokens to those of the sharp photograph (feature_loss)
  text:    cross-entropy of the correct transcription (src/frontend/text_loss.py)
They are compared with the reading accuracy stored in the run's val/it<N>.csv:
  - across filters: does the loss rank the filters as the accuracy does?
  - across images: Spearman correlation of the per-image loss with the per-image accuracy.

Usage (environment activated, on a GPU):
  python scripts/audit_losses.py configs/train_d_feature.yaml out.csv \
      start=runs/train_d_feature_X:it0 feature=runs/train_d_feature_X:it2000 pixel=runs/train_d_pixel_Y:it2000
Run folders are relative to DEFOCUS_ROOT; out.csv (per-image values) is written under DEFOCUS_ROOT.
"""
import csv
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.image import load_tif  # noqa: E402
from src.frontend.pairs import cache_key, load_radius_cache, split_pairs  # noqa: E402
from src.frontend.text_loss import TextLoss  # noqa: E402
from src.frontend.train import feature_loss, pixel_loss  # noqa: E402
from src.frontend.vlm_input import to_vlm_size  # noqa: E402
from src.frontend.wiener import LearnedWiener  # noqa: E402
from src.utils.run import defocus_root, load_config  # noqa: E402
from src.vlm.qwen import QwenReader  # noqa: E402


def spearman(a, b):
    """Spearman rank correlation of two equally long sequences (ties broken by order).

    Args:
        a: sequence of numbers.
        b: sequence of numbers.

    Returns:
        Float in [-1, 1].
    """
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    """Compute the three losses for every filter and validation image and report the agreement."""
    cfg = load_config(sys.argv[1])
    out_path = defocus_root() / sys.argv[2]
    # filters to audit: label=run_folder:iteration
    filters = []
    for arg in sys.argv[3:]:
        label, rest = arg.split("=")
        run, it = rest.split(":")
        filters.append((label, defocus_root() / run, it))

    m = cfg["model"]
    reader = QwenReader(defocus_root() / m["path"], dtype=m["dtype"], min_pixels=m["min_pixels"],
                        max_pixels=m["max_pixels"], max_new_tokens=m["max_new_tokens"],
                        prompt=m["prompt"], device=m.get("device", "cuda"))
    dev, wp, ls = reader.device, cfg["image"]["white_percentile"], cfg["image"]["long_side"]
    _, radii = load_radius_cache(cfg["data"]["radius_cache"])
    _, val = split_pairs(cfg["data"].get("steps"), cfg["data"]["val_every"])
    text = TextLoss(reader, load_tif(val[0].blurred).shape, ls)
    encoder, ip = reader.model.model.visual, reader.proc.image_processor

    rows = []
    for label, run, it in filters:
        model = LearnedWiener.load(run / "params" / f"{it}.json").to(dev)
        # the accuracy of this filter on each validation image, as measured during training
        with open(run / "val" / f"{it}.csv") as f:
            acc = {(int(r["step"]), r["font"], int(r["number"])): float(r["accuracy"]) for r in csv.DictReader(f)}
        for s in val:
            with torch.no_grad():
                sharp = to_vlm_size(torch.from_numpy(load_tif(s.sharp, wp))[None].to(dev), ls)
                blurred = torch.from_numpy(load_tif(s.blurred, wp))[None].to(dev)
                restored = to_vlm_size(model(blurred, torch.tensor([radii[cache_key(s.blurred)]], device=dev)), ls)
                # the three losses for this one image
                row = {"filter": label, "step": s.step, "font": s.font, "number": s.number,
                       "accuracy": acc[(s.step, s.font, s.number)],
                       "pixel": pixel_loss(restored, sharp).item(),
                       "feature": feature_loss(encoder, ip, reader.size, reader.model.dtype, restored, sharp).item(),
                       "text": text.per_image(restored, [s.text]).item()}
            rows.append(row)
        print(f"{label}: done", flush=True)

    # per-image values for later analysis
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # across filters: mean accuracy and mean loss of each filter
    print("\nmean per filter (accuracy higher is better; losses lower is better)")
    for label, _, _ in filters:
        rs = [r for r in rows if r["filter"] == label]
        print(f"  {label:10s} accuracy {np.mean([r['accuracy'] for r in rs]):.3f}  "
              + "  ".join(f"{k} {np.mean([r[k] for r in rs]):.4g}" for k in ("pixel", "feature", "text")))
    # across images: a good loss is high where the accuracy is low (negative correlation)
    print("\nSpearman correlation of per-image loss with per-image accuracy (more negative = better aligned)")
    for k in ("pixel", "feature", "text"):
        print(f"  {k:8s} all filters {spearman([r[k] for r in rows], [r['accuracy'] for r in rows]):+.3f}  "
              + "  ".join(f"{label} {spearman([r[k] for r in rows if r['filter'] == label], [r['accuracy'] for r in rows if r['filter'] == label]):+.3f}"
                          for label, _, _ in filters))
    # within each image: does the change of the loss between two filters predict the change of accuracy?
    base = filters[0][0]
    key = lambda r: (r["step"], r["font"], r["number"])  # noqa: E731
    ref = {key(r): r for r in rows if r["filter"] == base}
    print(f"\nper image, change from {base}: fraction of images where the loss and the accuracy move in agreeing directions")
    for label, _, _ in filters[1:]:
        changed = [(r, ref[key(r)]) for r in rows if r["filter"] == label and r["accuracy"] != ref[key(r)]["accuracy"]]
        parts = []
        for k in ("pixel", "feature", "text"):
            agree = np.mean([(r[k] - b[k]) * (r["accuracy"] - b["accuracy"]) < 0 for r, b in changed]) if changed else float("nan")
            parts.append(f"{k} {agree:.2f}")
        print(f"  {label} ({len(changed)} images whose accuracy changed): " + "  ".join(parts))


if __name__ == "__main__":
    main()
