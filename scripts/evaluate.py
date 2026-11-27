"""Evaluate one method: transcribe photographs with Qwen2-VL and score them.

The pipeline is the same for every method; the config fills its two slots:
  slot 1, `restore`:    image processing between load_tif() and to_pil() (src/restore/), null = none
  slot 2, `model.lora`: a LoRA adapter for the vision encoder, null = the original encoder
With both slots empty this is method (a) (camera: blurred) or the sharp reference (camera: sharp).

Usage (environment activated, on a GPU node; normally through slurm/gpu.sbatch):
  python scripts/evaluate.py configs/baseline_a.yaml
Writes runs/<name>_<time>/ with config.yaml, meta.json, predictions.csv and summary.csv.
"""
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.hdc import test_samples, train_samples  # noqa: E402
from src.data.image import load_tif, to_pil  # noqa: E402
from src.eval.metrics import char_accuracy  # noqa: E402
from src.eval.summary import summarise, write_summary  # noqa: E402
from src.restore import build_restorer  # noqa: E402
from src.utils.run import defocus_root, load_config, new_run  # noqa: E402
from src.vlm.qwen import QwenReader  # noqa: E402


def select(cfg):
    """Pick the samples named in the config's data section.

    Args:
        cfg: the config dict.

    Returns:
        List of Sample.
    """
    d = cfg["data"]
    samples = (test_samples if d["split"] == "test" else train_samples)(steps=d.get("steps"))
    numbers = d.get("sample_numbers")
    if numbers is not None:
        if d["split"] != "train":
            raise ValueError("sample_numbers is reserved for training subsets")
        samples = [sample for sample in samples if sample.number in numbers]
    # optionally keep only the first few images of each step, for quick checks
    if d.get("limit_per_step"):
        count, kept = {}, []
        for s in samples:
            count[s.step] = count.get(s.step, 0) + 1
            if count[s.step] <= d["limit_per_step"]:
                kept.append(s)
        samples = kept
    return samples


def main():
    """Run the reader over the selected images, write per-image predictions and a per-step summary."""
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    run = new_run(cfg_path, cfg["name"])
    print("run folder:", run, flush=True)

    samples = select(cfg)
    m = cfg["model"]
    # slot 2: an optional LoRA adapter, stored relative to DEFOCUS_ROOT like the weights
    lora = defocus_root() / m["lora"] if m.get("lora") else None
    reader = QwenReader(defocus_root() / m["path"], dtype=m["dtype"], min_pixels=m["min_pixels"],
                        max_pixels=m["max_pixels"], max_new_tokens=m["max_new_tokens"],
                        prompt=m["prompt"], device=m.get("device", "cuda"), lora=lora)
    # slot 1: an optional image-processing step, None when the config has `restore: null`
    restorer = build_restorer(cfg.get("restore"))
    camera = cfg["data"]["camera"]
    bs = cfg["batch_size"]
    rows, t0 = [], time.time()

    def prepare(s):
        """Turn one sample's photograph into the image the VLM reads.

        Args:
            s: a Sample.

        Returns:
            PIL image.
        """
        # read the chosen camera's photograph as floats, paper about 1
        a = load_tif(getattr(s, camera), cfg["image"]["white_percentile"])
        # slot 1, skipped when empty
        if restorer is not None:
            a = restorer(a, s)
        # convert to the VLM's input format
        return to_pil(a, cfg["image"]["long_side"])

    for i in range(0, len(samples), bs):
        batch = samples[i:i + bs]
        images = [prepare(s) for s in batch]
        preds = reader.read(images)
        # score each transcription against the ground truth
        for s, p in zip(batch, preds):
            rows.append({"step": s.step, "font": s.font, "number": s.number,
                         "file": getattr(s, camera).name, "truth": s.text, "prediction": p,
                         "accuracy": round(char_accuracy(p, s.text, cfg["keep_whitespace"]), 4)})
        print(f"{len(rows)}/{len(samples)} images, {time.time() - t0:.0f} s", flush=True)

    # per-image results
    with open(run / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # per-step means with confidence intervals
    summary = summarise(rows)
    write_summary(summary, run / "summary.csv")
    for r in summary:
        print(f"step {r['step']:2d}: {r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] (n={r['n']})")


if __name__ == "__main__":
    main()
