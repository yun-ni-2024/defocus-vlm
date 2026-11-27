"""Run the correctness checks of method (d) before training (src/frontend/checks.py).

Usage (environment activated, on a GPU; the radius cache of the training set must exist):
  python scripts/check_frontend.py configs/train_d_feature.yaml
Prints PASS or FAIL with the measured differences for each check; exits with status 1 if any fails.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.frontend import checks  # noqa: E402
from src.frontend.pairs import cache_key, load_radius_cache, split_pairs  # noqa: E402
from src.utils.run import defocus_root, load_config  # noqa: E402
from src.vlm.qwen import QwenReader  # noqa: E402

# validation photographs from a light, a medium and a heavy blur
CHECK_STEPS = (4, 10, 17)


def main():
    """Load the model and a few photographs, run every check and report."""
    cfg = load_config(sys.argv[1])
    m = cfg["model"]
    reader = QwenReader(defocus_root() / m["path"], dtype=m["dtype"], min_pixels=m["min_pixels"],
                        max_pixels=m["max_pixels"], max_new_tokens=m["max_new_tokens"],
                        prompt=m["prompt"], device=m.get("device", "cuda"))
    _, radii = load_radius_cache(cfg["data"]["radius_cache"])
    # the first validation photograph of each chosen step, with its cached radius
    _, val = split_pairs(cfg["data"].get("steps"), cfg["data"]["val_every"])
    samples = [next(s for s in val if s.step == k) for k in CHECK_STEPS]
    r_hat = [radii[cache_key(s.blurred)] for s in samples]
    heavy, r_heavy = samples[-1], r_hat[-1]

    results = [
        ("split", checks.check_split(cfg)),
        ("wiener", checks.check_wiener(cfg, heavy, r_heavy, reader.device)),
        ("restore", checks.check_restore(cfg, heavy, r_heavy, reader.device)),
        ("vlm_input", checks.check_vlm_input(cfg, samples, reader)),
        ("gradients", checks.check_gradients(cfg, heavy, r_heavy, reader)),
    ]
    for name, (ok, msg) in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}", flush=True)
    sys.exit(0 if all(ok for _, (ok, _) in results) else 1)


if __name__ == "__main__":
    main()
