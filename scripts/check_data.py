"""Check that the HDC2021 text images are complete: every step has the expected number of pairs.

Usage (environment activated, on a CPU node):
  python scripts/check_data.py
Expected: 40 test pairs and 200 training pairs per step, every image readable as a 1460 x 2360 array.
"""
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.hdc import data_root, test_samples, train_samples  # noqa: E402
from src.data.image import load_tif  # noqa: E402

EXPECTED = {"test": 40, "train": 200}


def main():
    """Count the pairs of each split and step, read one pair per step, and report problems.

    Returns:
        None. Exits with status 1 if anything is missing.
    """
    print("data folder:", data_root())
    ok = True
    for split, samples in (("test", test_samples()), ("train", train_samples())):
        # pairs per step and font
        per_step = Counter(s.step for s in samples)
        fonts = Counter(s.font for s in samples)
        print(f"{split}: {len(samples)} pairs, fonts {dict(fonts)}")
        for step in range(20):
            n = per_step.get(step, 0)
            if n != EXPECTED[split]:
                print(f"  step {step}: {n} pairs, expected {EXPECTED[split]}")
                ok = False
        # every sharp file must exist, and the first pair of each step must be readable
        missing = [s.sharp for s in samples if not s.sharp.exists()]
        if missing:
            print(f"  {len(missing)} sharp images missing, e.g. {missing[0]}")
            ok = False
        firsts = {s.step: s for s in reversed(samples)}
        for s in firsts.values():
            for path in (s.sharp, s.blurred):
                shape = load_tif(path).shape
                if shape != (1460, 2360):
                    print(f"  unexpected size {shape}: {path}")
                    ok = False
    print("all complete" if ok else "problems found")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
