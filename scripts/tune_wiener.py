"""Generate training-only smoke, sweep, or confirmation configs for Wiener."""
import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GRID = (0.0003, 0.001, 0.003, 0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "sweep", "confirm"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--regularization", type=float, nargs="+")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--include-baseline", action="store_true",
                        help="also generate a no-restoration config for the same photos")
    args = parser.parse_args()
    if args.phase == "confirm" and not args.regularization:
        parser.error("confirm requires the shortlisted --regularization values")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    values = args.regularization or ((0.003,) if args.phase == "smoke" else DEFAULT_GRID)
    if any(not 0 < value < float("inf") for value in values):
        parser.error("--regularization values must be finite and positive")

    args.output.mkdir(parents=True, exist_ok=True)
    for value in dict.fromkeys(values):
        config = yaml.safe_load((REPO / "configs/b_wiener.yaml").read_text())
        config["name"] = f"b_wiener_{args.phase}_k{value:g}"
        config["restore"]["regularization"] = value
        config["batch_size"] = args.batch_size
        config["data"].update(
            split="train", steps=[0, 7, 19] if args.phase == "smoke" else None,
            sample_numbers=[2, 3, 4] if args.phase == "confirm" else [1],
        )
        if args.model_path is not None:
            config["model"]["path"] = str(args.model_path.resolve())
        path = args.output / f"{config['name']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path)

    if args.include_baseline:
        config["name"] = f"b_wiener_{args.phase}_baseline"
        config["restore"] = None
        path = args.output / f"{config['name']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
