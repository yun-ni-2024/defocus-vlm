"""Train a LoRA adapter for the vision encoder of Qwen2-VL (method (c)) by feature distillation.

Usage (environment activated, on a GPU node; normally through slurm/gpu.sbatch):
  python scripts/train_lora.py configs/train_c_real.yaml
Writes runs/<name>_<time>/ with config.yaml, meta.json, log.csv, val/ and adapters/.
The chosen adapter is then set as model.lora in the evaluation config (e.g. configs/c_real.yaml)
and evaluated with scripts/evaluate.py.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.adapt.train import train  # noqa: E402
from src.utils.run import REPO, load_config, new_run  # noqa: E402

# model settings that must be the same at training and at test time
SHARED_MODEL_KEYS = ["path", "dtype", "min_pixels", "max_pixels", "max_new_tokens", "prompt"]


def check_matches_evaluation(cfg):
    """Stop if the model or image settings differ from those of the evaluation.

    The adapter must see images processed exactly as they are at test time, so the training
    config's model and image sections must equal those of configs/baseline_a.yaml.

    Args:
        cfg: the training config.

    Raises:
        ValueError: naming the first setting that differs.
    """
    ref = load_config(REPO / "configs" / "baseline_a.yaml")
    for k in SHARED_MODEL_KEYS:
        if cfg["model"].get(k) != ref["model"][k]:
            raise ValueError(f"model.{k} differs from configs/baseline_a.yaml")
    if cfg["image"] != ref["image"]:
        raise ValueError("the image section differs from configs/baseline_a.yaml")


def main():
    """Check the config, create the run folder and train."""
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    check_matches_evaluation(cfg)
    run = new_run(cfg_path, cfg["name"])
    print("run folder:", run, flush=True)
    train(cfg, run)


if __name__ == "__main__":
    main()
