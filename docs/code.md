# Code

What each file does. How they fit together: `pipeline.md`.

## `src/`: the modules

| File | What it does |
|---|---|
| `data/hdc.py` | Lists the HDC2021 text images. `test_samples(steps)` and `train_samples(steps)` return one `Sample` per sharp/blurred pair, with its split, step, font, number, the two file paths and the ground-truth text. Reads no pixels. |
| `data/image.py` | `load_tif(path)` reads a 16-bit TIFF as a float image with the white paper at about 1. `to_pil(a)` turns such an image into the 8-bit RGB image the VLM takes, shrunk to a long side of 1024. Slot 1 sits between the two. |
| `vlm/qwen.py` | `QwenReader` loads Qwen2-VL-2B once (optionally with a LoRA adapter for slot 2); `read(images)` turns a list of images into a list of transcriptions. |
| `eval/metrics.py` | `char_accuracy(pred, truth)`: 1 minus the edit distance divided by the length of the ground truth, clipped at 0. |
| `eval/summary.py` | `summarise(rows)` averages the scores of each step and adds a bootstrap 95% confidence interval; `write_summary` saves it as CSV. |
| `restore/__init__.py` | Slot 1. `build_restorer(cfg)` loads the method named in the config's `restore` section from `src/restore/<name>.py`, or returns None for an empty slot. |
| `restore/identity.py` | A restorer that returns the image unchanged, used to check that the slot itself changes nothing. |
| `utils/run.py` | `load_config` reads a YAML config; `new_run` creates the run folder and records the config, machine, Slurm job and git commit; `defocus_root` finds `$DEFOCUS_ROOT`. |

## `scripts/`: entry points

| File | What it does |
|---|---|
| `evaluate.py` | Runs the evaluation pipeline for one config and writes the run folder. Used for every method. |
| `check_data.py` | Checks that the dataset is complete: 40 test pairs and 200 training pairs per step, all images readable. |
| `download_hdc.py` | Downloads HDC2021 from Zenodo, checks each file's MD5, extracts it and deletes the zip. Resumes after an interruption. |

## `configs/`: one file per experiment

| File | Experiment |
|---|---|
| `baseline_a.yaml` | Method (a): blurred test photographs, both slots empty. Changing `camera` to `sharp` gives the sharp reference. |

## `slurm/`: job templates

| File | What it does |
|---|---|
| `gpu.sbatch` | Runs any Python script with its arguments on a GPU node, inside the project environment. |
| `download_hdc.sbatch` | Runs the data download on a CPU node. |

## Other files

| File | What it is |
|---|---|
| `requirements.txt` | Python packages with fixed versions. |
| `docs/` | Documentation: `cluster.md`, `data.md`, `pipeline.md`, `evaluation.md`, `code.md`, `findings.md`. |
