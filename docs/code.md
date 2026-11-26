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
| `restore/learned_wiener.py` | Slot 1 for method (d): loads a trained filter (`params/it<N>.json`), estimates the radius of each photograph and deconvolves it with `frontend/wiener.py`. |
| `frontend/wiener.py` | Method (d): `LearnedWiener`, a Wiener filter with a disc PSF of the estimated radius, a learned regularisation K(f; r) and a learned radius correction s(r), stored as two small tables; saved and loaded as JSON. |
| `frontend/vlm_input.py` | Method (d): differentiable version of `to_pil()` and the Qwen2-VL processor (`to_vlm_size`, `pixel_values`, `visual_tokens`), so gradients reach the filter. |
| `frontend/pairs.py` | Method (d): the training/validation split (same rule as method (c)), the radius cache, and a loader of full-resolution pairs with their estimated radius. |
| `frontend/train.py` | Method (d): the two losses (`feature_loss`, `pixel_loss`), validation through the evaluation path, and the training loop. |
| `frontend/checks.py` | Method (d): correctness checks run before training (split, filter, slot module, input path, gradients). |
| `utils/run.py` | `load_config` reads a YAML config; `new_run` creates the run folder and records the config, machine, Slurm job and git commit; `defocus_root` finds `$DEFOCUS_ROOT`. |

## `scripts/`: entry points

| File | What it does |
|---|---|
| `evaluate.py` | Runs the evaluation pipeline for one config and writes the run folder. Used for every method. |
| `check_data.py` | Checks that the dataset is complete: 40 test pairs and 200 training pairs per step, all images readable. |
| `download_hdc.py` | Downloads HDC2021 from Zenodo, checks each file's MD5, extracts it and deletes the zip. Resumes after an interruption. |
| `estimate_radii.py` | Estimates the radius of every blurred photograph of a split once (several processes) and writes a cache file under `DEFOCUS_ROOT`, used by method (d). |
| `train_wiener.py` | Trains the learnable Wiener filter of method (d) from a training config and writes the run folder (`log.csv`, `val/`, `params/`). |
| `check_frontend.py` | Runs the method (d) checks of `frontend/checks.py`. |

## `configs/`: one file per experiment

| File | Experiment |
|---|---|
| `baseline_a.yaml` | Method (a): blurred test photographs, both slots empty. Changing `camera` to `sharp` gives the sharp reference. |
| `train_d_feature.yaml`, `train_d_pixel.yaml` | Training of method (d) with the VLM-feature loss and with the pixel loss; identical except for `loss`. |
| `d_feature.yaml`, `d_pixel.yaml` | Evaluation of method (d): `baseline_a.yaml` with slot 1 set to the trained filter in `assets/d_<loss>/params.json`. |

## `slurm/`: job templates

| File | What it does |
|---|---|
| `gpu.sbatch` | Runs any Python script with its arguments on a GPU node, inside the project environment. |
| `download_hdc.sbatch` | Runs the data download on a CPU node. |

## `runpod/`: Runpod pods

| File | What it does |
|---|---|
| `run.sh` | The counterpart of `gpu.sbatch` on a Runpod pod: activates the environment and starts any Python script in the background, with its output in `logs/`. |

## Other files

| File | What it is |
|---|---|
| `requirements.txt` | Python packages with fixed versions. |
| `docs/` | Documentation: `cluster.md`, `runpod.md`, `data.md`, `pipeline.md`, `evaluation.md`, `code.md`, `findings.md`. |
