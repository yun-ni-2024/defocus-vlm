# Code

What each file does. How they fit together: `pipeline.md`.

## `src/`: the modules

| File | What it does |
|---|---|
| `data/hdc.py` | Lists the HDC2021 text images. `test_samples(steps)` and `train_samples(steps)` return one `Sample` per sharp/blurred pair, with its split, step, font, number, the two file paths and the ground-truth text. Reads no pixels. |
| `data/image.py` | `load_tif(path)` reads a 16-bit TIFF as a float image with the white paper at about 1. `to_pil(a)` turns such an image into the 8-bit RGB image the VLM takes, shrunk to a long side of 1024. Slot 1 sits between the two. |
| `vlm/qwen.py` | `QwenReader` loads Qwen2-VL-2B once (optionally with a LoRA adapter for slot 2); `read(images)` turns a list of images into a list of transcriptions. |
| `vlm/vlm_input.py` | Differentiable version of `to_pil()` and the Qwen2-VL processor (`to_vlm_size`, `pixel_values`, `visual_tokens`), so gradients reach what comes before the VLM (method (d)). |
| `vlm/text_loss.py` | `TextLoss`: the cross-entropy of the correct transcription under the VLM (teacher forcing), for a restored image tensor (method (d)) or a PIL image (method (c) with the text loss). |
| `eval/metrics.py` | `char_accuracy(pred, truth)`: 1 minus the edit distance divided by the length of the ground truth, clipped at 0. |
| `eval/summary.py` | `summarise(rows)` averages the scores of each step and adds a bootstrap 95% confidence interval; `write_summary` saves it as CSV. |
| `eval/char_classes.py` | Marks each ground-truth character as read or not (`read_mask`) and counts the recall of digits and of letters per step (`counts_per_step`, `group_counts`). |
| `restore/__init__.py` | Slot 1. `build_restorer(cfg)` loads the method named in the config's `restore` section from `src/restore/<name>.py`, or returns None for an empty slot. |
| `restore/identity.py` | A restorer that returns the image unchanged, used to check that the slot itself changes nothing. |
| `restore/learned_wiener.py` | Slot 1 for method (d): loads a trained filter (`params/it<N>.json`), estimates the radius of each photograph and deconvolves it with `frontend/wiener.py`. |
| `frontend/wiener.py` | Method (d): `LearnedWiener`, a Wiener filter with a disc PSF of the estimated radius, a learned regularisation K(f; r) and a learned radius correction s(r), stored as two small tables; saved and loaded as JSON. |
| `frontend/pairs.py` | Method (d): the training/validation split (same rule as method (c)), the radius cache, and a loader of full-resolution pairs with their estimated radius and transcription. |
| `frontend/train.py` | Method (d): the losses (`feature_loss`, `pixel_loss`, and the text loss of `vlm/text_loss.py`), validation through the evaluation path, and the training loop. An optional `model.lora` trains the filter for a VLM with a fixed adapter (E2). |
| `frontend/checks.py` | Method (d): correctness checks run before training (split, filter, slot module, input path, gradients). |
| `adapt/distill.py` | Method (c): `add_lora` puts a LoRA adapter into the vision encoder; `distill_loss` is the feature-distillation loss (adapted encoder on the blurred photograph against the original encoder on the sharp one). |
| `adapt/pairs.py` | Method (c): the training/validation split and a loader of sharp/blurred pairs as the PIL images the VLM reads, with their transcription; an optional slot-1 restorer is applied to the blurred photograph first (E1). |
| `adapt/train.py` | Method (c): the training loop with the feature or the text loss, and validation through the evaluation path; saves the adapter at each validation. An optional `restore` section puts a fixed slot 1 before the VLM (E1). |
| `utils/run.py` | `load_config` reads a YAML config; `new_run` creates the run folder and records the config, machine, Slurm job and git commit; `defocus_root` finds `$DEFOCUS_ROOT`. |

## `scripts/`: entry points

| File | What it does |
|---|---|
| `evaluate.py` | Runs the evaluation pipeline for one config and writes the run folder. Used for every method. |
| `check_data.py` | Checks that the dataset is complete: 40 test pairs and 200 training pairs per step, all images readable. |
| `download_hdc.py` | Downloads HDC2021 from Zenodo, checks each file's MD5, extracts it and deletes the zip. Resumes after an interruption. |
| `estimate_radii.py` | Estimates the radius of every blurred photograph of a split once (several processes) and writes a cache file under `DEFOCUS_ROOT`, used by method (d). |
| `train_lora.py` | Trains the LoRA adapter of method (c) from a training config and writes the run folder (`log.csv`, `val/`, `adapters/`). |
| `train_wiener.py` | Trains the learnable Wiener filter of method (d) from a training config and writes the run folder (`log.csv`, `val/`, `params/`). |
| `check_frontend.py` | Runs the method (d) checks of `frontend/checks.py`. |
| `audit_losses.py` | For saved method (d) filters, computes the pixel, feature and text losses of every validation photograph and reports how well each agrees with reading accuracy. |
| `char_classes.py` | Prints the recall of digits and of letters for result folders, per step or per group of steps (`--groups 0-4,5-14,15-19`). No GPU needed. |

## `configs/`: one file per experiment

| File | Experiment |
|---|---|
| `baseline_a.yaml` | Method (a): blurred test photographs, both slots empty. Changing `camera` to `sharp` gives the sharp reference. |
| `train_d_feature.yaml`, `train_d_pixel.yaml` | Training of method (d) with the VLM-feature loss and with the pixel loss; identical except for `loss`. |
| `train_c_real*.yaml` | Training of method (c), real pairs, with feature distillation: the starting config and the tuning runs (learning rate, length, rank). |
| `train_c_real_text*.yaml` | Training of method (c), real pairs, with the text loss: three learning rates, otherwise as `train_c_real.yaml`. |
| `c_real.yaml` | Evaluation of method (c), real pairs: `baseline_a.yaml` with slot 2 set to the chosen adapter. |
| `c_real_text.yaml` | Evaluation of method (c), real pairs, trained with the text loss: `c_real.yaml` with slot 2 set to that adapter. |
| `e_cd_feature.yaml`, `e_cd_text.yaml` | Evaluation of E0, (c) and (d) combined without retraining: `d_text.yaml` with slot 2 set to the adapter of `c_real.yaml` or `c_real_text.yaml` (`docs/cd-combination.md`). |
| `train_e1_cfeature.yaml`, `train_e1_ctext.yaml` | Training of E1: a new adapter on the output of the fixed (d) filter, otherwise as the chosen (c) run with feature distillation or the text loss. |
| `train_e2_cfeature.yaml`, `train_e2_ctext.yaml` | Training of E2: a new filter through the VLM with the fixed adapter of `c_real.yaml` or `c_real_text.yaml`, otherwise as `train_d_text_2k.yaml`. |
| `train_d_text.yaml`, `train_d_text_2k.yaml` | Training of method (d) with the text loss, 8 pairs per step (4 batches accumulated), 500 and 2000 steps. |
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
