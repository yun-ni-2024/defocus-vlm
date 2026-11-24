# Tasks: physics-based deblurring

This part of the project builds method (b): estimate the PSF of a blurred photograph from the photograph itself, deconvolve it, and let the frozen VLM read the result. It also measures the camera's PSFs from the calibration targets, which gives a reference with the correct PSF.

Read first: `pipeline.md` (how a method plugs into the evaluation), `data.md` (the dataset), `cluster.md` (environment, shared data, running jobs).

## Rules

1. **At test time, a method may use only the photograph.** It must not use the step, the file name or the calibration targets. The only exception is task 4, which is a reference, not a method.
2. **Tune only on the training set.** Every parameter (Wiener regularisation, ADMM weights and iterations, anything inside the radius estimator) is chosen on training images. The test set is used only for the final runs.
3. **PSF model.** The PSF of radius r is a uniform disc: the same value inside a circle of radius r, 0 outside, summing to 1; a pixel on the edge takes the fraction of its area that lies inside the circle. r is measured in pixels of the full-resolution image (2360 × 1460, i.e. an array of shape (1460, 2360)), from the centre of the disc to its edge. Every radius in this project, measured or estimated, follows this definition, so that the same number means the same PSF everywhere.
4. **Code layout.** `src/` holds reusable functions; `scripts/` holds the command-line programs that call them.

## Task 1. PSF calibration

Measure the disc radius of every defocus step from the calibration targets of the **training set**, blurred camera (`train/step<N>/<font>/CAM02/focusStep_<N>_PSF.tif`, `_LSF_X.tif`, `_LSF_Y.tif`).

The code goes in `src/psf/` and `scripts/`, organised as you like. The deliverable is one file:

**`assets/radius_table.json`** in the repository (committed, so everyone gets it with `git pull`):

```json
{
  "units": "pixels of the 2360 x 1460 image",
  "source": "training set, blurred camera (CAM02), calibration targets",
  "radius": {"0": <float>, "1": <float>, "2": <float>, ..., "19": <float>}
}
```

- Each step has calibration targets in both the `Times` and the `Verdana` folder. For some steps these are two separate photographs, for others the same file twice. The table needs one radius per step; combine them as you see fit (for example, average the two).
- All 20 steps must have a value. If a step cannot be measured well, still give your best value.
- Keys are the steps as strings (JSON allows only string keys).

## Task 2. Radius estimator

Estimate the disc radius from a single blurred photograph. The method is open: for example statistics of edges in the image, or comparing deconvolutions at different radii, etc.

The code goes in `src/psf/`, organised as you like. The deliverable is one function:

```python
from src.psf import estimate_radius

r = estimate_radius(a)
```

- `a`: a blurred photograph as returned by `load_tif()` (NumPy float array of shape (1460, 2360), paper about 1).
- `r`: the estimated radius (float), following the PSF definition in rule 3.
- It uses nothing but `a`, gives the same result every time, and takes at most a few seconds per image.

Why accuracy matters: in the pilot, a radius about 15% too small lost most of the benefit of deconvolution.

## Task 3. Method (b): four deconvolution methods with the estimated radius

This task has two parts: implement four deconvolution methods (3.1), then run each on the test set with the radius estimated by task 2 (3.2). The methods:

- Wiener filter
- ADMM with a TV prior
- plug-and-play with a DRUNet denoiser (DPIR)
- Reconstruct Anything Model (RAM)

### 3.1. Implementation

Each method is a module in `src/restore/` that `scripts/evaluate.py` applies to every photograph before the VLM reads it (see `pipeline.md`, "Slot 1").

Each method is one file `src/restore/<name>.py`, because a config selects it by that name (`restore: {name: <name>}` loads `src/restore/<name>.py`). The names are up to you; shared helpers can go in other files under `src/restore/` or `src/psf/`.

PnP and RAM use pretrained weights from the `deepinv` library as they are; nothing is trained on our data. Add `deepinv` with a fixed version to `requirements.txt`; its weights are downloaded into `$DEFOCUS_ROOT/models/` (`slurm/gpu.sbatch` sets `TORCH_HOME` and `HF_HOME` there).

Interface of every module (see `src/restore/__init__.py` and `src/restore/identity.py` for an example):

```python
def build(radius="estimated", **params):
    """radius: "estimated" (call estimate_radius(a)) or "calibrated" (task 4: look up
    sample.step in assets/radius_table.json, a path relative to the repository root).
    params: the method's own settings, e.g. regularisation weight, iterations, device."""
    def restore(a, sample):
        # a: NumPy float array from load_tif(), shape (1460, 2360), paper about 1
        # sample: the Sample; only sample.step may be read, and only when radius == "calibrated"
        # return: NumPy float array of the same shape, on the CPU, paper about 1
        #         (to_pil() clips it to [0, 1]); convert back if you compute with torch
        ...
    return restore
```

The `calibrated` branch needs the table from task 1.

Deliverable: the four modules.

### 3.2. Experiments

A config holds the settings shared by all methods (data, model, prompt) and, under `restore`, which module to plug in and its own parameters, which are passed to its `build()`.

Write one config per method, `configs/b_<name>.yaml`: a copy of `baseline_a.yaml` with `name` set to the file name (e.g. `name: b_wiener` in `b_wiener.yaml`) and `restore: {name: <name>, radius: estimated, ...}`. Everything else stays as in `baseline_a.yaml`, so the results compare directly with method (a).

Run a config on a GPU node with:

```bash
sbatch -J b_wiener -o ~/csc2529/logs/%x_%j.out slurm/gpu.sbatch scripts/evaluate.py configs/b_wiener.yaml
```

Each run writes its results to a new folder `$DEFOCUS_ROOT/runs/<name>_<time>/` (see `pipeline.md`). Which images are used is set in the config's `data` section (`split`, `steps`, `limit_per_step`).

Deconvolution runs on the full-resolution image before it is shrunk for the VLM, so ADMM and PnP can be slow; run them on the GPU.

Deliverables: the four configs, and the final test-set run of each, copied into the repository as `assets/results/<config name>/` (the whole run folder: `config.yaml`, `meta.json`, `predictions.csv`, `summary.csv`; for example `assets/results/b_wiener/`). If a final run is redone, replace the folder.

A final run uses committed code (its `meta.json` shows `"dirty": false`) and all 800 test images (`steps: null`, `limit_per_step: null`).

## Task 4. Reference: the same four methods with the calibrated (not estimated) radius

Run the four methods of task 3 again, with the radius taken from the radius table of task 1 for the photograph's step. This uses information no test-time method has, so it is a reference: the accuracy each method could reach if its PSF were exactly right. Comparing it with task 3 shows how much the radius estimation costs.

No new code: only configs, one per method, `configs/b_<name>_calibrated.yaml`, as in task 3 with `radius: calibrated`.

Deliverables: the four configs, and the final test-set run of each in `assets/results/<config name>/`, as in task 3.
