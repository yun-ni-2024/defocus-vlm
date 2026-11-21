# Data

We use the Helsinki Deblur Challenge 2021 (HDC2021) dataset (CC BY 4.0). How to download it and where it is on the cluster: `cluster.md`, section 5.

## What the data is

Two cameras photograph the same printed sheet at the same time through a beam splitter:

- **CAM01** is always in focus and gives the **sharp** image.
- **CAM02** is deliberately out of focus and gives the **blurred** image. How far it is out of focus is the **defocus step**, from 0 (almost sharp) to 19 (most blurred).

So every scene comes as an aligned pair: a sharp image and a blurred image of the same sheet. Each step was shot with its own new sheets, so different steps show different text.

Every image is a 2360 × 1460, 16-bit greyscale TIFF.

There are three kinds of images:

- **Text images.** Three lines of random characters, about ten per line, in Times or Verdana. Each comes with a `.txt` file holding the correct transcription. These are what the VLM reads.
- **Calibration targets.** Photos of a small black dot (`PSF.tif`) and of a thin horizontal and vertical line (`LSF_X.tif`, `LSF_Y.tif`). Their blurred images show the camera's point spread function (PSF) at that step.
- **Other images** (test set only): natural photos and a QR code. We do not use them.

## How much data there is

| | Training set | Test set |
|---|---|---|
| Defocus steps | 0 to 19 | 0 to 19 |
| Text image pairs per step | 200 (100 Times + 100 Verdana) | 40 (20 Times + 20 Verdana) |
| Text image pairs in total | 4,000 | 800 |
| Characters in the text | letters only | letters and digits |
| Calibration targets per step and camera | 3 | 3 |
| Other images per step and camera | none | 15 natural photos + 1 QR code |
| Image files in total | 8,240 | 2,360 |
| Size on disk | 54 GB | 16 GB |

How we use it:

- **Test set text images:** evaluation only. Every accuracy curve is computed on these 800 pairs. They are never used for training or tuning.
- **Training set text images:** training for methods (c) and (d), and any tuning.
- **Calibration targets:** measuring the PSF of each step, for the calibrated-PSF reference and the blur simulator of method (c). No method sees them at test time.

## How it is organised

Under `$DEFOCUS_ROOT/data/hdc2021/`:

```
train/
  step0/
    Times/
      CAM01/                                         sharp images
        focusStep_0_timesR_size_30_sample_0001.tif   text image
        focusStep_0_timesR_size_30_sample_0001.txt   its transcription
        ...
        focusStep_0_PSF.tif                          calibration targets
        focusStep_0_LSF_X.tif
        focusStep_0_LSF_Y.tif
      CAM02/                                         blurred images, same file names
    Verdana/
      CAM01/  CAM02/                                 file names use "verdanaRef"
  step1/ ... step19/
test/
  steps_0_to_4/
    CAM01_focused/                                   sharp images of steps 0 to 4
    CAM02_blurred/                                   blurred images, same file names
  steps_5_to_9/  steps_10_to_14/  steps_15_to_19/
```

- A sharp image and its blurred version have the same file name, in the CAM01 and CAM02 folders.
- The step is always in the file name (`focusStep_<step>_...`). In the test set, five steps share one folder, and both fonts are in the same folder.
- `src/data/hdc.py` hides these differences: `train_samples()` and `test_samples()` return one record per text image pair, with its step, font, the two file paths and the transcription.
