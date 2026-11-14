# HDC2021 dataset

The Helsinki Deblur Challenge 2021 (Juvonen et al., 2023) photographed printed text with two cameras at the same time through a beam splitter: one camera stays in focus (`CAM01`), the other is defocused by a chosen amount (`CAM02`). The amount is the **defocus step**, from 0 (almost sharp) to 19 (heaviest blur). Licence: CC BY 4.0. How to download it and where it lives on the cluster: `cluster.md`, section 5.

## Text images

Each text image shows three lines of random characters, about ten per line. Random characters cannot be guessed from language, so every character the model invents counts as an error.

| | Training set | Test set |
|---|---|---|
| text images per step and camera | 100 per font, 200 in total | 20 per font, 40 in total |
| fonts | Times, Verdana (separate folders) | Times (`timesR`), Verdana (`verdanaRef`), mixed in one folder |
| characters | letters only (no transcription contains a digit) | letters and digits (most lines contain at least one digit) |
| other images per step and camera | none | 15 natural photos, 1 QR code |

Every text image has a `.txt` file with the same name holding the correct transcription. Sharp and blurred images of the same scene have the same file name in the two camera folders.

## Calibration targets

Every step, in both cameras, also has three targets for measuring the PSF:

- `PSF.tif`: a small point of light; its blurred image is the PSF itself;
- `LSF_X.tif`, `LSF_Y.tif`: a thin line in each direction; their blurred images are line spread functions, from which the size of the PSF can be measured.

## File names

```
focusStep_<step>_<font>_size_30_sample_<number>.tif     text image
focusStep_<step>_<font>_size_30_sample_<number>.txt     its transcription
focusStep_<step>_PSF.tif / _LSF_X.tif / _LSF_Y.tif      calibration targets
focusStep_<step>_Image_<name>.tif, _QRcode.tif           test set only
```

The images are 16-bit greyscale TIFF files.

## Things to keep in mind

- The training set has no digits but the test set does, so a model trained on the training set alone has never seen a blurred digit.
- In the training set, the sharp and blurred images of each step are in separate font folders (`step3/Times/CAM01`); in the test set, five steps share one folder (`steps_5_to_9/CAM01_focused`) and the step is only in the file name.
