# PFPA: target-gradient block selection

Code release for the PFPA encoder-block selection and adaptation experiments described in the associated manuscript. The package contains the experiment scripts and model-block definitions. It does not contain training data, image files, manifests with private paths, pretrained checkpoints, adapted checkpoints, or a self-contained copy of the original dataset/model repository.

## What is included

- SegFormer-B0 candidate-block plasticity scoring and one-block update scripts.
- A frozen-protocol PFPA adaptation runner for the SegFormer-B0 configuration.
- Swin-T + UPerNet block definitions and the exhaustive 12-block intervention runner.
- A protocol summary, dependency list, and manifest schema.

The scripts are based on the retained experiment implementation. Server-specific absolute paths and GPU index assumptions have been replaced with environment-based paths. The algorithm settings and tensor operations have not been changed as part of packaging.

## Environment

Use Linux with Python 3.10–3.12 and an NVIDIA GPU with a CUDA-enabled PyTorch installation. Install a `torch`/`torchvision` pair that matches the machine's CUDA runtime, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching CUDA build of torch and torchvision for your host first.
python -m pip install -e .
```

The exact package versions used in the original remote experiment environment were not present in the retained logs. The dependency ranges in `pyproject.toml` describe API compatibility bounds, not a claim that every version combination has been benchmarked. For an exact environment reproduction, record the installed versions and CUDA driver alongside the run.

## Data and model inputs

No dataset or model weights are included. Obtain each dataset and checkpoint under its own access terms. The scripts consume CSV manifests with these columns:

| Column | Meaning |
|---|---|
| `sample_id` | Stable, unique image identifier. |
| `image` | Absolute path to an RGB image. |
| `mask` | Absolute path to a semantic mask for labeled source/evaluation manifests; omit or leave empty for unlabeled target calibration manifests. |

Masks use class IDs `0=background`, `1=crop`, `2=weed`; `255` is ignored. Images and masks are resized to the grid used by each runner. Relative manifest paths resolve from `PFPA_DATA_ROOT`.

The local Hugging Face MiT-B0 model directory must be supplied for SegFormer-B0, with network downloads disabled by the script. The source-only checkpoint must match the architecture and be accompanied by its provenance JSON where the frozen-protocol runner requests it. Swin-T runs require the compatible source checkpoint at `PFPA_SWIN_SOURCE_CHECKPOINT`.

## Run SegFormer-B0 candidate scoring

Set `PFPA_DATA_ROOT` to the directory containing your manifests and local model assets. Provide paths explicitly so the run does not depend on this repository's location:

```bash
export PFPA_DATA_ROOT=/path/to/experiment-data
pfpa-scan-b0 \
  --checkpoint /path/to/source_only_checkpoint.pt \
  --pretrained /path/to/local/mit-b0 \
  --source-train /path/to/manifests/source_train.csv \
  --source-val /path/to/manifests/source_val.csv \
  --target-train /path/to/manifests/target_unlabeled.csv \
  --target-val /path/to/manifests/target_validation.csv \
  --output /path/to/a/new/output-directory \
  --seed 3407 --calibration-images 128 --source-gradient-images 128
```

To run the function-preserving update utility, install the proxy-score CSV produced by the scoring step and pass it as `--proxy-matrix` to `pfpa-update-b0`. The confirmatory runner `pfpa-confirm-b0 --help` lists the required frozen-protocol, provenance and manifest inputs. It refuses to overwrite existing run artifacts.

## Run the Swin-T exhaustive intervention scan

The Swin runner reads manifests from the standard names below under `PFPA_DATA_ROOT`:

```text
manifests/phenobench_train.csv
manifests/cropandweed_sugar_beet_train_unlabeled.csv
manifests/cropandweed_sugar_beet_val.csv
manifests/cropandweed_maize_train_unlabeled.csv
manifests/cropandweed_maize_val.csv
manifests/weedsgalore_train_unlabeled.csv
manifests/weedsgalore_val.csv
```

Then set a new output path, checkpoint path, and visible physical GPU before running:

```bash
export PFPA_DATA_ROOT=/path/to/experiment-data
export PFPA_OUTPUT_ROOT=/path/to/a/new/output-directory
export PFPA_SWIN_SOURCE_CHECKPOINT=/path/to/swin_source_checkpoint.pt
export PFPA_VISIBLE_GPU=0
pfpa-scan-swin
```

This runner computes and freezes all 108 block deltas before it reads the target-validation masks for post-hoc descriptive evaluation. Its evaluation is on the target-validation split used in the manuscript's development protocol; it is not an untouched blind-test procedure. Use a new empty output directory: the runner intentionally stops rather than overwrite prior artifacts.

## Reproducibility and scope

The package produces score tables, diagnostics, and adapted model checkpoints. Checkpoint files are runtime outputs and should not be committed to GitHub. The source code does not fetch data, download weights, or silently infer local paths beyond the documented `PFPA_DATA_ROOT` convention. Dataset acquisition, source-model training and checkpoint creation are outside this package; provide compatible inputs before running.

This code package is separate from the manuscript submission source ZIP. It is not supplementary material and contains no experimental result archive.

## License

This project is distributed under the MIT License. See `LICENSE` for the complete terms.
