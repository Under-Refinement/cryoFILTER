# Publication contamination classifier

The UI, CryoSPARC typing, and `cryoFILTER type` use the frozen publication subtype classifier by default. It predicts Carbon, Crystalline, Aggregate, and Ethane within the supplied binary contamination mask. The previous heuristic remains available explicitly with `--classifier heuristic`.

The released classifier corresponds to `FINAL_top76_componentpost_aeexpert`, selected by the publication figure pipeline. It uses frozen FULL decoder embeddings, real-space/Fourier and geometry features, hierarchical logistic regressions, a second patch scale, an Aggregate/Ethane expert, and component-aware spatial aggregation. The exact learned regressions are stored in JSON, avoiding dependencies on Python pickle class locations or scikit-learn serialization versions.

## Installation

Download `classifier.pt` (229 MB) from [Zenodo record 22700873](https://zenodo.org/records/22700873), alongside the `cryoFILTER_FULL.pt` segmentation weights. From your cryoFILTER checkout:

```bash
mkdir -p pretrained_models &&
curl --fail --location --retry 3 \
  'https://zenodo.org/records/22700873/files/classifier.pt?download=1' \
  --output pretrained_models/classifier.pt
```

Git LFS is optional. The UI, CryoSPARC typing, and CLI automatically use `pretrained_models/classifier.pt` in the checkout, even when launched from another directory. The small learned regression models and configuration JSON files ship with cryoFILTER. Model loading validates checksums and fails clearly if weights are missing or incorrect; it does not silently switch to heuristic typing.

Weights are resolved in this order:

1. An explicit `--typing-checkpoint` path, or the Python API's `checkpoint` argument.
2. `CRYOFILTER_CLASSIFIER_CHECKPOINT`, if set. Both explicit options accept a file or a directory containing `classifier.pt`.
3. `classifier.pt` in `CRYOFILTER_WEIGHTS_DIR`. CryoSPARC supplies the selected segmentation checkpoint's directory automatically unless this variable is already set.
4. The typing manifest's directory and its `pretrained_models/` subdirectory (CLI), followed by the current working directory and the installed checkout. In each pair, `pretrained_models/classifier.pt` takes precedence over `classifier.pt` directly in that directory.
5. Packaged `cryofilter/data/publication/classifier.pt`, then the legacy `cryofilter/data/publication/encoder.pt`.

These are direct file checks, not recursive directory scans. For an arbitrary shared weights folder, set `CRYOFILTER_CLASSIFIER_CHECKPOINT=/absolute/path/to/classifier.pt` before launching the UI. Live and final CryoSPARC typing use the same discovery rules. Missing explicit paths fail instead of silently selecting another model.

Existing Git LFS installs can continue using `git lfs pull --include="cryofilter/data/publication/encoder.pt"`. An unmaterialized LFS pointer does not prevent a downloaded `classifier.pt` from being used.

The Zenodo `classifier.pt` is byte-identical to the legacy released `encoder.pt`. It contains exactly the publication feature-extractor tensors with training optimizer data removed. Its SHA256 is `e76443ced11f442878a410b2a3a01279dc0040c48ed36862642301d88e817a6f`. The JSON manifest records the remaining artifact hashes under their original package filenames and the original checkpoint provenance.

The segmentation checkpoint selected in the app controls the binary mask. Subtype feature extraction always uses its own matched publication weights. A different segmentation model can change which contamination pixels are captured, so its combined performance needs separate evaluation.

## Usage

```bash
cryoFILTER type --manifest typing_manifest.csv --output-dir typing \
  --incremental --typing-device auto --workers 4
```

The manifest needs `dataset_id`, `stem`, micrograph and binary-mask paths, and a valid pixel size from the manifest, override, or MRC header. Micrograph and mask rasters must align. Inputs are resampled to 2 Å/px and normalized with the publication's `percentile_extra_wide` recipe. Typed masks are returned at the supplied image size without labeling clean pixels. Tiny native pixels lost during resampling remain explicitly unclassified and are counted in `unclassified_area_px`.

`--typing-device` accepts `auto`, `cpu`, `cuda`, or `cuda:N`. `auto` uses CUDA when available. `--typing-batch-size` defaults to 16 and is reduced if GPU memory is exhausted; choose a smaller value when sharing a GPU. `--workers` controls CPU threads for publication inference. `--typing-checkpoint` can specify another location for the exact released `classifier.pt`, with checksum verification. `summary.json` records the resolved `classifier_checkpoint` path.

Per-image predictions are cached during live typing. Completed images are reused, modified inputs invalidate their predictions, and repeated finalization does not rewrite unchanged masks. The UI and CryoSPARC retain their background typing flow. This classifier performs more computation than the old heuristic; GPU inference is recommended.

The image and dataset summaries count actual subtype pixels, including mixed-type binary components. `component_type_assignments.csv` reports a majority subtype per binary component and also includes per-type pixel areas. Confidence values describe aggregation scores before the final geometry correction and are not calibrated correctness probabilities.

## Validation scope

Fresh inference with the distributed model reproduces **80.0851815% accuracy**, **80.7016057% balanced accuracy**, and **76.6957455% macro F1**, across 158 images and 18,440,384 captured, labeled contamination pixels. All predicted masks match the frozen publication predictions exactly, with zero differing pixels across 102,706 evaluated patch centers. This metric excludes clean and unlabeled pixels and contamination missed by the binary mask. Aggregate F1 is 54.41%; pooled accuracy alone does not describe every class equally well.

The [release validation record](../cryofilter/data/publication/validation.json) includes the exact confusion matrix, per-class F1 scores, and evaluation scope. The labeled validation snapshot is fixed; the older session paths retained in its original manifest must not be used as labels. No retraining or threshold tuning on the validation labels is part of this release conversion.

<!-- UI updates (classifier) -->
