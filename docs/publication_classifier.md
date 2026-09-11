# Publication contamination classifier

The UI, CryoSPARC typing, and `cryoFILTER type` use the frozen publication subtype classifier by default. It predicts Carbon, Crystalline, Aggregate, and Ethane within the supplied binary contamination mask. The previous heuristic remains available explicitly with `--classifier heuristic`.

The released classifier corresponds to `FINAL_top76_componentpost_aeexpert`, selected by the publication figure pipeline. It uses frozen FULL decoder embeddings, real-space/Fourier and geometry features, hierarchical logistic regressions, a second patch scale, an Aggregate/Ethane expert, and component-aware spatial aggregation. The exact learned regressions are stored in JSON, avoiding dependencies on Python pickle class locations or scikit-learn serialization versions.

## Installation

The conda environment includes Git LFS. After cloning or updating:

```bash
conda activate cryofilter
git lfs pull --include="cryofilter/data/publication/encoder.pt"
python -m pip install -e .
```

For an existing pip environment, install Git LFS using your system's package manager and run the same pull command. The feature extractor is 229 MB. Model loading validates the artifact checksums and fails clearly if weights are missing or incorrect; it does not silently switch to heuristic typing.

You can also [download encoder.pt directly](https://media.githubusercontent.com/media/Under-Refinement/cryoFILTER/main/cryofilter/data/publication/encoder.pt) and pass its location with `--typing-checkpoint`. For the UI's default path, place it at `cryofilter/data/publication/encoder.pt` in the checkout.

The `encoder.pt` file contains exactly the publication feature-extractor tensors with training optimizer data removed. Its SHA256 is `6c6927165c23d89daa4168aeaa8a32606628346d6418bac9125da6f1a08141b1`. The JSON manifest records the remaining artifact hashes and original checkpoint provenance.

The segmentation checkpoint selected in the app controls the binary mask. Subtype feature extraction always uses its own matched publication weights. A different segmentation model can change which contamination pixels are captured, so its combined performance needs separate evaluation.

## Usage

```bash
cryoFILTER type --manifest typing_manifest.csv --output-dir typing \
  --incremental --typing-device auto --workers 4
```

The manifest needs `dataset_id`, `stem`, micrograph and binary-mask paths, and a valid pixel size from the manifest, override, or MRC header. Micrograph and mask rasters must align. Inputs are resampled to 2 Å/px and normalized with the publication's `percentile_extra_wide` recipe. Typed masks are returned at the supplied image size without labeling clean pixels. Tiny native pixels lost during resampling remain explicitly unclassified and are counted in `unclassified_area_px`.

`--typing-device` accepts `auto`, `cpu`, `cuda`, or `cuda:N`. `auto` uses CUDA when available. `--typing-batch-size` defaults to 16 and is reduced if GPU memory is exhausted; choose a smaller value when sharing a GPU. `--workers` controls CPU threads for publication inference. `--typing-checkpoint` can specify another location for the exact released `encoder.pt`, with checksum verification.

Per-image predictions are cached during live typing. Completed images are reused, modified inputs invalidate their predictions, and repeated finalization does not rewrite unchanged masks. The UI and CryoSPARC retain their background typing flow. This classifier performs more computation than the old heuristic; GPU inference is recommended.

The image and dataset summaries count actual subtype pixels, including mixed-type binary components. `component_type_assignments.csv` reports a majority subtype per binary component and also includes per-type pixel areas. Confidence values describe aggregation scores before the final geometry correction and are not calibrated correctness probabilities.

## Validation scope

Fresh inference with the distributed model reproduces **80.0851815% accuracy**, **80.7016057% balanced accuracy**, and **76.6957455% macro F1**, across 158 images and 18,440,384 captured, labeled contamination pixels. All predicted masks match the frozen publication predictions exactly, with zero differing pixels across 102,706 evaluated patch centers. This metric excludes clean and unlabeled pixels and contamination missed by the binary mask. Aggregate F1 is 54.41%; pooled accuracy alone does not describe every class equally well.

The [release validation record](../cryofilter/data/publication/validation.json) includes the exact confusion matrix, per-class F1 scores, and evaluation scope. The labeled validation snapshot is fixed; the older session paths retained in its original manifest must not be used as labels. No retraining or threshold tuning on the validation labels is part of this release conversion.
