# cryoFILTER CLI Reference

The local browser app is the recommended entry point for most users:

```bash
cryoFILTER app
```

Use the CLI for scripted runs, cluster wrappers, regression tests, and advanced CryoSPARC workflows. The parser output from your installed version is always the source of truth:

```bash
cryoFILTER --help
cryoFILTER app --help
cryofilter infer --help
cryofilter filter-particles --help
cryofilter type --help
cryofilter train --help
cryofilter gui --help
cryofilter cryosparc --help
```

## `cryoFILTER app`

Launch the browser wrapper.

```bash
cryoFILTER app [--host 127.0.0.1] [--port 8765] [--work-dir .]
```

Flags:

- `--host`: bind address. Keep `127.0.0.1` for local or SSH-tunneled use.
- `--port`: local app port.
- `--work-dir`: directory for `.cryofilter_app/`, relative outputs, and job logs.

## `cryofilter infer`

Run contamination inference on one `.mrc` file or a directory of `.mrc` files. Adding `--particle-file` also filters particles and writes OTF diagnostic PNGs by default.

Required:

- `--input`: input `.mrc` file or directory.

Common output and resource flags:

- `--output-dir`
- `--device`
- `--gpus`
- `--num-cpus`
- `--num-gpus`
- `--recursive`
- `--skip-existing`

Model and inference flags:

- `--checkpoint`
- `--model-type`
- `--attention-type`
- `--use-psd`
- `--norm-type`
- `--normalization-method`
- `--patch-size`
- `--overlap`
- `--batch-forward-size`
- `--threshold`
- `--temperature`
- `--logit-bias`
- `--use-tta`
- `--tta`
- `--inference-profile`

Pixel-size and resampling flags:

- `--pixel-size-angstrom`
- `--target-pixel-size`
- `--target-apix`
- `--adaptive-downsample`
- `--no-adaptive-downsample`
- `--adaptive-downsample-method`
- `--adaptive-max-factor`
- `--adaptive-min-work-dim`
- `--small-pixel-auto`
- `--no-small-pixel-auto`
- `--small-pixel-cutoff-angstrom`
- `--multi-scale-target-pixel-sizes`
- `--no-resample`
- `--no-resample-or-gen`

Stitching and postprocessing flags:

- `--multiscale-psd-source`
- `--psd-scales`
- `--psd-multiscale-separate-channels`
- `--no-psd-multiscale-separate-channels`
- `--blending-window`
- `--blending-edge-px`
- `--stitch-phase-mode`
- `--stitch-phase-reduction`
- `--stitch-phase-blend-alpha`
- `--stitch-output-space`
- `--stitch-output-blend-alpha`
- `--stitch-output-preserve-threshold`
- `--stitch-output-transition`
- `--patch-prob-floor-quantile`
- `--patch-prob-floor-strength`
- `--hysteresis-mode`
- `--hysteresis-low-delta`
- `--hysteresis-low-min`
- `--hysteresis-auto-max-growth-ratio`
- `--hysteresis-auto-min-added-mean-prob-margin`
- `--min-area-pixels`
- `--fill-small-holes-max-area-pixels`
- `--mask-buffer-distance-angstrom`
- `--fill-small-holes-after-buffer-max-area-pixels`

Particle filtering flags:

- `--particle-file`
- `--particles`
- `--filtered-particle-file`
- `--particle-output`
- `--particle-csg`
- `--particle-exclusion-distance-angstrom`
- `--allow-unmatched-particles`
- `--overwrite-filtered-particles`

OTF and diagnostic image flags:

- `--render-particle-overlays`
- `--no-render-particle-overlays`
- `--particle-overlay-dir`
- `--particle-overlay-max-display-dim`
- `--particle-overlay-diameter-px`
- `--particle-overlay-mask-alpha`
- `--particle-overlay-raw-panel`
- `--no-particle-overlay-raw-panel`
- `--particle-overlay-probability-panel`
- `--no-particle-overlay-probability-panel`
- `--particle-overlay-contact-sheet`
- `--no-particle-overlay-contact-sheet`
- `--particle-overlay-contact-sheet-cols`
- `--particle-overlay-contact-sheet-tile`
- `--particle-overlay-contact-sheet-every`
- `--particle-overlay-warn-removed-fraction`
- `--render-images`
- `--image-output-dir`
- `--image-max-dim`
- `--image-dpi`

## `cryofilter filter-particles`

Filter an existing CryoSPARC or RELION particle file using already-generated cryoFILTER masks.

Required:

- `--input`
- `--mask-dir`
- `--particle-file`

Flags:

- `--output-dir`
- `--filtered-particle-file`
- `--particle-output`
- `--particle-csg`
- `--particle-exclusion-distance-angstrom`
- `--pixel-size-angstrom`
- `--recursive`
- `--allow-unmatched-particles`
- `--overwrite-filtered-particles`
- `--summary-file`

## `cryofilter type`

Classify connected contamination components as Carbon, Crystalline, Aggregate, or Ethane.

Required:

- `--manifest`
- `--output-dir`

Flags:

- `--bands-json`
- `--normalization-method`
- `--pixel-size-angstrom`
- `--min-component-area-px`
- `--crop-pad-px`
- `--min-crop-size-px`
- `--max-psd-crop-size-px`

## `cryofilter train`

Lightweight patch-based FULL training and quick fine-tuning utility. For the recommended user adaptation recipe, prefer the app Fine-tune tab or `scripts/train_patch_based_fast.py`.

Required:

- `--manifest`
- `--output-dir`

Flags:

- `--initial-checkpoint`
- `--resume-checkpoint`
- `--epochs`
- `--batch-size`
- `--patch-size`
- `--patches-per-micrograph`
- `--validation-patches-per-micrograph`
- `--positive-patch-fraction`
- `--validation-fraction`
- `--learning-rate`
- `--weight-decay`
- `--positive-weight`
- `--dice-weight`
- `--early-stop-patience`
- `--num-workers`
- `--device`
- `--seed`
- `--dry-run`

## `cryofilter gui`

Desktop matplotlib mask-review fallback. Most users should use the browser-native Annotation tab instead.

Required:

- `--manifest`

Flags:

- `--output-dir`
- `--micrograph-dir`
- `--max-display-dim`

## `cryofilter cryosparc`

Advanced direct CryoSPARC integration CLI. Most users should use the app CryoSPARC tab.

Global flags:

- `--config`
- `--host`
- `--bridge-command`
- `--remote-source-root`
- `--remote-work-root`
- `--cryosparc-base-url`
- `--cryosparc-host`
- `--cryosparc-base-port`
- `--cryosparc-email`
- `--ssh-option`
- `--json`

Subcommands:

- `doctor`
- `test-transport`
- `deploy-bridge`
- `inspect`
- `test-roundtrip`
- `stage-test`
- `build-diagnostics`
- `attach-diagnostics`
- `predict`
- `finalize-run`
- `resume`
- `finalize-from-run`

Common CryoSPARC refs use `J123:output_name`; the app also accepts shorthand such as `J123` and expands common micrograph and particle outputs.

Useful `predict` flags:

- `--project`
- `--workspace`
- `--micrographs`
- `--particles`
- `--run-id`
- `--model-id`
- `--checkpoint`
- `--threshold`
- `--title`
- `--limit-micrographs`
- `--micrograph-path-field`
- `--local-run-root`
- `--timeout`
- `--inference-timeout`
- `--num-cpus`
- `--num-gpus`
- `--max-transfer-gb`
- `--particle-exclusion-distance-angstrom`
- `--typing-summary`
- `--run-typing`
- `--no-run-typing`
- `--typing-normalization-method`
- `--typing-min-component-area-px`
- `--typing-pixel-size-angstrom`
- `--typing-timeout`
- `--max-overlay-images`
- `--max-bar-items`
- `--`

Arguments after `--` are forwarded to `cryofilter infer`.

## Full-Micrograph Fine-Tuning Script

The app Fine-tune tab wraps:

```bash
python scripts/train_patch_based_fast.py --help
```

This script has many research and ablation flags. The current app defaults are intentionally conservative for common 24 GB GPUs:

```text
--batch_size 16
--full_mic_patches_per_mic 64
--stitched_val_batch_forward_size 16
--num_workers 4
```

On larger GPUs, advanced users can raise these values after confirming that training processes micrographs without CUDA OOM.
