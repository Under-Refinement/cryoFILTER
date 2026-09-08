# Fine-Tuning Protocol

This page captures the recommended cryoFILTER fine-tuning recipe used for the membrane adaptation tests. It is meant for small user-provided annotation sets where the public FULL model is already close, but misses dataset-specific contamination appearance.

## Data

Prepare a train/validation manifest with one row per micrograph:

```csv
dataset_id,stem,micrograph_path,gt_mask_path,pixel_size_angstrom
1,FoilHole_0001,/path/to/micrographs/FoilHole_0001.mrc,/path/to/FoilHole_0001_gt_mask.npy,1.06
1,FoilHole_0002,/path/to/micrographs/FoilHole_0002.mrc,/path/to/FoilHole_0002_gt_mask.npy,1.06
101,FoilHole_val_0001,/path/to/micrographs/FoilHole_val_0001.mrc,/path/to/FoilHole_val_0001_gt_mask.npy,1.06
```

Fine-tuning currently expects numeric `dataset_id` values. Keep validation images held out from training. If several micrographs come from the same acquisition condition, keep related images in the same split when possible. For very small adaptation sets, explicit `--train_dataset_ids` and `--val_dataset_ids` are safer than a random split.

Review the exact masks that training will consume before launching a long job. This is especially important for membrane or mesh-style annotations, where holes inside a broad region may be real background.

If you are drawing masks from scratch in the app, you can point the Annotation
tab directly at motion-corrected micrographs or a CryoSPARC micrograph output;
the wrapper creates the starting manifest for you, including a deterministic
row-level `split` column. The Fine-tune tab then prepares a training-ready
manifest from saved annotation rows only and, when Train IDs and Val IDs are
blank, assigns those saved rows to a deterministic 70/30 train/validation
split. If you launch the lower-level GUI command directly, start with the same
manifest but omit `gt_mask_path`. The annotation GUI writes an edited manifest
with `gt_mask_path` filled in.

## Annotation

The recommended annotation path is the browser-native Annotation tab in the
local app:

```bash
cryoFILTER app
```

Use it with a local micrograph directory, a CryoSPARC micrograph output, or a
previous `manifest_edited.csv` to resume a session. The desktop matplotlib GUI
is kept as an expert fallback. Launch it from the repository root with an
interactive display, such as VNC or `ssh -Y`:

```bash
conda activate cryofilter

# Optional cluster fallback if Qt/xcb fails:
pip install -e ".[gui]"
export CRYOFILTER_ANNOTATION_BACKEND=TkAgg

cryofilter gui \
  --manifest /path/to/my_images.csv \
  --output-dir ./annotation_exports/my_data_v1
```

Basic controls:

- Left click adds polygon points.
- Right click or Enter closes a polygon.
- `x` erases a polygon area from an existing mask.
- `s` saves the current image.
- `e` finalizes the export.
- `q` saves a checkpoint and quits.

Use the finalized manifest for training:

```text
annotation_exports/my_data_v1/manifest_edited.csv
```

## Recommended Recipe

The current best default is full-micrograph training from the released FULL checkpoint:

In the wrapper app, the Fine-tune tab can infer micrographs from manifests
produced by the Annotation tab, accept a local micrograph directory, or reuse a
CryoSPARC micrograph output. The lower-level CLI command still expects
`MIC_DIR`, especially for older or custom manifests.

```bash
EDITED_MANIFEST=annotation_exports/my_data_v1/manifest_edited.csv
MIC_DIR=/path/to/micrographs
TRAIN_IDS=1
VAL_IDS=101

python scripts/train_patch_based_fast.py \
  --manifest "$EDITED_MANIFEST" \
  --mic_dir "$MIC_DIR" \
  --pretrained_model pretrained_models/cryoFILTER_FULL.pt \
  --output_dir ./finetune_my_data_v1 \
  --model_type unet_attention \
  --attention_type global \
  --use_psd true \
  --adaptive_downsample \
  --adaptive_downsample_method fourier \
  --target_pixel_size 2.0 \
  --working_patch_size 256 \
  --working_stride 96 \
  --normalization_method percentile_extra_wide \
  --norm_type group \
  --batch_size 16 \
  --num_gpus 1 \
  --num_epochs 120 \
  --lr 2e-5 \
  --warmup_steps 80 \
  --weight_decay 0.01 \
  --decoder_dropout 0.1 \
  --label_smoothing 0.0 \
  --train_dataset_ids "$TRAIN_IDS" \
  --val_dataset_ids "$VAL_IDS" \
  --no_region_gt_fill_holes \
  --full_micrograph_training \
  --full_mic_patches_per_mic 64 \
  --full_mic_overlap 0.625 \
  --full_mic_prefetch_workers 1 \
  --skip_patch_cache_in_full_mic \
  --psd_frequency_band_channels \
  --psd_frequency_bands 0.03:0.05,0.09:0.11,0.14:0.16,0.22:0.24 \
  --use_dice_loss \
  --lambda_dice 2.0 \
  --lambda_focal 1.0 \
  --lambda_edge 0.3 \
  --stitched_val_every_n_epochs 3 \
  --stitched_val_mics_per_dataset 1 \
  --stitched_val_overlap 160 \
  --stitched_val_batch_forward_size 16 \
  --stitched_val_blending_window tukey \
  --stitched_val_blending_edge_px 16 \
  --stitched_val_thresholds 0.05,0.15,0.25,0.35,0.45,0.55,0.70 \
  --stitched_val_metrics_max_samples 3000000 \
  --early_stop_patience 14 \
  --early_stop_min_epochs 30 \
  --early_stop_min_stitched_cycles 10 \
  --num_workers 4
```

Every `dataset_id` in the edited manifest must appear in either `TRAIN_IDS` or `VAL_IDS`. Run the command from a source checkout so `scripts/train_patch_based_fast.py` is available. Use `python scripts/train_patch_based_fast.py --help` to see the complete full-micrograph training options; `cryofilter train --help` documents the lightweight patch-based utility.

The shown defaults are sized for common 24 GB GPUs. If the log shows CUDA out-of-memory skips or zero processed micrographs, lower `--batch_size` and `--stitched_val_batch_forward_size`. On larger-memory GPUs, raise `--batch_size`, `--full_mic_patches_per_mic`, and `--stitched_val_batch_forward_size` only after a short smoke test.

## Why These Settings

`--full_micrograph_training` makes the training distribution match inference and validation. This is preferred over GT-biased crop training for fine-tuning, because small annotation sets can otherwise teach an inflated contamination prior.

`--adaptive_downsample --adaptive_downsample_method fourier --target_pixel_size 2.0` keeps micrographs at a consistent physical sampling while avoiding stride-aliasing artifacts.

`--normalization_method percentile_extra_wide` is the deployment-matched normalization that gave the best overall validation behavior in the membrane tests. Robust and SimpliPyTEM-style normalization variants can be useful ablations, but they should be treated as separate model variants because changing normalization shifts the learned probability calibration.

`--no_region_gt_fill_holes` preserves the raw annotation topology. Use it when holes inside annotated membrane, mesh, or carbon regions should stay background. Omit this flag only when the intended GT target is a solid filled region.

`--psd_frequency_band_channels` with the four listed bands adds physically defined Fourier texture channels while preserving the real-space micrograph channel.

The Dice/Focal/edge loss settings favor shape agreement and reduce patch-edge artifacts without forcing excessive positive area.

## Validation

The training script writes stitched validation metrics and probability maps. Promote checkpoints only after checking both metrics and validation overlays:

- `best_model.pt`: best validation PR-AUC, often best for broad recall.
- `best_score_model.pt`: best score balancing PR-AUC against false-positive mass; usually the safest default.
- `best_composite_model.pt`: best low-FP-cap recall composite.

Stitched validation maps and overlays are written under `finetune_my_data_v1/stitched_prob_maps/epoch_###/`.

For datasets with broad carbon or membrane contamination, inspect a threshold sweep instead of assuming `0.50` is optimal. A lower threshold such as `0.35` can recover broad low-confidence regions when the PR curve supports it. Keep the final threshold fixed before evaluating held-out results.

## Small Ablations

If the recommended recipe still misses obvious broad regions, try short ablations before launching another long run:

- normalization: `percentile_extra_wide`, `robust`, `simplipytem`, `simplipytem_local8bit`
- checkpoint selection: PR-AUC versus score versus composite
- threshold: coarse sweep around `0.25` to `0.55`
- raw GT handling: verify whether hole filling is intended before training

Normalization ablations should be compared using the same validation split and the same rendering method used for final deployment.
