---
title: Fine-tune and compare results
---

# Fine-tune from saved annotations

## Prerequisites

Complete [annotation](annotation.md), save several representative micrographs,
and inspect their masks. Have the base weights, allocated GPU resources, and
held-out images for validation. Keep a completed training run available for the
presentation so you can show its results without waiting for training to finish.

## Launch training

1. Open **Fine-tune**. Check **Edited manifest**, which **Done** in Annotation
   can populate automatically, or enter the saved `manifest_edited.csv` yourself.
2. For a local annotation session, use **Micrograph source → Auto from manifest**.
   If the annotations came from CryoSPARC, check the populated CryoSPARC source
   fields. **Local directory** is available when you need to supply the image path.
3. Set **Base weights** to `cryoFILTER_FULL.pt` and **Output** to a fresh directory.
4. Leave both **Train IDs** and **Val IDs** blank for the automatic split of saved
   rows. The defaults are **Train split** `0.7` and **Split seed** `42`. Inspect the
   split and use explicit dataset IDs when related acquisitions must stay together.
5. Start with the app's recipe: **Epochs** `120`, **Batch size** `16`, **GPUs** `1`,
   **Learning rate** `2e-5`, **Workers** `4`, and **Patches per mic** `64`. Match
   resources to your allocation; reduce batch size if memory is insufficient.
   Keep **Fill region holes** off when holes should remain background.
6. Select **Launch**. Inspect the training log in **Monitor** and confirm that
   saved rows enter both training and validation and that batches are processed.

Only saved annotation rows with usable masks enter the prepared training
manifest. For the live presentation, show initialization and progress, then open
an explicitly identified completed rehearsal run to explain the outputs.

## Inspect and reuse a checkpoint

Inspect validation metrics and the stitched overlays under
`stitched_prob_maps/epoch_###/`. The training recipe writes checkpoint candidates
including `best_model.pt`, `best_score_model.pt`, and `best_composite_model.pt`;
choose using held-out metrics and visual inspection, not the training loss alone.

In **Inference**, enter the chosen checkpoint under **Weights** and use a new
output directory. Compare with the original model on the same held-out images
and the same threshold/settings. Check both missed contamination and removal of
usable regions. Fine-tuning does not guarantee an improvement.

## Screenshots to add

- `fine-tuning-01-inputs.png` — alt: Fine-tune form with a saved annotation manifest, base weights, and split settings; redact paths.
- `fine-tuning-02-comparison.png` — alt: Original-model and fine-tuned results on the same held-out micrograph at matching settings.

## What success looks like

Training processes saved annotations and produces a loadable checkpoint with
validation outputs. You can run that checkpoint in Inference and compare its
held-out behavior with the original model. Return to the [demo guide](../demo.md).
