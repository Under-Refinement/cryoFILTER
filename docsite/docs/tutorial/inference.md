---
title: Your first inference run
---

# Run inference alone

## Prerequisites

Have motion-corrected `.mrc` micrographs with valid pixel-size metadata, the
segmentation weights, a writable output location, and a running app. Place
`classifier.pt` beside custom segmentation weights when using **Run typing**. The
micrographs can be a path readable by the app server or a completed CryoSPARC
output. Particle picks are not required.

## Generate masks

Open **Inference** and fill in the form:

| Field | Setting for the demonstration |
|---|---|
| Source | `Local path` |
| Micrographs | Absolute path to one micrograph or a small representative directory |
| Weights | Absolute path to `cryoFILTER_FULL.pt` |
| Output | A fresh directory, for example `/path/to/project/demo_inference` |
| Particles / Particle group | Leave both empty |
| Device / CPUs / GPUs | Match the resources allocated to your session |
| Threshold / Profile | `0.6` / `balanced` |
| Export masks | Checked; these files are needed for the next workflow |
| Binned masks | Unchecked for this walkthrough |
| OTF images | Leave checked to generate diagnostic previews; particles are optional |
| Extra args | Leave empty for this walkthrough |

Leave **Recursive** checked if the input contains nested directories. Set
**Pixel size** only when the MRC header is missing or incorrect and you know the
correct value in Å/pixel. Select **Launch**, then open **Monitor** and select the run.

!!! note "Previews work without picks"
    With empty particle fields, **OTF images** produces micrograph, mask,
    probability, and retained-region previews. If you supply particles, it
    instead produces overlays that include particle decisions. Keep the particle
    fields empty for this inference-only demonstration.

## Use a completed CryoSPARC job instead

The same **Inference** tab can stage its input directly from CryoSPARC:

1. Open **CryoSPARC**, enter the instance URL and credentials, and select
   **Connect**.
2. Return to **Inference** and change **Source** to **CryoSPARC output**.
3. Enter the **Project**, **Workspace**, and **Micrograph job**. A shorthand
   such as `J54` tries the usual micrograph outputs. For a multi-output job,
   select the result group explicitly: for example, use `J272:split_0` for an
   Exposure Sets Tool split. Curate Exposures normally resolves its accepted
   result automatically; `J273:micrographs_accepted` selects it explicitly.
4. Leave **Particle job** empty for mask-only inference, or enter an optional
   particle output such as `J56` to include pick filtering and particle overlays.
5. Choose the normal weights, output, and compute settings, then select
   **Launch**. Leave **Run typing** checked to classify contamination subtypes.
   Under **CryoSPARC staging**, leave **Publish CryoSPARC job** checked. This
   section also contains typing CPU/accuracy controls, an optional micrograph
   limit, and a 25 GB transfer safety cap.

The app immediately creates a cryoFILTER External Job connected to the selected
micrograph output. With typing selected and at least two GPUs requested, the GPUs
are split between segmentation and typing just like OTF: completed masks are typed
incrementally on dedicated GPU(s). The app monitor changes to subtype charts as
typing summaries arrive, and the app artifacts, CryoSPARC card, event log, and
`micrographs` output progressively gain the same typed five-panel previews. Untyped
previews are suppressed. With one GPU, typing waits for segmentation to finish so
the two models do not contend for memory. When the run finishes, the
card exposes a connectable `micrographs` output and retains the mask index used
by **Filter Picks**. The app monitor log prints the card number, for example
`P280/J301`.

Keep the inference output and its `cryofilter_runs/inference_staging/<run-id>`
directory in place: the card's reusable mask index refers to files there. Clear
**Publish CryoSPARC job** only when you intentionally want staged input without
a CryoSPARC result card.

To filter a later particle-picking result, open **Filter Picks**, enter this
Inference card as **cryoFILTER mask job**, and enter the completed CryoSPARC
particle job. That operation creates a new External Job with connectable
`particles_accepted` and `particles_rejected` outputs; the Inference card can be
reused for additional particle jobs. See [Filter particle picks](filter-picks.md).

## Inspect the output

Find the per-micrograph `<stem>_mask.npy` and `<stem>_prob.npy` files in the output.
With typing enabled, subtype summaries and typed masks are written under `typing/`,
and typed PNGs are written under the run's `card_previews/` directory. The five
CryoSPARC panels show the raw micrograph, contamination mask, typed mask,
probability map, and retained regions. Keep this output and the same input
micrographs for the next step.

??? example "Same thing from the command line"

    Replace the example paths before running:

    ```bash
    cryofilter infer \
      --input /path/to/micrographs \
      --checkpoint /path/to/cryoFILTER_FULL.pt \
      --output-dir /path/to/project/demo_inference \
      --recursive --device cuda \
      --threshold 0.6 --inference-profile balanced \
      --no-render-particle-overlays --render-images
    ```

## Screenshots to add

- `inference-01-inputs.png` — alt: Inference form using a local source with empty particle fields, exported masks enabled, and diagnostic images requested.
- `inference-01b-cryosparc-source.png` — alt: Inference source selector showing CryoSPARC project, workspace, micrograph job, and optional particle job fields.
- `inference-02-results.png` — alt: Diagnostic image showing the raw micrograph, mask, probability map, and retained regions.

## What success looks like

The run finishes with a probability map and binary mask for each processed
micrograph. A CryoSPARC-backed run also finishes with a completed External Job,
a `micrographs` output, a reusable mask index, and—when selected—completed
contamination typing. Inspect representative images, including clean and
contaminated regions, before
[filtering particle picks](filter-picks.md).
