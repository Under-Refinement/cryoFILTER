---
title: Filter particle picks
---

# Filter picks using existing masks

## Choose a mask source

### CryoSPARC Inference or OTF card

For the integrated CryoSPARC route, first complete a CryoSPARC-backed
[Inference](inference.md) run, or start an [OTF](otf.md) run. Note its
cryoFILTER External Job number. The batch Inference card must be complete; an
OTF card may still be running if you choose the pending-mask policy. Also have a
completed CryoSPARC particle-picking job whose micrograph UIDs match the mask
card.

1. Open **Filter Picks** and choose **CryoSPARC mask job**.
2. Enter **Project**, **Workspace**, the Inference or OTF card number under
   **cryoFILTER mask job**, and the completed particle job.
3. Start with **Exclusion** at `100` Å. Leave **Missing masks** as **Error** for
   completed batch inference. For a still-running OTF card, **Pending output**
   publishes currently unprocessed picks separately.
4. Select **Filter picks** and follow the run in **Monitor**.

The app creates a new CryoSPARC External Job connected to the particle input.
It publishes connectable `particles_accepted` and `particles_rejected` outputs,
plus `particles_pending` when requested. Reuse the same mask card with later
particle-picking jobs without rerunning segmentation.

### Local mask files

Complete [inference alone](inference.md) with **Export masks** enabled and
**Binned masks** disabled. Keep the original micrographs and obtain a matching
CryoSPARC `.cs` export or RELION `.star` particle file. Include the `.csg` and
referenced group files when the CryoSPARC export is split across files.

## Run the local-file filter

1. Open **Filter Picks** and set **Mask source** to **Local files**.
2. Set **Micrographs** to the same original micrographs used for inference.
3. Set **Masks** to the inference output containing `<micrograph-stem>_mask.npy`.
4. Set **Particles** to the exported pick file. Set **Particle group** to its
   `.csg` when applicable; keep referenced files available.
5. Set **Output** to a fresh location, such as `/path/to/project/demo_filtered`.
   Leave **Filtered file** blank for the generated output name.
6. Start with **Exclusion** at `100` Å, the app default. Particle centers at or
   within that distance of a contaminated pixel are removed. Explain that this
   exclusion margin and the inference threshold are different controls.
7. Leave **Allow unmatched** and **Overwrite** unchecked. Keep **Recursive**
   enabled for nested inputs. Select **Launch** and inspect the run in **Monitor**.

!!! warning "Match the inputs"
    Micrograph names and pixel sizes must agree with the particle export and
    masks. Resolve a missing-mask or unmatched-particle error before continuing;
    do not enable Allow unmatched merely to suppress a mismatch.

## Inspect the output

Open `particle_filter_summary.json` and the filtered particle export. By default,
its name ends in `_cryofiltered.cs` or `_cryofiltered.star`; grouped CryoSPARC
exports also produce the corresponding group output. The source particle file
is preserved. This workflow reuses masks without rerunning segmentation.

The summary records `particles_input`, `particles_kept`, and `particles_removed`.
For a matched run, verify that kept plus removed equals input. Use the actual
counts from your run when presenting.

??? example "Same thing from the command line"

    Replace the example paths before running:

    ```bash
    cryofilter filter-particles \
      --input /path/to/micrographs \
      --mask-dir /path/to/project/demo_inference \
      --particle-file /path/to/particles.cs \
      --output-dir /path/to/project/demo_filtered \
      --recursive --particle-exclusion-distance-angstrom 100
    ```

## Screenshots to add

- `filter-picks-01-inputs.png` — alt: Local files filtering form using the preceding inference masks and a matching particle export.
- `filter-picks-02-results.png` — alt: Completed filtering run showing the filtered export and input, kept, and removed counts.

## What success looks like

For a CryoSPARC route, the new output card contains accepted and rejected
particle outputs and the counts reconcile. For a local route, the filtered
export and summary are present and the original pick file remains available.
Continue with [OTF](otf.md) to show how masks can instead be generated while
motion correction runs.
