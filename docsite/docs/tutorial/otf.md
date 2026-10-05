---
title: On-the-fly with motion correction
---

# Run OTF with motion correction

## Prerequisites

Use CryoSPARC 5 and a processing node that can directly read the project files.
Connect in the app's **CryoSPARC** tab first. Prepare a small Patch Motion
Correction job that is queued or running, model weights, and GPU capacity
allocated separately from motion correction. A completed particle job is only
needed for the later, optional filtering step.

!!! note "Default: no particle picks"
    Start OTF with motion-corrected micrographs as they arrive. You do not need
    particle picks to generate masks or view live previews. Picking and particle
    filtering can happen later, after masks are available.

## Start the live run

1. Queue Patch Motion Correction in CryoSPARC. Open the app's **OTF** tab.
2. Enter **Project**, **Workspace**, and **Motion-correction job** using the
   actual source job number.
3. Set **GPU devices** to the allocated IDs and **CPUs** to the available budget,
   leaving capacity for motion correction. With `CUDA_VISIBLE_DEVICES` set,
   use IDs from that allocation.
4. With one GPU, use segmentation alone; **Run typing** is disabled. With two
   or more, enable it to divide GPUs between segmentation and typing, or leave
   it off to use all selected GPUs for segmentation.
5. Set **Weights** and a fresh, persistent **Output root**. Start with
   **Threshold** `0.6`, **Profile** `balanced`, and **Live previews** enabled.
   Review **Output budget (GiB)**, whose default is `25`, against available storage.
6. Select **Start OTF**. In **Monitor**, watch the segmented count and previews
   increase. If typing is enabled, its count can lag behind mask generation.
7. Open the new cryoFILTER External Job in CryoSPARC. In **Event Log**, choose
   **Follow latest** to inspect the current live gallery and checkpoint.

Masks become available as micrographs are processed. Keep the output root in
place: the OTF card refers to its state and masks for filtering and recovery.

## Optional: filter from the OTF card

1. Open **Filter Picks**, keeping **Mask source** as **CryoSPARC mask job**.
2. Enter **Project**, **Workspace**, **cryoFILTER mask job** (the new OTF card,
   not the motion-correction card), and a completed **Particle job**.
3. Set **Exclusion (Å)** and select **Filter picks**. No mask path is required.
4. Inspect the new filtering card's `particles_accepted` and `particles_rejected`.

!!! note "Filtering before all masks are ready"
    **Missing masks** defaults to **Stop and report missing masks**. To filter
    the available masks during acquisition, choose **Separate pending particles**.
    The new card also exposes `particles_pending`. Run filtering again when more
    masks are ready; each filtering run represents that point in time.

For a presentation, keep a completed OTF card ready to inspect if the live
motion-correction job is still queued. Identify it as the completed rehearsal.
Cancelling OTF stops its workers; motion correction continues independently.

## Screenshots to add

- `otf-01-live.png` — alt: OTF run with increasing segmented counts while motion correction is running.
- `otf-02-gallery.png` — alt: CryoSPARC Event Log showing the latest checkpoint and live micrograph gallery.
- `otf-03-filter.png` — alt: Filtering from an OTF card with the missing-mask policy visible.

## What success looks like

Masks and preview images arrive before motion correction finishes. The final
run accounts for the source's successful micrographs. Filtering produces
accepted/rejected outputs, with a separate pending output when requested;
those counts account for the source particle input. Next, demonstrate
[annotation](annotation.md).
