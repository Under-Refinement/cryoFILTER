---
title: Connect to CryoSPARC
---

# Run from completed CryoSPARC inputs

## Prerequisites

Start the app on the processing server, with access to the CryoSPARC API and
project files. Have a completed motion-correction output and matching particle
output ready, compatible CryoSPARC Tools installed, and both model weights
available. See the [demo setup](../demo.md#start-the-app-and-this-site).

This walkthrough uses the dedicated **CryoSPARC** tab to run inference and
particle filtering together and write accepted/rejected particles back in one
External Job. For reusable mask generation without requiring a particle job,
the [Inference walkthrough](inference.md#use-a-completed-cryosparc-job-instead)
can publish a mask card first; **Filter Picks** can then apply it to one or more
later particle jobs.

## Connect and select inputs

1. Open **CryoSPARC**. Enter **Instance URL**, **Username/email**, and **Password**,
   then select **Connect**. The run form unlocks after a successful connection.
2. Enter **Project** and **Workspace**, such as `P1` and `W1`, replacing these
   examples with your IDs.
3. Set **Micrographs** to the completed source job/output, for example
   `J1:micrographs`, and **Particles** to the matching completed output, such as
   `J2:particles`. Use the actual output name if it differs.
4. Set **Weights** to the server path of `cryoFILTER_FULL.pt`. Keep `classifier.pt`
   alongside it when **Run typing** is enabled.
5. Set **Device** to `cuda` and **GPUs**/**CPUs** to the resources allocated to this
   session. **Typing CPUs** controls the typing thread budget. **Typing updates**
   offers **Live background** and **Final only**; a single GPU uses final typing.
6. In **Advanced**, use a fresh **Output root**. Keep **Threshold** at `0.6` and
   **Profile** at `balanced` for the initial demonstration. **Limit** can restrict
   the micrographs for a short rehearsal; interpret counts against that run's
   processed input, rather than the entire original particle job.
7. In **Bridge**, leave **Bridge mode** at `local` when the app runs on the
   processing server. Select **Launch**.

## Inspect the run

Open **Monitor** and select the run to follow its log, artifacts, and progress.
Inference, typing, finalization, and upload are separate stages; wait for the
run to finish before treating the CryoSPARC outputs as final.

In CryoSPARC, open the new External Job. Inspect the diagnostic images and the
`particles_accepted` and `particles_rejected` outputs. Use the accepted output
for downstream processing. Point out examples near a contamination boundary.

!!! tip "Start with generous picks"
    For this demonstration, use a permissive blob-picker result or crYOLO export.
    Keep enough questionable picks to demonstrate what the contamination mask
    excludes. The [picking notes](particle-picking.md) describe what to record.

## Screenshots to add

- `cryosparc-01-inputs.png` — alt: Connected CryoSPARC form with example input jobs and resource settings; redact the connection summary.
- `cryosparc-02-results.png` — alt: External Job with accepted and rejected particle outputs and contamination diagnostics.

## What success looks like

The app run finishes and its CryoSPARC External Job contains both particle
splits and diagnostic images. The images show which regions caused rejection.
Continue with [inference alone](inference.md) to demonstrate the file-based path.
