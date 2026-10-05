---
title: App behavior
---

# App behavior

!!! note "Page in preparation"
    This guide is being written. Instructions and screenshots will be added here.

Look up app behavior, session persistence, inputs, and outputs.

The **Inference** tab accepts either a local MRC path or a completed CryoSPARC
micrograph output. Connect in the **CryoSPARC** tab first, choose **CryoSPARC
output** as the Inference source, and provide project, workspace, and micrograph
job IDs. Multi-output jobs require an output name, such as `J272:split_0` for an
Exposure Sets Tool split; Curate Exposures accepted outputs are detected
automatically. A particle job is optional. This stages inputs for the normal
inference pipeline and, by default, creates a live CryoSPARC External Job with
progress previews, a connectable `micrographs` output, and a reusable mask
index. **Run typing** is enabled by default for this route. With at least two
requested GPUs it splits them between segmentation and live typing, just like
OTF; one-GPU runs type after segmentation. The app and CryoSPARC publish only
typed five-panel previews. Enter that card in **Filter Picks** with a completed
particle job to publish accepted and rejected particle outputs in a new
External Job.
