---
title: Demo walkthrough
---

# Demo walkthrough

## Prerequisites

Use an existing cryoFILTER installation on your processing server, with the
segmentation and classifier weights already available. Have completed
motion-correction and particle-picking jobs ready for the first workflow.
Use a permissive CryoSPARC blob-picking result, or a crYOLO result, so you can
show how contamination filtering changes the retained picks.

!!! note "Working tutorial"
    These walkthroughs use the current app controls. The reproducible EMPIAR-10532
    subset, measured runtime/counts, and illustrated screenshots are still being
    prepared. Use your existing data for this demonstration.

## Start the app and this site

Use one terminal and one SSH tunnel for both the app and tutorial. From your
laptop, connect to the processing server with port forwarding enabled:

```bash
ssh -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 user@processing-server
```

In that same terminal, now on the server, prepare your cryoFILTER checkout:

```bash
cd /path/to/cryoFILTER
git pull --ff-only origin main
conda activate cryofilter
python -m pip install -e .
bash docsite/serve.sh --setup --build
screen -S cryofilter-demo
```

The docs setup finds an available Python 3.10–3.14 with virtual-environment
support and installs the pinned docs dependencies into `docsite/.venv`.
It builds the tutorial and returns to the prompt. After later documentation
updates, run `bash docsite/serve.sh --build`; no reinstall is needed.

Inside that screen session, launch the app from your chosen working directory:

```bash
cd /path/to/project
cryoFILTER app --port 8765
```

Open the [app](http://127.0.0.1:8765/) on your laptop and click **Tutorial** in
the sidebar. It opens the tutorial in a new browser tab using the same server.
The demo hub is at <http://127.0.0.1:8765/tutorial/demo/>. Keep this SSH terminal
open; you can detach from screen with ++ctrl+a++ followed by ++d++ while the
SSH tunnel remains connected. Paths entered into the app refer to server files.
If you present directly from the server, use the same URLs without an SSH tunnel.

!!! warning "Keep the app session alive"
    Closing the browser is fine. Stopping the app server can prevent it from
    reconnecting to running jobs. Reattach with `screen -r cryofilter-demo`.
    Match the tunnel port to the port actually used by the app.

## Prepare once before presenting

- Record the project/workspace and completed micrograph/particle output refs.
- Identify a small, representative micrograph directory and matching particle
  export for the two file-based workflows. Keep the original micrograph names.
- Identify the absolute path to `cryoFILTER_FULL.pt` and keep `classifier.pt`
  alongside it for typing.
- Reserve CPU/GPU resources for the app; OTF needs GPU capacity separate from
  motion correction. Run on the allocated processing node that can read the data.
- Prepare a small Patch Motion Correction job to queue for the live OTF segment.
- Keep an annotation session with several saved images and a completed fine-tune
  result available. Training does not need to finish during the presentation.
- Rehearse once and keep completed run cards and outputs for each workflow
  available to inspect while a new run is processing.

Use fresh output directories for the rehearsal and presentation. Confirm actual
paths and job IDs privately; the examples below use placeholders.

## Present in this order

| Segment | Open | Demonstrate | Show the result |
|---|---|---|---|
| 1. Existing CryoSPARC inputs | [CryoSPARC walkthrough](tutorial/cryosparc.md) | Connect, select completed inputs, launch | External Job, accepted/rejected particles, diagnostics |
| 2. Inference alone | [Inference walkthrough](tutorial/inference.md) | Leave particle fields empty and generate masks | Mask/probability files and diagnostic images |
| 3. Filter existing masks | [Filter Picks walkthrough](tutorial/filter-picks.md) | Reuse segment 2's masks with an exported pick set | Filtered export and kept/removed counts |
| 4. Live OTF | [OTF walkthrough](tutorial/otf.md) | Follow a queued/running motion-correction job | Increasing processed counts and live gallery |
| 5. Annotation and adaptation | [Annotation](tutorial/annotation.md), then [Fine-tune](tutorial/fine-tuning.md) | Draw, save, resume, and launch training | Edited manifest, training progress, prepared validation outputs |

## Capture for the tutorial

Capture at 1440×900. Each walkthrough lists the screenshots it needs and their
intended alt text. Redact credentials, instance URLs, internal hostnames and
paths before sharing or committing captures. Contributors can use the five
walkthroughs as the starting point for adding images and refining explanations.

## What success looks like

You can move through all five workflows with the app and tutorial open side by
side. Each segment ends at a real run, saved artifact, or prepared completed
example. Report the observed counts and resource use from your rehearsal;
the tutorial does not prescribe unmeasured expected values.
