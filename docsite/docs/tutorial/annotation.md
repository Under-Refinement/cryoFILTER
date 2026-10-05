---
title: Annotate micrographs
---

# Annotate and resume in the browser

## Prerequisites

Have readable motion-corrected micrographs and a writable annotation output
location. A CryoSPARC source requires an existing connection in the app's
**CryoSPARC** tab. The browser editor does not require Qt or X forwarding.

## Start a session

1. Open **Annotation**. Set **Source** to **Micrographs** and enter a file or
   directory in **Micrographs**. Alternatively, choose **CryoSPARC output** and
   provide its **Project**, **Workspace**, and **Micrographs** reference.
2. Set **Output root** and a new **Output name**. In **Advanced**, keep **Editor**
   at **Browser**. For shared CryoSPARC storage, leave **Copy micrographs** off.
3. Select **Start**. Adjust **Contrast**, **Brightness**, or **Gamma** as needed;
   these are display controls, not edits to the underlying micrograph.

## Draw, save, and resume

1. Select **Action → Add mask** and **Mode → Polygon**. Click vertices around
   contamination, then use **Close** to finish the polygon. Use **Circle / ellipse**
   to draw an oval by dragging instead.
2. Choose a **Region type** if known. **Unlabeled** regions still mark contamination
   for binary fine-tuning.
3. Select **Save** or **Save + Next**. Use **Action → Erase mask** to subtract a
   drawn region; save again after editing. Inspect each saved mask.
4. Save several images, including useful clean regions as well as contamination.
   Use **Done** to finalize the session and populate the Fine-tune form.
5. Demonstrate **Recent session → Resume**. For a moved session, choose
   **Source → Resume manifest** and supply its `manifest_edited.csv`.

!!! tip "Prepare enough saved annotations"
    Automatic training/validation splitting needs at least two saved micrographs.
    That is a software minimum, not an adequate validation study. Use a larger,
    representative saved set for a meaningful adaptation demonstration.

## Screenshots to add

- `annotation-01-draw.png` — alt: Browser editor with a polygon around contamination and the drawing controls visible.
- `annotation-02-resume.png` — alt: Saved annotation session available through Recent session and Resume; redact paths.

## What success looks like

Saved masks and `manifest_edited.csv` are present. Resuming recovers the saved
session, and **Done** fills the edited-manifest field for
[fine-tuning](fine-tuning.md). Confirm the saved mask outlines before launching training.
