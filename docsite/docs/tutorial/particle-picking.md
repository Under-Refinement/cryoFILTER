---
title: Prepare particle picks
---

# Prepare particle picks

## Prerequisites

Use particle picks from the same motion-corrected micrographs you will process.
For the live demo, motion correction and particle picking are already complete.

## Picking choice for the demo

Use CryoSPARC blob picking as the default; crYOLO is also suitable. Keep the
picking reasonably permissive: retain a fair number of questionable picks so
the demo can show how contamination filtering changes the result. Inspect the
picks to confirm that useful particles are still included.

Record the software version, picker settings, micrograph source, and output job
or export used in the rehearsal. A universal numeric threshold is not prescribed
here; the exact reproducible recipe will accompany the selected EMPIAR subset.

For [the CryoSPARC workflow](cryosparc.md), provide the completed particle job
output. For [file-based filtering](filter-picks.md), export matching `.cs`/`.csg`
or `.star` inputs and keep their micrograph identifiers consistent. If using
crYOLO, confirm that the export has been converted or imported into a supported
input with correct coordinate geometry before presenting.

[Back to the tutorial overview](index.md).

## What success looks like

You can identify the micrograph source for every particle input, inspect picks
over the images, and reproduce the chosen picking settings. Keep the original
input so you can compare it with the filtered result.
