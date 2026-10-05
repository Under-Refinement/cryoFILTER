# Phase 3: proposed tutorial structure

Status: navigation and theme approved by the maintainer.
The outline is also available as working site navigation with clearly labeled
placeholder pages, following the maintainer's request to see the tabs in the
browser. Full tutorial pages will be drafted individually in Phase 5, after
the dataset work in Phase 4, except for the demo priority below.

On 2026-09-26 the maintainer prioritized a presentation of the five existing
workflows using already processed inputs. `docs/demo.md` and six workflow pages
now support that presentation. Detailed dataset selection, benchmarking, and the
remaining page drafts follow after the demo; no new scientific measurements are
claimed by these guides.

## Proposed navigation

Five top-level entries: **Home**, **Get started**, **Tutorial**, **Reference**,
and **About**. Paths below are planned paths relative to `docsite/docs/`.

```text
Home                                      index.md

Get started
  Install                                 get-started/install.md
  Download weights                        get-started/weights.md
  Launch the app                          get-started/launch.md
  Remote servers & SSH tunnel             get-started/remote.md
  Troubleshooting                         get-started/troubleshooting.md

Tutorial
  Demo walkthrough                        demo.md
  Overview & prerequisites                tutorial/index.md
  Core walkthrough
    1. Get the tutorial data               tutorial/data.md
    2. Your first inference run            tutorial/inference.md
    3. Read the results                    tutorial/results.md
    4. Prepare particle picks              tutorial/particle-picking.md
    5. Filter particle picks               tutorial/filter-picks.md
  CryoSPARC workflows
    Connect to CryoSPARC                   tutorial/cryosparc.md
    On-the-fly with motion correction      tutorial/otf.md
  Adapt the model
    Annotate micrographs                   tutorial/annotation.md
    Fine-tune and compare results          tutorial/fine-tuning.md

Reference
  CLI reference                           reference/cli.md
  Expert notes                            reference/expert-notes.md
  App behavior                            reference/app.md
  Classifier & weight lookup              reference/classifier.md
  CryoSPARC remote integration             reference/cryosparc-remote.md
  OTF technical guide                     reference/otf.md
  Fine-tuning protocol                     reference/fine-tuning.md

About                                     about.md
```

## Reading order and page boundaries

Home introduces the problem, intended reader, existing overview animation, and
what a completed run provides. It points new readers to Get started and returning
readers to the tutorial overview. About contains citation, related work, license,
and instructions for reporting a problem.

Get started covers installation through a running app. Launch the app links to
the remote-server instructions before its launch step when needed; both cover
the persistent server session. Troubleshooting has one main home, with relevant
sections linked from each workflow.

The core walkthrough uses the browser app with files accessible to the app
server. It starts with mask generation in **Inference**, then uses **Monitor** to
inspect outputs. It introduces particle inputs only after readers have seen
what a contamination mask means. **Filter Picks** then reuses those masks.

| Tutorial page | Prerequisites to state | Intended self-check |
|---|---|---|
| Overview & prerequisites | Installed app and weights; selected route | Reader can identify the inputs, resources, and pages required for that route |
| Get the tutorial data | Access to EMPIAR-10532; storage estimate once measured | The exact selected files are present, with motion-corrected micrographs and usable pixel-size metadata ready for inference |
| Your first inference run | Prepared micrographs, segmentation weights, writable output location | The run completes and produces the expected masks and previews |
| Read the results | Completed inference outputs | Reader can relate a micrograph, probability map, and binary mask and recognize contamination versus usable regions |
| Prepare particle picks | The same micrographs; access to the selected picking software | A reproducible pick set/export matches those micrographs; expected counts are supplied after measurement |
| Filter particle picks | Matching micrographs, existing masks, particle inputs | Filtering summaries and diagnostic images agree with the measured example; output files are identifiable |
| Connect to CryoSPARC | Access to an instance, compatible Tools, readable project data and suitable micrograph/particle outputs | The browser workflow produces a CryoSPARC External Job with accepted/rejected particle outputs and diagnostics |
| On-the-fly with motion correction | Working CryoSPARC connection; CryoSPARC 5 Patch Motion Correction input; readable project files and allocated resources | Live masks/previews appear; filtering from the OTF card produces identifiable outputs |
| Annotate micrographs | Selected motion-corrected micrographs; writable session location | Saved masks and the edited manifest can be inspected and the session can be resumed |
| Fine-tune and compare results | Saved annotations, held-out validation images, model weights, measured resource requirements | Training produces a usable checkpoint; comparable before/after results are inspected on held-out images |

These are acceptance criteria for future pages, not measured dataset results.
The data page will explain any required preparation from downloaded files to
motion-corrected micrographs once Phase 4 establishes exactly what is available.
Particle picking takes place in the selected external processing software; the
cryoFILTER tutorial will document its reproducible inputs and handoff.

CryoSPARC workflows and Adapt the model are optional extensions. The overview
provides direct entry points and explicit prerequisites, so readers with existing
CryoSPARC jobs or annotations can start at the appropriate page. The CryoSPARC
page covers connection and an integrated run; OTF adds the live motion-correction
workflow and links back to filtering. The adaptation path ends with comparison
against the original model, without promising an improvement.

**Monitor** is explained where readers need it: inference progress, results,
filtering artifacts, and subsequent workflows. It does not need a separate
mandatory chapter. Subtype interpretation is introduced when those outputs are
available; classifier lookup and technical details live in Reference.

## Consistent tutorial page format

Each tutorial page opens with **Prerequisites**, including relevant previous
outputs and any local/remote differences. The overview and data page carry the
resource summary once measured. Steps lead with the browser UI; genuine
alternatives use tabs and verified CLI equivalents use a collapsed example.

Each page ends with **What success looks like**: an annotated image or expected
output, measured counts where appropriate, and the next step. Until screenshots
are supplied, the draft states the exact shot needed and its intended alt text.
Do not substitute fabricated counts or screenshots for these checks.

## Existing source material to reuse

Source paths below refer to the repository at app commit `081007c`.

| Existing source | Proposed use |
|---|---|
| `README.md` | Installation, weights, launch, workflow names, Home, and About |
| `docs/app/LOCAL_WRAPPER.md` | App behavior reference; browser and remote workflow details |
| `docs/cryosparc/REMOTE_INTEGRATION.md` | CryoSPARC remote integration reference; advanced bridge details |
| `docs/cryosparc/OTF.md` | OTF technical reference and the guided OTF page |
| `docs/fine_tuning.md` | Fine-tuning protocol reference and the adaptation walkthrough |
| `docs/publication_classifier.md` | Classifier and weight lookup reference |
| `docs_for_expert_users/README.md` | Expert notes reference; interpretation and troubleshooting details |
| `docs_for_expert_users/cli_reference.md` | CLI reference and verified collapsed examples |
| `cryofilter/app/static/index.html` and `app.js` | Current visible labels and browser interactions |
| `docs/screenshots/cryoFILTER_app_wrapper.png` and the README animation | Candidate imagery to inspect and redact before copying |

During page drafting, copy and adapt reference material into `docsite/`, preserving
the originals. Resolve links to local site pages/assets, and record each copy's
source revision. Review internal paths, hostnames, credentials, stale labels,
and validation statements before reuse. Do not include source files from outside
`docsite/` at build time; the site must remain transferable as one directory.

Two concrete source checks for the later drafts: the wrapper's tab list omits
OTF, while the current app includes it; the OTF guide's pending-particle option
wording differs from the current form. Verify the current UI wording when those
pages are written. The maintainer has confirmed that the current OTF workflow
works; older pending-validation language must not be presented as current status.

## Next work after the demo

The approved structure contains the core walkthrough, the two optional workflow
groups, and the particle-picking page. The demo guide links the presentation's
five segments in the requested order. Complete the remaining placeholders as
their inputs and reference material are reviewed.

Phase 4 will use the already approved **EMPIAR-10532**, downloaded by readers
directly from EMPIAR. Specify a reproducible subset and the picking procedure
with the maintainer, then measure storage, runtime, and counts. Those decisions
are intentionally still open; this outline selects no files, software settings,
or expected measurements.
