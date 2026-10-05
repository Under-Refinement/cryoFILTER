# Demo guide handoff

Scope requested on 2026-09-26: present the five existing app workflows using
prepared inputs, with the tutorial visible alongside the app. This priority
supersedes the original dataset-first and one-page-per-review sequence for the
demo pages. The maintainer also requested that previews work without particle
picks by default; a small app command-builder fix supports that behavior.

Start at [docs/demo.md](docs/demo.md), also linked from Home and Tutorial. Use
`bash docsite/serve.sh --setup --build` once, then launch the app normally and
click **Tutorial**. One SSH session forwards the app port for both. After docs
updates, `bash docsite/serve.sh --build` rebuilds and exits. The separate docs
development server remains available to contributors who want live reload.

## Source verification

App baseline: the integrated OTF application with mask-only preview support.
UI labels/defaults were checked against
`cryofilter/app/static/index.html`; connection and annotation behavior against
`connect.js`, `app.js`, and `cryofilter/app/training_manifest.py`. Inference and
filtering commands were checked against `cryofilter/app/server.py` and
`cryofilter/cli.py`. OTF and training outputs also follow the existing guides in
`docs/cryosparc/OTF.md` and `docs/fine_tuning.md`.

Important details retained in the walkthroughs:

- Inference with **OTF images** enabled now selects `--render-images` when
  particle fields are empty, and particle overlays when picks are supplied.
  The previous command builder requested particle overlays unconditionally,
  which failed without a particle file. The CLI's particle-overlay validation
  remains in place. Turning previews off still disables automatic rendering.
- Live OTF requires no particle input. Its default demo follows motion correction
  and produces masks/previews; filtering from the OTF card is optional and later.
- Existing-mask filtering uses **Mask source → Local files** and the original
  micrograph geometry. Its local export and summary differ from CryoSPARC's
  accepted/rejected output slots.
- The visible resource label is **Typing CPUs**. The OTF missing-mask choice is
  **Separate pending particles**; older prose uses different wording.
- Annotation **Done** populates Fine-tune. Automatic splitting requires at least
  two saved rows; that software minimum is not a validation recommendation.
- The maintainer confirms live OTF already works. The older guide's unvalidated
  live-test statement is not repeated as current status.

## Rehearsal and contributor work

The presenter supplies real paths, job IDs, allocated resources, and saved
results privately. Each walkthrough names the screenshots needed and their alt
text. Capture at 1440×900, redact, and keep scientific overlay colors unchanged.

The guides have not been exercised against the presenter's CryoSPARC instance,
GPU allocation, or actual particle data in this documentation task. Rehearse
those integrations using the checklist in the demo guide. Dataset-specific
sizes, runtime, counts, exact picker settings, and validation images remain open.
No new training, inference, or large data downloads were run to prepare the site.

Validation: eight targeted app/OTF regression cases pass, covering inference
with optional particle inputs and previews, and OTF launch without picks.
CLI examples pass shell syntax checks and use source-defined flags. The site
passes the strict MkDocs build; the docs launcher passes Bash syntax checking.
Browser checks pass for the demo hub and six workflow pages in both themes at
1440px and 320px, including the Home link, no horizontal overflow, and no script
errors. All 1,065 internal links and fragments resolve across the 25-page site.

The same-server tutorial integration passes all 45 app-server tests, including
page/asset serving, redirects, a missing-build message, and traversal/symlink
rejection. The build-only launcher passes a strict build and exits normally.
Browser checks confirm the app's Tutorial link, navigation across all seven demo
pages, search under `/tutorial/`, theme persistence, and the 320px layout.
