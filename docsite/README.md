# cryoFILTER tutorial site

Phases 1–2 provide search, the approved cryoFILTER dark/light theme, local IBM
Plex fonts, the app-derived logo and favicon, Markdown extensions, and screenshot
zoom support. The navigation is approved. The [demo guide](docs/demo.md) now
connects six walkthrough pages covering the five requested presentation segments.
Remaining pages are clearly labeled where they are still in preparation.

The site shows Home, Get started, Tutorial, Reference, and About tabs; on narrow
screens, open the navigation drawer. [INFORMATION_ARCHITECTURE.md](INFORMATION_ARCHITECTURE.md)
records the structure. The maintainer prioritized the live demo on 2026-09-26;
dataset selection and benchmarking are deferred while these pages are rehearsed.

This directory is self-contained. Existing application documentation remains
in its original location.

## Use the tutorial from the app

From the repository root, prepare the tutorial once:

```bash
bash docsite/serve.sh --setup --build
```

This builds `docsite/site/` and exits. Launch `cryoFILTER app` normally and click
**Tutorial** in the sidebar, or open <http://127.0.0.1:8765/tutorial/>. Both use
the app server and its existing SSH tunnel. The app serves the generated files
without importing MkDocs or installing docs dependencies into its own environment.
This integration uses the source checkout; rebuild with
`bash docsite/serve.sh --build` after documentation updates. The
[demo guide](docs/demo.md) gives the single-terminal SSH and launch sequence.

The helper detects Python 3.10–3.14 with `venv` and `ensurepip` support. If it
cannot find one, activate an existing compatible environment and retry, or set
`DOCS_PYTHON=/path/to/python`. Dependencies always go into the separate docs
environment, even when its base interpreter comes from the cryofilter environment.

## Preview while editing docs

Contributors can use `bash docsite/serve.sh` for an automatically reloading docs
preview on localhost:8000. This separate development server is optional.
The helper works from any directory when invoked with its full path.

For manual setup, use a compatible Python and a dedicated virtual environment.
Keep these dependencies out of the scientific cryofilter environment. From the
repository root (with `python3` resolving to Python 3.10–3.14):

```bash
cd docsite
mkdir -p .tmp
export TMPDIR="$PWD/.tmp"
python3 -m venv .venv
source .venv/bin/activate
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -r requirements.txt
mkdocs serve
```

Open <http://127.0.0.1:8000/>. On subsequent visits, activate the environment
and run `mkdocs serve` from this directory. Stop the server with Ctrl-C.

If the checkout is on a remote server, start a tunnel from your laptop:

```bash
ssh -L 8000:127.0.0.1:8000 user@processing-server
```

Run the preview command on that server, then open the same local URL on your
laptop. The documentation preview uses port 8000; the cryoFILTER app has its
own separate server.

Temporary files stay in `.tmp/` inside this directory. The install command uses
PyPI directly so unrelated scientific package indexes do not affect docs setup.

## Validate

From this directory with the virtual environment active:

```bash
python -m pip check
mkdocs build --strict
```

The generated `site/` directory and the virtual environment are ignored by Git.
MkDocs normally regenerates `site/`; keep only generated output there. To retain
an existing build, choose a fresh output path with `--site-dir`.

The revision-date plugin is installed but not enabled yet. It will be configured
when the tutorial pages have committed history. Fonts and branding are bundled
locally; [BRANDING.md](BRANDING.md) records their provenance and licenses.

The first visit uses the dark theme, and Material remembers manual theme changes.
[THEME_PROPOSAL.md](THEME_PROPOSAL.md) records the approved palette and its app
sources. [review/README.md](review/README.md) explains how to preview the component
fixture separately and records Phase 2 validation.

## Development boundary

Submit tutorial changes to the public repository's `main` branch through the
normal review process. Before every push, check the remote URL, branch, and
latest commit. Never run `mkdocs gh-deploy`, push a `gh-pages` branch, or publish
this site without the maintainer's explicit instruction. This scaffold has no
deployment workflow.

Do not commit credentials, `.env` files, SSH configuration, scientific data,
model weights, or unredacted screenshots. Store weights outside this site.

## Recorded maintainer decisions

- Use EMPIAR-10532, downloaded directly by readers from EMPIAR.
- A precisely specified subset is allowed; choose it during the dataset phase.
- Document the particle-picking procedure so readers can reproduce the inputs.
- Measure download size, disk use, runtime, and particle counts after selecting
  the subset. No measurements or expected counts are claimed yet.
- The maintainer confirms that the current OTF workflow works. Earlier source
  documentation describing live validation as pending is out of date.
- The maintainer approved the implemented Phase 2 theme.
- The maintainer approved the navigation and requested a demo-focused batch of
  workflow pages on 2026-09-26, ahead of the full dataset phase and screenshots.
- Use permissive CryoSPARC blob picking by default; crYOLO is an alternative.
- Live OTF starts without picks; subsequent particle filtering is optional.
  Inference previews also work with empty particle fields by default.
- Resume the original review stages after preparing the requested demonstration.
