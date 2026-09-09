# cryoFILTER CryoSPARC Remote Integration

This integration lets a local cryoFILTER checkout communicate with a CryoSPARC
instance through a small Python bridge installed on a host that can reach the
CryoSPARC API and read CryoSPARC project storage.

## Architecture

- Local orchestrator: `cryofilter cryosparc ...`
- Remote bridge: `cryofilter-bridge ...`
- Transport: OpenSSH for commands, `rsync` for staged file transfer
- CryoSPARC API: configured with a master host/base port or base URL
- CryoSPARC data: resolved from the project directory reported by CryoSPARC Tools

The SSH host does not need to be the CryoSPARC master. It only needs Python,
`cryosparc-tools`, access to the CryoSPARC API, and filesystem visibility of the
project directories referenced by CryoSPARC datasets.

## Environment

Create the local cryoFILTER environment, then align CryoSPARC Tools with your
server version:

```bash
conda env create -f environment.yml
conda activate cryofilter
python -m pip install -e .
cryoFILTER install-cryosparc-tools
```

The helper shows a command-line menu of `cryosparc-tools` minor-version
families available from PyPI. Select the minor version matching your CryoSPARC
server, or pass it directly for scripted installs:

```bash
cryoFILTER install-cryosparc-tools --cryosparc-version 5.0
```

Press Enter in the helper, or pass `--latest`, to install the latest
`cryosparc-tools` release with a compatibility warning. CryoSPARC recommends
matching Tools to the CryoSPARC minor release at your site: for CryoSPARC
`vX.Y.Z`, use the latest `vX.Y` Tools package, and the `Z` component does not
need to match.

The bridge host must run a Python environment with `cryosparc-tools` and
`pydantic` available. `deploy-bridge` copies the cryoFILTER bridge source bundle
to the remote host; it does not create or modify the remote Python environment.
When using the deployed source-bundle launcher, set `[cryosparc.bridge].python`
to the Python executable from that prepared environment and
`[cryosparc.bridge].command` to the deployed `bin/cryofilter-bridge` launcher.

## Configuration

Start from `docs/cryosparc/cryosparc.example.toml` and replace the placeholders:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml doctor
```

Do not store passwords in the config. For CryoSPARC Tools v5, authenticate on the
remote bridge host with the CryoSPARC Tools login command so token state is owned
by the remote user. For older deployments, use a site-owned connection file if
that is your CryoSPARC Tools convention.

## Commands

Check connectivity:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml doctor
```

Deploy the bridge source bundle:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml deploy-bridge
```

Inspect a project, workspace, or job:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml inspect --project P1 --workspace W2 --job J3
```

Validate particle write-back without inference or MRC transfer:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml test-roundtrip --project P1 --workspace W2 --particles J3:particles
```

Stage a small micrograph batch and pull it locally:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml stage-test \
  --project P1 \
  --workspace W2 \
  --micrographs J3:micrographs \
  --particles J3:particles \
  --limit-micrographs 1
```

`stage-test` is manifest-only by default on the CryoSPARC side, so it does not
create a scratch External Job unless `--create-external-job` is passed. It also
checks the staged byte count before pulling and aborts above `--max-transfer-gb`.

Run cryoFILTER against CryoSPARC inputs and finalize the prepared External Job:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml predict \
  --project P1 \
  --workspace W2 \
  --micrographs J3:micrographs \
  --particles J4:particles \
  --checkpoint /path/to/cryoFILTER_FULL.pt \
  --threshold 0.6 \
  -- --device cuda
```

`predict` stages the selected micrographs and particle coordinates through the
bridge, pulls the dereferenced micrograph files locally, writes a RELION-style
STAR coordinate adapter for cryoFILTER overlays, runs `cryofilter infer`, runs
contamination typing by default, builds diagnostic PNGs, pushes UID split
manifests and diagnostics back to the bridge, then saves `particles_accepted`
and `particles_rejected` outputs on the prepared CryoSPARC External Job.

Typing writes a contamination-typing manifest from the staged micrographs and
final binary masks, runs `cryofilter type`, uploads compact typing JSON/CSV
summaries, and adds contamination type pie/bar charts to the job-card
diagnostics. The larger typed-mask arrays remain local by default. Use
`--no-run-typing` to skip automatic typing, or `--typing-summary` to include an
existing typing summary in the diagnostics.

Additional inference arguments can be supplied after `--` and are forwarded to
`cryofilter infer`. The command marks the prepared External Job failed if local
inference errors after staging, so failures are visible in CryoSPARC instead of
leaving an unfinished card.

Finalize an existing local run directory after inference or typing was completed
outside the one-shot `predict` command:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml finalize-run \
  --local-run-dir /path/to/cryofilter_runs/cryosparc/<run_id>
```

`finalize-run` infers `transfer_manifest.json`, `transfer/`, and the local
`inference_summary.json` when they follow the standard run-directory layout. It
runs typing by default unless a local `typing/summary.json` already exists or
`--no-run-typing` is passed, rebuilds the accepted/rejected particle split,
uploads diagnostics, and calls the same CryoSPARC finalizer used by `predict`.
Use `--inference-summary`, `--remote-run-dir`, or `--remote-transfer-manifest`
when resuming a run with a non-standard layout. `resume` and
`finalize-from-run` are aliases for the same command.

Build CryoSPARC-ready diagnostic images after inference:

```bash
cryofilter cryosparc build-diagnostics \
  --inference-summary /path/to/inference_summary.json \
  --typing-summary /path/to/typing/summary.json \
  --output-dir /path/to/cryosparc_diagnostics
```

The diagnostics manifest can include:

- OTF-style particle filtering contact sheet
- Up to 10 individual OTF overlay PNGs for the most contaminated micrographs by default
- Total clean/contaminated area pie chart
- Ranked per-micrograph contamination bar chart with compact rank labels
- Contamination type pie/bar charts when typing was run

The OTF overlay cap is configurable with `--max-overlay-images`. The selected
overlay ranks and contamination percentages are stored in the diagnostics
manifest so users can trace each displayed overlay back to its micrograph.

Attach those diagnostics to a CryoSPARC job card:

```bash
cryofilter cryosparc --config /path/to/cryosparc.toml attach-diagnostics \
  --project P1 \
  --job J99 \
  --diagnostics-manifest /path/to/cryosparc_diagnostics/cryosparc_diagnostics_manifest.json
```

The local command pushes the PNGs and a patched manifest to the bridge host, then
the bridge calls CryoSPARC Tools `log_plot` on the target job. This is the same
attachment path used by the full `predict` finalizer.

## Current Boundary

Implemented and validated:

- Remote bridge deployment
- CryoSPARC authentication and metadata inspection
- Particle External Job round-trip write-back
- Micrograph staging manifest creation
- One-batch rsync pull from a bridge symlink transfer directory
- CryoSPARC-ready diagnostics manifest and PNG chart generation
- Bridge-side diagnostic image attachment through CryoSPARC Tools `log_plot`
- Full `predict` orchestration from staging through local inference, diagnostics
  upload, accepted/rejected particle output save, and diagnostic job-card images
  in the prepared External Job
- Optional automated contamination typing during `predict`, including compact
  typing summary upload and contamination type diagnostics
- Individual OTF overlay attachment is ranked by contaminated area and capped at
  10 images by default, so large CryoSPARC cards remain compact

Recommended release validation:

- Run `doctor` against a CryoSPARC instance that can be reached from the app
  host.
- Run `test-roundtrip` on a small particle output before using a new
  CryoSPARC site.
- Run `stage-test --limit-micrographs 1` to confirm micrograph path resolution
  and transfer limits.
- Run `predict` on a small workspace and confirm that accepted/rejected
  particle outputs and diagnostic images appear on the CryoSPARC External Job.
