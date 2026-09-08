# cryoFILTER App Wrapper

The app wrapper is the recommended default entry point for cryoFILTER. It keeps
the command-line tools available for scripting, but gives external users a
single browser surface for launching CryoSPARC-integrated prediction, local mask
generation, existing-mask particle filtering, annotation, fine-tuning, and live
job monitoring.

Recommended remote-server workflow:

```bash
ssh -L 8766:127.0.0.1:8766 user@their-server
cd /path/to/writeable/project-directory
cryoFILTER app --port 8766
```

Then open this URL in a browser on the laptop that started the SSH session:

```text
http://127.0.0.1:8766/
```

The `-L` port forward is what makes the server-side `127.0.0.1` URL work from
the laptop browser.

The default Annotation tab is browser-native and does not need X forwarding or
Qt/XCB. Advanced > Editor > Desktop window keeps the older desktop annotation
GUI available. Use `ssh -Y` or `ssh -XY` and verify the server-side Python GUI
backend only for that fallback:

```bash
echo "$DISPLAY"
python -c "import PySide6; print('PySide6 ok')"
```

If using a source-only checkout instead of the conda environment and you want
the desktop fallback, install the optional GUI extra into the same Python used
to launch the wrapper:

```bash
pip install -e ".[gui]"
```

On Linux, errors mentioning the `xcb` Qt platform plugin usually mean a package
such as `libxcb-cursor0` is missing; use the browser editor, use the conda
environment, or ask an administrator to install the missing XCB runtime
libraries.

Launch from the directory where you want app state and relative outputs:

```bash
cryoFILTER app
```

Open:

```text
http://127.0.0.1:8765/
```

The app writes job records and logs under:

```text
.cryofilter_app/
```

The lowercase command is equivalent if preferred: `cryofilter app`.

## Current Tabs

- CryoSPARC: stage micrographs/particles from a CryoSPARC workspace, run
  `cryoFILTER`, and push accepted/rejected particles plus diagnostics back to
  a CryoSPARC External Job. This is the first tab because it is the expected
  processing-server workflow.
- Inference: generate probability maps and binary contamination masks from
  micrographs. Particle files are optional; leaving Particles blank runs mask
  generation only.
- Filter Picks: filter a CryoSPARC `.cs` or RELION `.star` particle file using
  pre-existing `<micrograph-stem>_mask.npy` files without rerunning neural
  network inference.
- Monitor: live job list, command log, throughput metrics, and output artifact
  previews.
- Annotation: creates or resumes an annotation manifest from local
  motion-corrected micrographs, a CryoSPARC micrograph output, or a previous
  edited manifest, then opens a browser-native annotation canvas.
- Fine-tune: wraps the recommended full-micrograph fine-tuning script, prepares
  a training manifest from saved annotation rows, and can infer micrographs
  from the edited manifest or reuse a CryoSPARC micrograph output.

## CryoSPARC Tab

The default CryoSPARC tab first asks for the instance URL from the user's
browser plus username/email and password. After the connection check succeeds,
the run form unlocks and asks for project/workspace IDs, micrograph/particle
output refs, and a checkpoint path. Advanced bridge, SSH, split host/port,
staging, and typing override settings stay behind the Advanced disclosure.

When the wrapper is launched on the processing server, Advanced can remain at
its default local bridge mode. cryoFILTER runs the bundled Python bridge module
directly and does not require users to know about bridge commands.

The CryoSPARC tab defaults to the validated `balanced` inference profile.
Advanced CPU/GPU resource controls can set the local subprocess thread count
and visible CUDA devices. CryoSPARC-backed inference still writes individual OTF
PNG overlays, but skips the growing inference-time contact sheet refresh;
diagnostics builds the top overlay sheet after inference.

## Annotation Tab

The Annotation tab has three source modes:

- Micrographs: point at one `.mrc/.mrcs` file or a directory of
  motion-corrected micrographs. The wrapper writes `source_manifest.csv` in the
  annotation session directory before launching the editor.
- CryoSPARC output: after connecting in the CryoSPARC tab, provide a project,
  workspace, and motion-correction or curation micrograph output ref such as
  `J12` or `J12:micrographs`. The wrapper stages the micrographs, builds the
  manifest, and launches the editor. By default this is manifest-only: it uses
  the original CryoSPARC project source paths and does not copy the full
  micrograph output. Enable Advanced > Copy micrographs only when the app
  server cannot read those source paths directly.
- Resume manifest: load a previous `manifest_edited.csv`; by default, edits
  continue in that same session directory.

Recent browser annotation sessions from the current app workspace appear at the
top of the tab. Resume opens the selected session and jumps to the first
unsaved micrograph, so closing the browser or stopping the app does not require
manual row tracking. If a session is moved or the app-state directory is not
available, use Source > Resume manifest and select the previous
`manifest_edited.csv`.

Generated manifests include a deterministic row-level `split` column so a
single CryoSPARC workspace can still have train/validation filtering and later
fine-tuning without inventing multiple dataset IDs.

The browser editor streams normalized PNG previews from the app server, stores
polygon vectors in `browser_annotations/`, writes binary masks to `masks/`,
writes subtype-compatible labels to `typed_labels/`, and keeps
`manifest_edited.csv` current for the Fine-tune tab. Action > Add mask marks
contamination, while Action > Erase mask subtracts after you draw and save.
Polygon mode adds vertices by clicking, while Circle / ellipse mode draws a
stretched oval by dragging across rounded contamination. Existing ellipses show
center and corner handles in Circle / ellipse mode; drag the center to move the
shape or drag a corner to rotate it, then Save to rewrite the row mask.
Overlapping add masks render as one combined overlay and are selected/deleted as
a connected group. To remove a completed shape, click the mask region to select
it, press Delete or Backspace, then Save. Contrast, brightness, gamma, and
invert controls are display-only and can be adjusted per viewed micrograph.
Advanced > Editor > Desktop window keeps the older matplotlib
editor available for users who want that workflow; the desktop fallback needs X
forwarding and a working Qt/Tk backend.

If CryoSPARC annotation reports that staged micrographs exceed the transfer GB
cap, the app is either running an older build or Advanced > Copy micrographs is
enabled. Leave Copy micrographs off for normal worker/shared-filesystem use.

## Existing-Mask Particle Filtering

Use Filter Picks when masks already exist and the user wants to try new particle
picks, larger pick sets, different exclusion distances, or re-exported
CryoSPARC/RELION files. The required inputs are the original micrographs, the
mask directory, and the particle file. cryoFILTER reads the original micrograph
headers for pixel size and matches particles to masks by micrograph filename.

The input particle file is never modified. Filtered `.cs`, `.csg`, and `.star`
outputs appear in Monitor artifacts.

## Troubleshooting

If CryoSPARC Tools reports `Cannot specify host or base_port when base_url is
specified`, clear stale CryoSPARC environment variables before relaunching the
wrapper:

```bash
unset CRYOFILTER_CRYOSPARC_API_HOST
unset CRYOFILTER_CRYOSPARC_BASE_PORT
unset CRYOFILTER_CRYOSPARC_BASE_URL
unset CRYOSPARC_HOST
unset CRYOSPARC_BASE_PORT
unset CRYOSPARC_BASE_URL
```

The wrapper accepts a simple Instance URL in the visible form and converts it
to a single clean CryoSPARC connection shape internally. A direct command should
also use either `--cryosparc-base-url` or the split
`--cryosparc-host`/`--cryosparc-base-port` options, not both.

The header GPU badge only reports whether the app can see a CUDA/Slurm GPU
signal or `nvidia-smi`. Actual inference still depends on the active Python
environment having a CUDA-enabled PyTorch build. If `torch.cuda.is_available()`
is false, use CPU for a smoke test or launch the wrapper from a CUDA PyTorch
environment.

## Boundary

The wrapper does not store CryoSPARC passwords in job metadata or logs. The
CryoSPARC tab can pass a password directly to the launched job environment, or
sites can use a `cryosparc.toml`, token login, or environment configuration.
