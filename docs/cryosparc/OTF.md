# cryoFILTER-OTFwMC

The OTF tab follows CryoSPARC 5 Patch Motion Correction while it runs. It creates
a cryoFILTER External Job card, keeps segmentation models loaded, and publishes
each binary mask as soon as its micrograph is processed. Particle picking and
typing do not block mask availability.

## Run from the UI on Silva

1. Launch the updated cryoFILTER app on the server that can read the CryoSPARC
   project files. Use the existing CryoSPARC connection form to connect.
2. Queue Patch Motion Correction in CryoSPARC. Open **OTF** and enter its project,
   workspace and motion-correction job number.
3. Enter the GPU IDs allocated to cryoFILTER, separately from the GPUs used by
   motion correction. With one GPU, typing is disabled. With multiple GPUs,
   leave **Run typing** unchecked to dedicate all GPUs to segmentation, or enable
   it to divide the GPUs between segmentation and typing. For three GPUs the
   first handles segmentation and the other two handle typing. IDs must belong
   to the current CUDA_VISIBLE_DEVICES allocation when that variable is set.
4. Select segmentation weights, a CPU budget and a persistent output root.
   Typing uses the existing publication classifier and fast stride 64 by default;
   balanced stride 32 and highest-accuracy stride 16 remain available.
5. Start OTF. The monitor and log show the new CryoSPARC card number, segmented
   and typed counts, queue sizes and processing time per worker. The card also
   receives progress logs and sampled preview images. During processing, live
   images refresh on the card and in cryoFILTER. In the **Event Log**, select
   **Follow latest** to see a live gallery of up to ten micrographs, newest first.
   Each changed gallery starts a new checkpoint, and Follow latest automatically
   switches to it. Previous galleries remain accessible in checkpoint history.
6. In **Filter Picks**, select **CryoSPARC OTF job**, enter that card number and a
   completed particle job from the same project, then choose an exclusion distance.
   No mask-directory entry is needed. Bare particle job numbers and explicit
   outputs such as J20:particles are supported.

The new filtering card exposes particles_accepted and particles_rejected.
Choose the accepted output for downstream processing. Filtering retains particle
UIDs, order and the source metadata through CryoSPARC passthrough.

The OTF card can still be running when filtering starts. The default missing-mask
policy stops filtering if any particle's mask is unfinished, unavailable or has
changed. **Keep unprocessed separately** publishes those particles as
particles_pending, alongside the accepted and rejected outputs. Rerun filtering
when more masks are available to create a fresh snapshot.

The **Local files** option retains the existing .cs/.star filtering workflow.
Packed OTF masks are resolved through the OTF-card workflow.

## Discovery, storage and recovery

The runner reads only the motioncorrected directory and accepts dose-weighted
Patch Motion Correction filenames whose UID belongs to the input movies.
It waits for four seconds of unchanged size/mtime and a valid, complete MRC
header/payload before scheduling an image. This is a conservative file-readiness
check; final CryoSPARC micrograph paths are reconciled when motion correction
finishes. Late arrivals and queued typing are drained before the OTF card completes.

Micrographs are read in place. Full-resolution binary masks are losslessly packed
and compressed as NPZ files; filtering uses these exact masks. Probability arrays
are float16 previews with a maximum long edge of 1024 pixels. Typed maps retain
full mask resolution. Monitor summaries come from a small SQLite index, and PNG
previews refresh at most once every 15 seconds when results change, plus a final
update. The local gallery shows the most recently segmented 1–10 micrographs,
with typing panels added as those results become available. Delayed typing does
not reorder the micrographs. Rendering and uploads run in the background, reuse
cached previews, and submit PNG galleries for inline display in CryoSPARC while
processing is still running. Each checkpoint contains one gallery and its own
updating progress message. Identical gallery contents do not add another image
or checkpoint, including the final flush and unchanged resumes.
Exact masks remain lossless.

Raw display panels use area averaging before the existing 3 x 3 median cleanup
and 0.2–99.8% contrast stretch. Averaging suppresses sampling noise while keeping
the display processing small and fast. Mask and typing overlays use nearest
neighbor resizing to preserve their labels. This changes previews only; inference
inputs and stored masks retain their original data. Preview cache names include
the display version so resumed runs regenerate older pixel-skipped previews.

CryoSPARC 5's public API cannot replace an image event in place. Live galleries
therefore use the standard checkpoint mechanism. **Follow latest** shows the
newest checkpoint instead of displaying all galleries in one long view. Choosing
**Show from top** or an older checkpoint intentionally browses historical events.
This bounds the current visible gallery, not the stored history: image events and
their uploaded assets remain in CryoSPARC. The integration uses ordinary
cryoSPARC credentials and cryosparc-tools; no admin SSH access, server patches,
or installation changes are required. See the
[CryoSPARC Event Log controls](https://guide.cryosparc.com/application-guide-v4.0%2B/inspecting-job-data).

A small local state file records the current snapshot and event IDs. If an upload
response is lost, the runner checks the latest checkpoint before retrying so it
can recover the already published image. History is read only on first use or
recovery, not at every refresh. Progress updates use explicit event IDs so
checkpoint resets cannot redirect them to an older, hidden message. The local
latest_micrographs.png is replaced on refresh. Disabling live previews also
disables checkpoint/gallery uploads.

CryoSPARC card and Event Log exports use white backgrounds, black panel captions,
and PNG format. They have a separate hidden .cryosparc export cache; the cryoFILTER
UI retains its existing preview styling. Micrograph pixels, overlay colors,
inference inputs and masks are unaffected by the export theme.

The output root must remain accessible for later filtering and resume. Its
cryofilter_otf.json and index.sqlite files are referenced by metadata in the
CryoSPARC card directory. Moving or removing those outputs breaks card lookup.
The default storage budget is 25 GiB. Reservations conservatively include future
typing and preview overhead, so the runner may stop before actual disk usage
reaches the displayed budget. Completed masks remain available. No input movies
or micrographs are copied or modified.

Use the CLI to resume an interrupted run with the same source and model settings:

```bash
cryofilter cryosparc --host local --config /path/to/cryosparc.toml otf \
  --project P1 --workspace W2 --micrographs J12 \
  --gpu-devices 1 --num-cpus 8 \
  --checkpoint /path/to/cryoFILTER_FULL.pt \
  --local-run-root /path/to/cryofilter_runs/otf \
  --resume-job J13 --max-output-gb 50
```

Use your existing connection configuration/environment; credentials do not belong
in the command line. Include the original threshold, profile and typing options
if they differed from the defaults. GPU/CPU counts and the output budget can
change; enabling/disabling typing or changing the model recipe requires a new
card. A lock prevents two runners from owning the same card. Resume rechecks
cached file stamps and waits for file readiness before reprocessing invalid data.
Resume is currently exposed through the CLI.

Cancelling OTF stops its workers and marks its own card as interrupted; the Patch
Motion Correction job continues independently. A failed upstream job makes OTF
report failure while preserving completed masks for filtering.

## Silva acceptance test

Use a small new Patch Motion Correction job for the first live test:

1. Start OTF before motion correction finishes, with one GPU allocated to
   cryoFILTER. Confirm typing is disabled and segmented counts increase while the
   source card remains running.
2. Confirm one cryoFILTER card is created and its white preview refreshes. The
   cryoFILTER gallery should grow to ten and then show the newest ten. In the
   Event Log, select Follow latest: a white gallery should appear before motion
   correction finishes and the active checkpoint should advance as it updates.
   Only the current gallery should remain in that view; older ones should be
   available in checkpoint history. Let the source finish; the OTF count should
   match its final successful micrographs and the last gallery should include
   late typing results. Completion should not duplicate an unchanged gallery.
3. Run a picker, then filter via the OTF card number. Verify accepted plus
   rejected equals the source particle count and that the accepted output connects
   to downstream jobs. Inspect a boundary region at the chosen exclusion distance.
4. Repeat filtering while OTF is still processing: the default should report
   unfinished masks; the pending policy should publish a separate pending output.
5. Stop an OTF run and resume its card using the CLI. Completed, unchanged masks
   should not be recomputed. Repeat with two GPUs and typing enabled, checking
   that masks become available before typing catches up.

The implementation has local regression tests for incremental arrivals, readiness,
UID matching, mask equality, metadata passthrough, restart locking, cache
invalidation and storage reservations. The existing inference core also produces
bit-identical batch and streaming masks with a small test model, loading the
model once per stream. CryoSPARC publication tests use a fake API.

A local synthetic CPU benchmark across eight 1024 x 1024 masks processed 100,000
picks in 0.130 seconds versus 4.207 seconds for the existing file filter
(32.4x), with identical accepted particle UIDs. One million picks took 0.447
seconds. These are filtering measurements, not live acquisition or GPU inference
throughput claims.

Read-only inspection of a completed Silva Patch Motion Correction job confirmed
the expected dose-weighted filenames, padded UID convention, micrograph geometry
and final output layout. A real live GPU run and actual External Job publication
remain to be validated on Silva.
