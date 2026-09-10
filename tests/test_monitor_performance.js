// Run with: node tests/test_monitor_performance.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../cryofilter/app/static/app.js"), "utf8");
const elements = new Map();
const context = vm.createContext({
  console, URLSearchParams,
  document: { querySelector: (name) => elements.get(name), querySelectorAll: () => [] },
  window: { addEventListener: () => {} },
});
vm.runInContext(source.slice(0, source.lastIndexOf("boot().catch(")), context);

async function main() {
  const job = { kind: "cryosparc_predict", status: "running", started_at: "2026-09-10T00:00:00Z" };
  const log = "Using 2 GPUs for 60 micrograph(s)\nprogress: 5/30 complete; elapsed_s=100; last_s=20\n";
  const summary = { available: true, source: "typing", n_images_completed: 2, n_images_total: 60,
    inference_progress: { source: "multi_gpu_workers", n_images_completed: 11, n_images_total: 60 } };
  assert.equal(context.latestMicrographProgress(job, log, summary).completed, 11);
  assert.equal(context.latestMicrographProgress(job, log, summary).total, 60);
  assert.match(context.liveSummaryMeta(summary, job), /2\/60 images typed/);
  assert.doesNotMatch(context.liveSummaryMeta(summary, job), /whole data set/);
  assert.equal(context.monitorPhase(job, log + "Phase: GPU inference.", summary), "Inference");
  assert.equal(context.monitorPhase(job, log + "Phase: GPU inference.\nLive typing update:", summary), "Inference + live typing");
  const complete = log + "Finished inference on 60 micrograph(s).\nPhase: GPU inference complete in 300.000s; final typing/finalization.";
  assert.equal(context.monitorPhase(job, complete, summary), "Final typing/finalization");
  assert.match(context.timePerMicrographMetric(job, complete, summary).value, /avg 5.00s.*60\/60/);
  assert.equal(context.monitorPhase(job, complete + "\nPhase: upload/register results to CryoSPARC.", summary), "Upload/register results");
  assert.equal(context.monitorPhase({ ...job, status: "succeeded" }, complete, summary), "done");

  // A slow API request must not cause setInterval to queue more monitor polls.
  let release;
  let requests = 0;
  context.api = () => { requests++; return new Promise((resolve) => { release = resolve; }); };
  context.renderJobs = () => {};
  context.renderLiveIndicator = () => {};
  context.renderSelected = async () => {};
  const first = context.loadJobs();
  await context.loadJobs();
  assert.equal(requests, 1);
  release({ jobs: [] });
  await first;
  context.api = async () => { throw new Error("temporary failure"); };
  await assert.rejects(context.loadJobs(), /temporary failure/);
  context.api = async () => { requests++; return { jobs: [] }; };
  await context.loadJobs();
  assert.equal(requests, 2, "failed polls must release the in-flight guard");

  // Repeated artifact responses keep existing image DOM nodes intact.
  let writes = 0;
  let markup = "";
  const grid = { hidden: false, get innerHTML() { return markup; },
    set innerHTML(value) { writes++; markup = value; } };
  elements.set("#artifactGrid", grid);
  vm.runInContext("state.selected = 'run-1'", context);
  const artifact = { suffix: ".png", relative_path: "OTF_images/mic.png", url: "/artifact/mic.png", mtime: 1 };
  context.api = async () => ({ artifacts: [artifact] });
  await context.renderArtifacts();
  await context.renderArtifacts();
  assert.equal(writes, 1);
  artifact.mtime = 2;
  await context.renderArtifacts();
  assert.equal(writes, 2);
  context.api = () => new Promise((resolve) => { release = resolve; });
  const stale = context.renderArtifacts();
  vm.runInContext("state.selected = 'run-2'", context);
  release({ artifacts: [] });
  await stale;
  assert.equal(writes, 2, "a stale run response must not replace the current run's images");
  console.log("Monitor performance and phase checks passed.");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
