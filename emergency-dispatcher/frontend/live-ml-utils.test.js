const test = require("node:test");
const assert = require("node:assert/strict");
const { extractExplicitComplaintCode, createLiveComplaintClassifier } = require("./live-ml-utils.js");

test("only an explicitly stated complaint code is extracted, never a symptom description", () => {
  assert.equal(extractExplicitComplaintCode("I have chest pain and feel faint."), null);
  assert.equal(extractExplicitComplaintCode("My complaint code is r07.9."), "R07.9");
  assert.equal(extractExplicitComplaintCode("code: 101.3"), "101.3");
  assert.equal(extractExplicitComplaintCode("complaint code unavailable"), null);
  assert.equal(extractExplicitComplaintCode("my complaint code is chest pain"), null);
});

test("duplicate finalized utterances do not send repeated model requests", async () => {
  let calls = 0;
  const updates = [];
  const classifier = createLiveComplaintClassifier({
    fetchImpl: async () => {
      calls += 1;
      return { ok: true, json: async () => ({ training_coverage: "seen_in_training", input_code: "R07.9", prediction: { chapter: "Cardiorespiratory" } }) };
    },
    onUpdate: (update) => updates.push(update),
  });
  await classifier.finalize("My complaint code is R07.9", "session-a");
  await classifier.finalize("My complaint code is R07.9", "session-a");
  assert.equal(calls, 1);
  assert.equal(updates.filter((item) => item.status === "success").length, 1);
});

test("duplicate arriving while inference is pending does not cancel the in-flight request", async () => {
  let calls = 0;
  let release;
  const updates = [];
  const classifier = createLiveComplaintClassifier({
    fetchImpl: () => {
      calls += 1;
      return new Promise((resolve) => { release = resolve; });
    },
    onUpdate: (update) => updates.push(update),
  });
  const pending = classifier.finalize("Complaint code R07.9", "session-a");
  await classifier.finalize("Complaint code R07.9", "session-a");
  release({ ok: true, json: async () => ({ training_coverage: "seen_in_training", input_code: "R07.9", prediction: { chapter: "Cardiorespiratory" } }) });
  await pending;
  assert.equal(calls, 1);
  assert.equal(updates.at(-1).status, "success");
});

test("stale inference is aborted and cannot overwrite newer transcript status", async () => {
  const updates = [];
  let releaseFirst;
  const classifier = createLiveComplaintClassifier({
    fetchImpl: (_url, options) => new Promise((resolve, reject) => {
      if (options.body.includes("R07.9")) {
        releaseFirst = () => resolve({ ok: true, json: async () => ({ training_coverage: "seen_in_training", input_code: "R07.9", prediction: { chapter: "Cardiorespiratory" } }) });
      } else {
        resolve({ ok: true, json: async () => ({ training_coverage: "unseen_code", input_code: "ZZZ.9", prediction: { chapter: "Other complaints" } }) });
      }
      options.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
    }),
    onUpdate: (update) => updates.push(update),
  });
  const stale = classifier.finalize("Complaint code R07.9", "session-a");
  await classifier.finalize("Complaint code ZZZ.9", "session-a");
  releaseFirst();
  await stale;
  assert.equal(updates.at(-1).status, "awaiting");
  assert.equal(updates.some((item) => item.status === "success"), false);
});

test("model failures are visible and reset invalidates pending results", async () => {
  let release;
  const updates = [];
  const classifier = createLiveComplaintClassifier({
    fetchImpl: () => new Promise((resolve) => { release = resolve; }),
    onUpdate: (update) => updates.push(update),
  });
  const pending = classifier.finalize("Code R07.9", "session-a");
  classifier.reset();
  release({ ok: false, status: 503, json: async () => ({ detail: "Model unavailable" }) });
  await pending;
  assert.equal(updates.at(-1).status, "pending");
  assert.equal(updates.some((item) => item.status === "error"), false);
});

test("unavailable model service is distinct from a classification error", async () => {
  const updates = [];
  const classifier = createLiveComplaintClassifier({
    fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({ detail: "Model unavailable" }) }),
    onUpdate: (update) => updates.push(update),
  });
  await classifier.finalize("Code R07.9", "session-a");
  assert.equal(updates.at(-1).status, "unavailable");
});
