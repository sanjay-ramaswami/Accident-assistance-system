const test = require("node:test");
const assert = require("node:assert/strict");
const { presentComplaintCodeResult } = require("./complaint-code-utils.js");

test("known complaint-code result shows category, recognition, and uncalibrated score", () => {
  const result = presentComplaintCodeResult({
    input_code: "R07.9",
    prediction: { chapter_id: "1", chapter: "Cardiorespiratory" },
    uncalibrated_model_score: 0.6971389863534136,
    training_coverage: "seen_in_training",
  });

  assert.match(result, /Input complaint code: R07\.9/);
  assert.match(result, /Seen in model training data/);
  assert.match(result, /Predicted broad clinical-functional category: Cardiorespiratory/);
  assert.match(result, /Uncalibrated model score \(not confidence\): 0\.6971/);
  assert.match(result, /must not be read as confidence/);
});

test("valid unseen codes are marked unknown to training while retaining the actual prediction", () => {
  const result = presentComplaintCodeResult({
    input_code: "ZZZ.999",
    prediction: { chapter_id: "8", chapter: "Other complaints" },
    uncalibrated_model_score: 0.42082018495797324,
    training_coverage: "unseen_code",
  });

  assert.match(result, /Not seen in model training data/);
  assert.match(result, /Other complaints/);
  assert.match(result, /Treat this prediction as uncertain/);
});

test("malformed or missing model response does not create invented output", () => {
  assert.throws(() => presentComplaintCodeResult(null), /did not include a code and predicted category/);
  const noScore = presentComplaintCodeResult({
    input_code: "R07.9",
    prediction: { chapter: "Cardiorespiratory" },
    training_coverage: "seen_in_training",
  });
  assert.match(noScore, /Uncalibrated model score \(not confidence\): Not provided/);
});
