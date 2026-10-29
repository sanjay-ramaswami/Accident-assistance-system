const test = require("node:test");
const assert = require("node:assert/strict");
const { buildEnglishOperatorSummary, isEnglishSummary, selectOperatorUnknowns } = require("./summary-utils.js");

test("Human Desk falls back to assessment unknowns and deduplicates them", () => {
  assert.deepEqual(
    selectOperatorUnknowns({ assessmentMissing: ["breathing_status", "location_candidate", "breathing_status"] }),
    ["breathing_status", "location_candidate"]
  );
});

test("current triage unknowns take precedence over stale assessment fields", () => {
  assert.deepEqual(
    selectOperatorUnknowns({ triageMissing: ["victim_count"], assessmentMissing: ["location_candidate"] }),
    ["victim_count"]
  );
});

test("Malayalam caller speech stays separate from the English operator summary", () => {
  const callerTranscript = "എന്റെ അമ്മയ്ക്ക് ശ്വാസം എടുക്കാൻ കഴിയുന്നില്ല";
  const summary = buildEnglishOperatorSummary({ summaries: [], issueType: "MEDICAL", location: "Kochi" });

  assert.equal(callerTranscript, "എന്റെ അമ്മയ്ക്ക് ശ്വാസം എടുക്കാൻ കഴിയുന്നില്ല");
  assert.equal(summary, "Medical reported. Location: Kochi.");
  assert.equal(isEnglishSummary(summary), true);
});

test("Hindi caller speech is not used as the English summary fallback", () => {
  const callerTranscript = "मेरी माँ को साँस लेने में परेशानी हो रही है";
  const summary = buildEnglishOperatorSummary({ summaries: [], issueType: null, location: "Pending" });

  assert.equal(callerTranscript, "मेरी माँ को साँस लेने में परेशानी हो रही है");
  assert.equal(summary, "Awaiting a finalized caller description. No incident facts have been confirmed yet.");
});

test("empty live session summary does not claim caller details arrived", () => {
  assert.equal(
    buildEnglishOperatorSummary(),
    "Awaiting a finalized caller description. No incident facts have been confirmed yet."
  );
});

test("English assistant operator summaries are retained and Indic summaries are not", () => {
  assert.equal(buildEnglishOperatorSummary({ summaries: ["Caller reports difficulty breathing."] }), "Caller reports difficulty breathing.");
  assert.equal(isEnglishSummary("ശ്വാസതടസ്സം റിപ്പോർട്ട് ചെയ്തു"), false);
  assert.equal(isEnglishSummary("साँस लेने में परेशानी"), false);
  assert.equal(buildEnglishOperatorSummary({ summaries: ["साँस लेने में परेशानी"], issueType: "MEDICAL" }), "Medical reported.");
});
