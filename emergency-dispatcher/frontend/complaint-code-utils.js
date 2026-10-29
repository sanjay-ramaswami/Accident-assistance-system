(function (root) {
  function presentComplaintCodeResult(payload) {
    const code = String(payload?.input_code || "").trim();
    const category = String(payload?.prediction?.chapter || "").trim();
    if (!code || !category) throw new Error("The classifier response did not include a code and predicted category.");

    const seenInTraining = payload.training_coverage === "seen_in_training";
    const unseenCode = payload.training_coverage === "unseen_code";
    const recognition = seenInTraining
      ? "Seen in model training data"
      : unseenCode
        ? "Not seen in model training data"
        : "Training coverage not provided";
    const score = Number(payload.uncalibrated_model_score);
    const scoreText = Number.isFinite(score)
      ? score.toFixed(4)
      : "Not provided";
    const uncertainty = seenInTraining
      ? "The model score is uncalibrated and must not be read as confidence or a clinical probability. Human review is required."
      : unseenCode
        ? "This valid code was not in the model's training-code index. Treat this prediction as uncertain; human review is required."
        : "Training coverage information is unavailable. The model score is not calibrated confidence; human review is required.";

    return [
      `Input complaint code: ${code}`,
      `Recognition status: ${recognition}`,
      `Predicted broad clinical-functional category: ${category}`,
      `Uncalibrated model score (not confidence): ${scoreText}`,
      uncertainty,
    ].join("\n");
  }

  const api = { presentComplaintCodeResult };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.DispatcherComplaintCodeUtils = api;
})(typeof window !== "undefined" ? window : globalThis);
