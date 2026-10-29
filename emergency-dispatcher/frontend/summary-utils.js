(function (root) {
  const INDIC_SCRIPT = /[\u0900-\u097F\u0D00-\u0D7F]/u;

  function isEnglishSummary(value) {
    const text = String(value || "").trim();
    return Boolean(text) && !INDIC_SCRIPT.test(text);
  }

  function englishFactLabel(value) {
    return String(value || "")
      .trim()
      .replaceAll("_", " ")
      .toLowerCase()
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function selectOperatorUnknowns({ triageMissing = [], assessmentMissing = [], handoffUnknowns = [] } = {}) {
    const source = triageMissing.length
      ? triageMissing
      : assessmentMissing.length
        ? assessmentMissing
        : handoffUnknowns;
    return [...new Set(source.map((value) => String(value || "").trim()).filter(Boolean))];
  }

  function buildEnglishOperatorSummary({ summaries = [], issueType, location } = {}) {
    for (const candidate of summaries) {
      if (isEnglishSummary(candidate)) return String(candidate).trim();
    }

    const facts = [];
    if (issueType && issueType !== "Pending" && isEnglishSummary(issueType)) {
      facts.push(`${englishFactLabel(issueType)} reported`);
    }
    if (location && location !== "Pending" && isEnglishSummary(location)) {
      facts.push(`Location: ${String(location).trim()}`);
    }
    if (facts.length) return `${facts.join(". ")}.`;
    return "Awaiting a finalized caller description. No incident facts have been confirmed yet.";
  }

  const api = { isEnglishSummary, buildEnglishOperatorSummary, selectOperatorUnknowns };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.DispatcherSummaryUtils = api;
})(typeof window !== "undefined" ? window : globalThis);
