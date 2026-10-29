(function (root) {
  const EXPLICIT_CODE = /\b(?:complaint\s+)?code(?:\s+(?:is|number))?\s*[:#-]?\s*([A-Z0-9][A-Z0-9.-]{0,23})\b/i;

  function extractExplicitComplaintCode(text) {
    const source = String(text || "");
    const match = source.match(EXPLICIT_CODE);
    // All supported codes in the verified training index contain a digit;
    // this prevents phrases such as “complaint code unavailable” becoming input.
    return match && /\d/.test(match[1]) ? match[1].toUpperCase() : null;
  }

  function createLiveComplaintClassifier({ fetchImpl, onUpdate }) {
    let version = 0;
    let activeController = null;
    let lastKey = "";

    async function finalize(text, sessionId) {
      const candidate = extractExplicitComplaintCode(text);
      if (!candidate) {
        version += 1;
        activeController?.abort();
        activeController = null;
        lastKey = "";
        onUpdate({ status: "awaiting", text: String(text || ""), code: null });
        return;
      }
      const dedupeKey = `${sessionId || ""}:${candidate}`;
      if (dedupeKey === lastKey) return;
      const currentVersion = ++version;
      activeController?.abort();
      activeController = null;
      lastKey = dedupeKey;
      const controller = new AbortController();
      activeController = controller;
      onUpdate({ status: "pending", text: String(text || ""), code: candidate });
      try {
        const response = await fetchImpl("/api/triage/complaint-code", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code: candidate }),
          signal: controller.signal,
        });
        const payload = await response.json();
        if (currentVersion !== version || controller.signal.aborted) return;
        if (!response.ok) {
          const failure = new Error(payload.detail || `Request failed (${response.status})`);
          failure.status = response.status;
          throw failure;
        }
        if (payload.training_coverage !== "seen_in_training") {
          onUpdate({ status: "awaiting", text: String(text || ""), code: candidate, reason: "Code is not covered by the model's verified training index." });
          return;
        }
        onUpdate({ status: "success", text: String(text || ""), code: candidate, payload, timestamp: new Date().toISOString() });
      } catch (error) {
        if (error?.name === "AbortError" || currentVersion !== version) return;
        onUpdate({
          status: error?.status === 503 || error?.name === "TypeError" ? "unavailable" : "error",
          text: String(text || ""),
          code: candidate,
          message: error?.message || "Classification unavailable.",
        });
      }
    }

    return {
      finalize,
      reset() {
        version += 1;
        activeController?.abort();
        activeController = null;
        lastKey = "";
      },
    };
  }

  const api = { extractExplicitComplaintCode, createLiveComplaintClassifier };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.DispatcherLiveMlUtils = api;
})(typeof window !== "undefined" ? window : globalThis);
