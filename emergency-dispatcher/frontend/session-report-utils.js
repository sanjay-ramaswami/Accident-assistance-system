(function (root) {
  const TRANSIENT_HTTP_STATUSES = new Set([502, 503, 504]);

  class SessionReportError extends Error {
    constructor(kind, message, details = {}) {
      super(message);
      this.name = "SessionReportError";
      this.kind = kind;
      Object.assign(this, details);
    }
  }

  function utf8Length(value) {
    return new TextEncoder().encode(value).length;
  }

  function sessionStatusFields(status, timestamp = new Date().toISOString()) {
    return status === "live" ? { status } : { status, ended_at: timestamp };
  }

  function isValidResponse(payload) {
    return Boolean(
      payload &&
      typeof payload === "object" &&
      typeof payload.status === "string" &&
      (payload.incident_assessment || payload.stale_report_ignored === true)
    );
  }

  async function postSessionReport({
    fetchImpl,
    url,
    payload,
    timeoutMs = 12000,
    retries = 1,
    onAttempt = () => {},
  }) {
    if (typeof fetchImpl !== "function") throw new TypeError("fetchImpl is required");
    const body = JSON.stringify(payload);
    const requestBytes = utf8Length(body);
    let lastError;

    for (let attempt = 0; attempt <= retries; attempt += 1) {
      const controller = new AbortController();
      let timedOut = false;
      const timeout = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs);
      onAttempt({ attempt: attempt + 1, requestBytes });
      try {
        const response = await fetchImpl(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
          signal: controller.signal,
        });
        if (!response.ok) {
          const error = new SessionReportError("http", "Session report request failed.", {
            status: response.status,
          });
          if (TRANSIENT_HTTP_STATUSES.has(response.status) && attempt < retries) {
            lastError = error;
            await new Promise((resolve) => setTimeout(resolve, 200 * (attempt + 1)));
            continue;
          }
          throw error;
        }

        let parsed;
        try {
          parsed = JSON.parse(await response.text());
        } catch {
          throw new SessionReportError("malformed_response", "Session report response was not valid JSON.");
        }
        if (!isValidResponse(parsed)) {
          const missingField =
            parsed && typeof parsed === "object" && !parsed.incident_assessment && parsed.stale_report_ignored !== true
              ? "incident_assessment"
              : null;
          throw new SessionReportError("invalid_schema", "Session report response did not match the expected schema.", {
            missingField,
          });
        }
        return { payload: parsed, attempts: attempt + 1, requestBytes };
      } catch (error) {
        if (error instanceof SessionReportError) throw error;
        const kind = timedOut ? "timeout" : error?.name === "AbortError" ? "cancelled" : error instanceof TypeError ? "network" : "unknown";
        const classified = new SessionReportError(kind, "Session report request did not complete.", {
          causeName: error?.name || "Error",
        });
        if ((kind === "network" || kind === "timeout") && attempt < retries) {
          lastError = classified;
          await new Promise((resolve) => setTimeout(resolve, 200 * (attempt + 1)));
          continue;
        }
        throw classified;
      } finally {
        clearTimeout(timeout);
      }
    }
    throw lastError || new SessionReportError("unknown", "Session report request did not complete.");
  }

  function describeSessionReportError(error) {
    if (error?.kind === "timeout") return "request timed out";
    if (error?.kind === "network") return "network unavailable after retry";
    if (error?.kind === "cancelled") return "request cancelled";
    if (error?.kind === "malformed_response") return "malformed server response";
    if (error?.kind === "invalid_schema") {
      return error.missingField === "incident_assessment"
        ? "server response omitted the incident assessment"
        : "unexpected server response";
    }
    if (error?.kind === "http") return `server returned HTTP ${error.status}`;
    return "request failed";
  }

  root.DispatcherSessionReportUtils = { postSessionReport, describeSessionReportError, sessionStatusFields, utf8Length };
})(typeof window !== "undefined" ? window : globalThis);
