const test = require("node:test");
const assert = require("node:assert/strict");
require("./session-report-utils.js");

const { postSessionReport, describeSessionReportError, sessionStatusFields, utf8Length } = globalThis.DispatcherSessionReportUtils;

test("live reports omit ended_at and terminal reports preserve their final status", () => {
  assert.deepEqual(sessionStatusFields("live", "2026-10-30T00:00:00Z"), { status: "live" });
  assert.deepEqual(sessionStatusFields("stopped", "2026-10-30T00:00:00Z"), {
    status: "stopped",
    ended_at: "2026-10-30T00:00:00Z",
  });
});

const successfulResponse = (payload = { status: "ok", incident_assessment: { session_id: "s1" } }) => ({
  ok: true,
  status: 200,
  text: async () => JSON.stringify(payload),
});

test("large live report is posted without the keepalive body limit", async () => {
  const report = { session_id: "s1", status: "live", transcript: "x".repeat(70000) };
  let options;
  const result = await postSessionReport({
    fetchImpl: async (_url, requestOptions) => {
      options = requestOptions;
      return successfulResponse();
    },
    url: "/api/session-report",
    payload: report,
  });

  assert.equal(result.requestBytes > 64 * 1024, true);
  assert.equal(options.keepalive, undefined);
  assert.equal(result.payload.status, "ok");
});

test("transient network failure retries once and then recovers", async () => {
  let calls = 0;
  const result = await postSessionReport({
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) throw new TypeError("Failed to fetch");
      return successfulResponse();
    },
    url: "/api/session-report",
    payload: { session_id: "s1" },
    retries: 1,
  });

  assert.equal(calls, 2);
  assert.equal(result.attempts, 2);
});

test("transient server errors retry within the configured bound", async () => {
  let calls = 0;
  const result = await postSessionReport({
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) return { ok: false, status: 503 };
      return successfulResponse();
    },
    url: "/api/session-report",
    payload: { session_id: "s1" },
    retries: 1,
  });

  assert.equal(calls, 2);
  assert.equal(result.payload.status, "ok");
});

test("timeout, permanent HTTP error, malformed JSON, and bad schema stay distinct", async () => {
  await assert.rejects(
    postSessionReport({
      fetchImpl: (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")))),
      url: "/api/session-report",
      payload: { session_id: "s1" },
      timeoutMs: 5,
      retries: 0,
    }),
    (error) => error.kind === "timeout" && describeSessionReportError(error) === "request timed out"
  );
  await assert.rejects(
    postSessionReport({ fetchImpl: async () => ({ ok: false, status: 422 }), url: "/api/session-report", payload: {}, retries: 0 }),
    (error) => error.kind === "http" && error.status === 422
  );
  await assert.rejects(
    postSessionReport({ fetchImpl: async () => ({ ok: true, text: async () => "not-json" }), url: "/api/session-report", payload: {} }),
    (error) => error.kind === "malformed_response"
  );
  await assert.rejects(
    postSessionReport({ fetchImpl: async () => ({ ok: true, text: async () => JSON.stringify({ status: "ok", triage_update: null }) }), url: "/api/session-report", payload: {} }),
    (error) => error.kind === "invalid_schema" &&
      error.missingField === "incident_assessment" &&
      describeSessionReportError(error) === "server response omitted the incident assessment"
  );
});

test("request byte count uses UTF-8 and does not expose payload text in errors", async () => {
  const result = await postSessionReport({
    fetchImpl: async () => successfulResponse(),
    url: "/api/session-report",
    payload: { note: "മലയാളം" },
  });
  assert.equal(result.requestBytes, utf8Length(JSON.stringify({ note: "മലയാളം" })));
});
