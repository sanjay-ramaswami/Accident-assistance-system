# Runtime reliability notes

## Session-report network failure

The dashboard posts a rolling session report about every two seconds. The
payload includes growing finalized transcripts and event records. The frontend
previously set Fetch `keepalive: true` on these ordinary live requests. Browser
Fetch limits a keepalive request body to 64 KiB ([MDN Fetch `RequestInit`](https://developer.mozilla.org/en-US/docs/Web/API/RequestInit)). Stored session client-report
sections in this workspace reached 65,799 and 67,164 bytes, enough to exceed
that limit and make the browser report a network error before FastAPI receives
the request. Live reports now use an ordinary bounded-time fetch. `sendBeacon`
is reserved for small unload reports, and oversized unload reports are marked
not queued rather than reported as saved.

Transient network errors/timeouts and 502/503/504 responses are retried once.
After three consecutive failed report operations, automatic sync pauses and
shows a manual retry button. HTTP status, timeout, network failure, malformed
JSON, and unexpected response schema are shown as distinct safe categories.
Diagnostics include attempt count and byte length, not transcript text.

## Map reset

The old `updateMap()` path removed the marker and centered India whenever a
render had no valid latitude/longitude. Session polls and other state updates
can temporarily omit location coordinates, so an unrelated partial update
could erase a previously valid pin. The map now retains the last valid pin for
the active session, ignores lower-confidence and stale revisions, and labels
when it is retaining an earlier location. It clears the pin only on a new
session or explicit console clear.

## Provider timeout

The former 15-second timeout displayed a synthetic assistant reply when no
provider text/audio had arrived. Live calls now display only a timeout status;
they do not insert or speak a synthetic reply. Offline deterministic scenarios
remain separate and labeled as simulated.

## Verification boundary

Frontend helper tests cover oversized report bodies, bounded retry, timeout,
HTTP, malformed JSON, and response-schema errors. Map helper tests cover partial
updates, confidence, session reset, and stale revisions. These tests do not
replace a live browser/Gemini timing run. The frontend records monotonic
caller-turn stages so a live run can compare caller finalization, provider
response, returned audio, enqueue, and playback start.
