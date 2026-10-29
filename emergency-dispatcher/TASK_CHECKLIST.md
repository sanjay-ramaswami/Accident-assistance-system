# Engineering Task Checklist — Emergency Dispatcher

Baseline (2026-10-29): pytest **161 passed**; node --check app.js/audio-utils.js/review.js passed; no listener on :8001; Python 3.12.11, sklearn 1.9.1, fastapi 0.162.0.
Current (2026-10-29, after this work): pytest **176 passed** (161 baseline + 15 new), 0 failed.

Status legend: [ ] pending, [~] in progress, [x] done, [!] blocked

## Phase 0 — Audit
- [x] Directory structure and docs read
- [x] IMPLEMENTATION_REPORT.md read
- [x] Baseline test suite run (161 passed)
- [x] Source audit: routes, adapters, first-aid, location, dispatch, frontend

## Phase 1 — Complaint-code model
- [x] Model artifact loads in clean process; metadata verified
- [x] Training script reproduces saved model (retrain to temp paths: identical split/metrics/predictions vs saved artifact)
- [x] Unknown/malformed/missing code handling + validation limits (tests pass)
- [x] Endpoint contract, dispatch_authorized=false, human review (tests pass)
- [x] Dashboard panel real response rendering (loading/error present in app.js)

## Phase 2 — Symptom severity model
- [x] Investigate legitimate public datasets (licence, labels, access)
  - Mendeley dd359z6trk: downloaded+inspected — 111 MCI scenario rows, expert text embeds label (leakage) → unusable
  - HF fedmml-ed-triage: requires accepting access conditions + login → blocked (user action needed)
  - Kaggle ilkeryildiz triage: requires Kaggle account/ToS; no local credentials → blocked
  - Zenodo 14058516: files Restricted → blocked
  - Zenodo 1308993 (Yale): target = admission outcome (post-outcome), not urgency → not relabelled
  - PhysioNet (MIETIC/MIMIC-IV/eICU): CITI training + DUA required → not accessed
- [x] Conclusion: no suitable labelled urgency data obtainable without user-side terms acceptance → documented blocker, no model forced

## Phase 3 — Severity inference integration (conditional)
- [x] Not performed — gated on Phase 2; no dataset obtainable → no service built (documented blocker)

## Phase 4 — First-aid safety
- [x] Critical-symptom rules present and deterministic (10 scenarios, tests pass)
- [x] Regression tests for overrides and ambiguous inputs (tests pass)
- [x] Source spot-check vs authoritative references (RCUK 2025 BLS fetched + confirmed; Red Cross 403 = WAF, documented)

## Phase 5 — Live voice repair
- [x] Root cause confirmed with evidence: OpenAI 429 insufficient_quota/credit_balance_exhausted; gradbot native swallows the error (logs only to native stderr), server never sees it → silent session → historical 1011 EOF
- [x] One-shot LLM pre-flight probe before session start; quota refusal path (close 1011 + explicit `llm_quota_exhausted` error)
- [x] Runtime watchdog: push_to_llm cycles with zero llm_started → `llm_no_response` terminal error (no silent retry)
- [x] Frontend surfaces server-sent `error` frames verbatim (code + message) instead of generic audio handler
- [x] Tests: pre-flight (ok/quota/non-quota/server/transport), quota refusal, watchdog terminate, watchdog-stays-silent-on-healthy, duplicate EOT, EOF, text-only
- [x] Caller-leg (STT) verified live with synthetic speech (real transcripts); assistant leg blocked by quota (documented, not faked)

## Phase 6 — Location, geocoding, map
- [x] Full-name preservation, no transcript geocoding, ambiguity handling (source + mocked tests)
- [x] Mocked success/ambiguous/empty/invalid/failure tests (41 tests in test_dispatch_tools.py pass)

## Phase 7 — Dispatch lifecycle
- [x] Idempotency (unchanged update does not rerun simulator)
- [x] Duplicate end-of-turn not counted as assistant completion
- [x] Simulation labelling (SIMULATED DISPATCH ONLY; no real services contacted)
- [x] Stale/duplicate-session rejection (active-session guard: duplicate `start` with a live session_id rejected 1008; 3 new tests)
- [x] Duplicate-input / repeated-event tests

## Phase 8 — Frontend integration
- [x] Panels reflect real backend state; loading/error/offline states
- [x] Frontend failure-contract tests added (5 new) and audio-utils node tests pass (8)

## Phase 9 — Security/config
- [x] Secrets not exposed; env vars documented by name only (never printed)
- [x] Provider error messages sanitized (type/code only; no URLs/tokens retained)
- [x] Session_id validation; payload size limits present

## Phase 10 — Full verification
- [x] Full pytest: 176 passed
- [x] compileall OK; node --check OK (4 files); node --test 8 passed; pip check clean
- [x] Runtime verification: healthz, demo, audio-config, first-aid, complaint-code, browser demo + live WS quota frame (1011)
- [x] IMPLEMENTATION_REPORT.md updated with evidence (13 sections)
