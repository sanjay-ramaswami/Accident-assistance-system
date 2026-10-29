# Emergency Dispatcher Implementation Report

Date: 2026-10-29 (completion pass)
Prototype status: integrated and testable. Live **caller speech→text (STT)** and all
local/HTTP features verified; live **assistant speech (LLM→TTS→playback)** is blocked by
an external account-quota limit and is surfaced explicitly rather than silently.

---

## 1. Architecture

The app is a FastAPI service (`src/emergency_dispatcher/server.py`) that serves a
single-page dashboard (`frontend/`) and a set of JSON APIs. The WebSocket `/ws`
(`_websocket_session_ai_coustics`) drives a live call through the `gradbot` native SDK,
which multiplexes Gradium STT, an OpenAI-compatible LLM, and AI-Coustics audio
enhancement over PCM audio.

Components:

| Layer | Module | Role |
| --- | --- | --- |
| HTTP API | `server.py` | `/healthz`, `/demo`, `/api/audio-config`, `/api/first-aid`, `/api/triage`, `/api/triage/complaint-code`, `/api/session-report`, `/api/sessions[/{id}]` |
| Live voice | `server.py` + `gradbot_adapter.py` | `/ws` handler: pre-flight, provider start, input/output loops, watchdog, dispatch tool callbacks |
| First aid | `first_aid.py` | Deterministic, source-linked rules engine (no LLM, no persistence) |
| Complaint-code ML | `triage_ml.py` | Character n-gram TF-IDF + LogisticRegression classifier over ICD-style codes |
| Triage / ledger | `triage.py`, `slm_fact_ledger.py`, `slm_guidance.py` | Utterance triage, shared fact ledger, SLM guidance |
| Location / dispatch | `dispatch_tools.py` | Address extraction, geocoding (semantic Berlin index + Nominatim + Google), response bases, simulated dispatch |
| Session store | `session_store.py` | Per-session JSON persistence, recorder patch API |
| Frontend | `frontend/app.js`, `audio-utils.js` | Dashboard, WebSocket client, PCM/Opus pipeline, map, panels |

Safety invariants that are pinned by tests and never delegated to ML:

- First-aid severity and life-threatening overrides are deterministic
  (`first_aid.py`), independent of any classifier, and can only be raised, never
  lowered.
- The complaint-code model returns `dispatch_authorized=false` always and is advisory.
- Dispatch is **simulated** and labelled "SIMULATED DISPATCH ONLY"; no real emergency
  service is ever contacted (`LIVE_DISPATCH_MODE` gates provider tools only).
- The app never places a call; emergency-number text is generic.

## 2. Live-voice pipeline and root-cause repair (the main fix in this pass)

### Root cause (confirmed with evidence)

A previously recorded session showed caller STT text arriving, then silence, then
WebSocket close 1011, with 0 `llm_started` events. Direct probing reproduced and
explained it:

1. `POST {OPENAI_BASE_URL}/chat/completions` with the configured key returns
   **HTTP 429** with `error.type=insufficient_quota`, `error.code=credit_balance_exhausted`
   ("You have no credits remaining"). This is an **account-side blocker**, not a code bug.
2. The `gradbot` native SDK logs this failure only to its **native stderr**
   (`LLM: failed to push text … returned 429`) and does **not** forward it to the Python
   output stream. Verified: a Python logging capture handler recorded **0** records while
   the native trace printed the 429. The server therefore saw only silence and eventually
   an output-stream EOF → historical silent hang → 1011.

### Repair

Because the native layer swallows the error, the server now detects a dead LLM two ways
and surfaces it explicitly (`server.py`):

- **Pre-flight probe** (`_llm_preflight_check`): one one-token `chat/completions` request
  before the session starts. On quota exhaustion the server sends
  `{"type":"error","code":"llm_quota_exhausted",…}` and closes **1011** *before* starting a
  provider session, instead of beginning a call that cannot answer. Error `type`/`code`
  are retained; message bodies/URLs/tokens are discarded (sanitized).
- **Runtime watchdog** (`LLM_START_TIMEOUT_S=8.0`, `LLM_NO_RESPONSE_LIMIT=2`): counts
  `push_to_llm` cycles that produce no `llm_started`; after the limit it emits a terminal
  `{"type":"error","code":"llm_no_response",…}` and closes 1011 rather than retrying
  silently. `human_takeover_active` suppresses the watchdog during operator takeover.
- **Output loop** polls the native receive via a non-cancelling
  `asyncio.ensure_future` + `asyncio.wait(timeout=1.0)` so the watchdog can tick without
  ever cancelling a pending native receive; the task is cancelled after `gather`.
- **Frontend** (`app.js`) intercepts server `{"type":"error"}` frames and renders the
  code + message verbatim (status "Error", playback status, event log) instead of routing
  them to the generic audio handler that discarded them. `getSocketCloseOutcome` maps
  1011/1006/unclean to "Unavailable".

### Live evidence

- **Caller leg (STT) verified end-to-end** with synthetic speech (`scripts/probe_live_voice.py --wav`):
  real transcripts of the spoken utterance were returned (`'"Hello!'`, `'Please'`,
  `'help!"'`, `'There'`, `'has'`, `'been'`, `'a'`, `'car'`), with 8 `stt_text`, 15
  `push_to_llm`, 14 `end_of_turn` events. No assistant text/audio (quota-blocked).
- **Assistant leg (LLM→TTS)** produced 0 text / 0 audio — expected, blocked by the
  exhausted quota. Not fabricated.
- **Server-side runtime proof**: a real browser WebSocket against the running server
  received the explicit frame and close:
  `received=[{type:"error",code:"llm_quota_exhausted",message:"Live voice is unavailable: the assistant language model quota is exhausted. …"}]`, `close code=1011, reason="LLM quota exhausted"`.
  The uvicorn log recorded
  `provider_failure session=browser-quota-proof stage=llm_preflight provider=openai category=llm_quota_exhausted http_status=429`.

## 3. Symptom-severity model — documented blocker (no model forced)

No severity/urgency model was trained. Every candidate dataset with genuine urgency
labels is gated behind terms acceptance, account creation, credentialing, or is
restricted. Accepting those terms or creating accounts on the user's behalf was out of
scope, and relabelling an unrelated target (e.g. predicting hospital *admission* as a
stand-in for *urgency*) would be clinically misleading, so it was not done.

| Source | Access | Outcome |
| --- | --- | --- |
| Mendeley `dd359z6trk` | Public | **Downloaded + inspected**: 111 MCI scenario rows; expert text embeds the label (leakage) → unusable |
| HF `olaflaitinen/fedmml-ed-triage` | Access-conditions acceptance + login | Blocked (user action needed) |
| Kaggle `ilkeryildiz/…triage` | Account + ToS | Blocked (no local credentials) |
| Zenodo `14058516` | Files Restricted | Blocked |
| Zenodo `1308993` (Yale) | Open | Target = admission outcome, not urgency → not relabelled |
| PhysioNet (MIMIC-IV-ED/eICU/MIETIC) | CITI + DUA | Not accessed |

Full detail: `data/triage_dataset/SEVERITY_DATASET_INVESTIGATION.md`. Phase 3 (severity
inference service) was therefore **not performed**, by design.

## 4. Complaint-code model (verified)

A genuine scikit-learn model trains from the public CC BY 4.0
[EDEN-RICAPPS Manchester complaint classification](https://zenodo.org/records/19606710)
(DOI `10.5281/zenodo.19606710`). See
`data/triage_dataset/eden_ricapps_zenodo_19606710/SOURCE.md` for provenance.

- 6,989 rows → conflict/duplicate handling → **1,089 unique codes**, 8 chapters
  (class counts **106, 69, 60, 93, 141, 29, 250, 341**).
- Pipeline: char 1–4-gram TF-IDF + class-weighted LogisticRegression; stratified 80/20
  split (seed 20261029), zero code overlap.
- Held-out: accuracy **0.743**, macro F1 **0.741** (majority baseline 0.312 / 0.059).
- **Reproducibility confirmed this pass**: retraining to a temp path reproduced the saved
  artifact exactly — identical split, identical metrics, identical known codes and
  predictions.
- Endpoint `POST /api/triage/complaint-code` is advisory, `dispatch_authorized=false`,
  always `human_review_required`, and `training_coverage` distinguishes
  `seen_in_training` / `unseen_code`. Missing code → 422. Runs offline, independent of
  voice credentials.

Limitations: substantial class imbalance (chapter 6 has 29 examples / test support 6);
one framework's methodological mapping, not a validated clinical model; score uncalibrated.

## 5. First-aid engine (verified + source spot-check)

Deterministic rules for 10 scenarios (cardiac arrest, unconsciousness, severe bleeding,
choking, stroke, breathing difficulty, seizure, burns, poisoning/overdose, fracture).
Unknown/ambiguous input returns conservative scene-safety guidance and forces human
review; a "not breathing / only gasping" signal raises a cardiac-arrest override
independently of any classifier. Never lowered by ML.

**Source spot-check (this pass):** the most safety-critical rule (cardiac arrest /
unresponsiveness) was verified against the live
[Resuscitation Council UK 2025 Adult BLS](https://www.resus.org.uk/professional-library/2025-resuscitation-guidelines/adult-basic-life-support-guidelines)
page — it confirms the encoded actions (call first on speaker, gasping is abnormal
breathing, start compressions when unresponsive + not breathing normally, use AED as soon
as available). American Red Cross pages returned HTTP 403 to the automated check (WAF
blocking non-browser clients), an access limitation, not a broken link; the encoded
bleeding/burns/choking guidance reflects standard published practice. Recorded in
`FIRST_AID_GUIDANCE.md`.

## 6. Location, geocoding, and map (verified via mocked tests)

`dispatch_tools.py` preserves full address names, never geocodes raw transcript text,
and returns candidate lists (without coordinates) for ambiguous matches. Coverage in
`tests/test_dispatch_tools.py` (**41 tests**): geocoding success, preferred-provider
selection, ambiguous/no-match, invalid coordinates, generic/vague-phrase rejection,
city-country approximation, provider failure fallback, JSON serializability. Map defaults
to India center and clears the pin for unresolved/out-of-range coordinates
(`tests/test_frontend_map_contract.py`). Actual external geocoding was not exercised
(no external calls during testing).

## 7. Dispatch lifecycle (verified)

- **Idempotency**: an unchanged auto-dispatch update does not re-run the simulator
  (`test_unchanged_auto_dispatch_update_does_not_rerun_simulator`);
  `test_dispatch_simulation_reuses_state_until_inputs_or_milestone_changes`.
- **Duplicate end-of-turn**: a repeated `end_of_turn` without an active LLM cycle is not
  counted as an assistant completion
  (`test_end_of_turn_without_active_llm_cycle_is_not_assistant_completion`).
- **Stale/duplicate session rejection (added this pass):** an in-process active-session
  registry (`_acquire_live_session` / `_release_live_session`) ensures only one live
  WebSocket owns a `session_id`; a duplicate/stale `start` is rejected with **1008
  "Session already active"** and does not touch the active session. Released on every exit
  path (quota refusal, provider-init failure, setup exception, normal stop).
- Simulation is labelled; no real services contacted.

## 8. Frontend integration (verified in browser)

The dashboard surfaces real backend state with loading/error/offline states. During a
provider outage the local first-aid and complaint-code panels remain fully functional
(verified in a real browser against the running server):

- First-aid panel: "Possible cardiac arrest · CRITICAL", "CALL LOCAL EMERGENCY SERVICES
  NOW", CPR actions, "Human review required" — all rendered locally with no LLM/WS.
- Complaint-code panel: `R07.9` → Cardiorespiratory, `seen_in_training`, advisory/human
  review, with model version and dataset DOI shown.
- Server-sent WS `error` frames are surfaced verbatim (code + message) — proven by the
  live `llm_quota_exhausted` frame above.
- The full voice-start UI path could not be driven to a live session in the automated
  browser because it denies microphone access (`getUserMedia` → "Permission denied"), an
  environment limitation; the WS error path was instead proven with a direct socket (see §2).

## 9. Security / configuration

- `.env` was never read or printed; env vars are documented **by name only**
  (`GRADIUM_API_KEY`, `AI_COUSTICS_API_KEY`, `OPENAI_API_KEY`, `OPENAI_MODEL`,
  `OPENAI_BASE_URL`, `TRIAGE_API_KEY`, `TRIAGE_DECODER_MODEL`, `LIVE_DISPATCH_MODE`).
- Provider error messages are sanitized: only `type`/`code`/HTTP status are retained;
  URLs, tokens, and raw bodies are discarded (asserted in tests).
- `session_id` is validated against a strict pattern; the active-session guard prevents
  cross-session writes.
- Frontend renders server text via `textContent` (no `innerHTML` injection of provider
  data).

## 10. Test results (evidence)

Full suite after all changes:

```
.\.venv312\Scripts\python.exe -m pytest -q
176 passed, 1 warning   (baseline was 161; +15 new)
```

New tests this pass:

- `tests/test_llm_failfast.py` (7): pre-flight ok/quota/non-quota-429/server-error/
  transport-only-type; quota refusal before provider start; watchdog terminates on
  repeated silent pushes; watchdog stays silent when the LLM responds.
- `tests/test_frontend_voice_failure_contract.py` (5): server WS errors surfaced with
  code+message; connected only from backend confirmation; 1011→Unavailable; text-only →
  not played; local panels wired independently.
- `tests/test_session_guard.py` (3): acquire/release registry; duplicate `start` rejected
  1008 without touching the active session; session released after a normal stop.

Updated: `tests/test_live_readiness.py` — existing WS tests patched with `_preflight_ok`
to stay hermetic (never probe the real LLM).

Verification battery (all green):

```
pytest -q                     176 passed, 1 warning
compileall -q src tests scripts   OK
node --check (app.js, review.js, audio-utils.js, audio-utils.test.js)   OK
node --test frontend/audio-utils.test.js   8 passed
pip check                     No broken requirements found
```

Runtime (against a live server on 127.0.0.1:8001, started only after confirming no
listener, stopped afterward):

```
GET  /healthz                              200 {"status":"ok"}
GET  /demo                                 200 (15,202 bytes)
GET  /api/audio-config?triage_mode=llm_only 200 (config keys present)
POST /api/first-aid                        200 cardiac_arrest · critical · call_now=true · human_review=true
POST /api/triage/complaint-code  R07.9     200 chapter=1 Cardiorespiratory · dispatch_authorized=false
POST /api/triage/complaint-code  ZZ99.99   200 unseen_code · dispatch_authorized=false
WS   /ws  (browser)                        1011 llm_quota_exhausted (see §2)
```

## 11. How to reproduce / launch

From the project root (`F:\yeahh\emergency-dispatcher-main\emergency-dispatcher-main`):

```powershell
# Full test suite
.\.venv312\Scripts\python.exe -m pytest -q

# Retrain the complaint-code model (reproduces the saved artifact)
.\.venv312\Scripts\python.exe scripts\prepare_eden_ricapps.py --force
.\.venv312\Scripts\python.exe scripts\train_complaint_chapter_model.py --force

# Headless live-voice probe (sanitized; optional --wav for synthetic speech)
.\.venv312\Scripts\python.exe scripts\probe_live_voice.py --seconds 28 --wav "$env:TEMP\opencode\caller_speech.wav"

# Run the server (check port 8001 is free first; do not start a duplicate)
Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue
.\.venv312\Scripts\python.exe -m uvicorn emergency_dispatcher.server:app --host 127.0.0.1 --port 8001
# then open http://127.0.0.1:8001/demo
```

## 12. Limitations / blockers

- **Assistant voice (LLM→TTS→playback) is blocked** by the OpenAI account having zero
  credits (`insufficient_quota` / `credit_balance_exhausted`). External blocker: add
  credits or point `OPENAI_BASE_URL`/key at a funded provider. The failure is now surfaced
  explicitly (pre-flight + watchdog + frontend), never silently.
- **No severity/urgency model** — all suitable datasets gated by terms/credentialing
  (§3). First-aid severity remains deterministic and is never ML-driven.
- **Automated browser cannot grant microphone access**, so the full voice-start UI flow
  was not driven to a live session there; the WS error path was proven with a direct
  socket instead. Real-device voice testing needs a user microphone.
- **External geocoding not exercised** during testing; location behavior is covered by
  mocked tests only.
- Complaint-code model: class imbalance, one framework's mapping, uncalibrated score;
  advisory only, human review always required.
- Repo has **no `.git`** — no diff/status available. `.env` untouched.

## 13. Evidence index

- Complaint-code reproducibility + endpoint: `tests/test_triage_ml.py`; artifacts under
  `data/triage_models/`.
- Severity blocker: `data/triage_dataset/SEVERITY_DATASET_INVESTIGATION.md`.
- Voice root-cause + repair: `server.py` (`_llm_preflight_check`, watchdog, output loop),
  `tests/test_llm_failfast.py`, `scripts/probe_live_voice.py`.
- Frontend failure surfacing: `frontend/app.js`,
  `tests/test_frontend_voice_failure_contract.py`, `frontend/audio-utils.test.js` (8).
- Session guard: `server.py` (`_acquire_live_session`), `tests/test_session_guard.py`.
- First-aid + spot-check: `first_aid.py`, `FIRST_AID_GUIDANCE.md`, `tests/test_first_aid.py`.
- Location/dispatch: `dispatch_tools.py`, `tests/test_dispatch_tools.py`,
  `tests/test_frontend_map_contract.py`, `tests/test_live_readiness.py`,
  `tests/test_live_audio_contract.py`.
- Working checklist: `TASK_CHECKLIST.md`.
