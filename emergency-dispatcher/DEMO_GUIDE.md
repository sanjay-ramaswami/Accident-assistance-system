# Windows Demo Guide

## Prerequisites

- Windows 10 or 11 and Python 3.12 (Python 3.13 is also supported).
- Internet access for package installation. The text triage exercise works without provider credentials.
- No database, migration, frontend package install, or seed command is required.

This workspace has Python 3.12.11 installed through the per-user `uv` runtime. In PowerShell, from the repository root, create and install its isolated environment with:

```powershell
$python = Join-Path $env:APPDATA 'uv\python\cpython-3.12.11-windows-x86_64-none\python.exe'
& $python -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[dev]"
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

On another computer with Python 3.12 registered in the Windows launcher, use `py -3.12 -m venv .venv` and replace `.venv312` with `.venv` in the commands below. Never overwrite an existing `.env`; add only credentials you are authorized to use.

## Configuration

The deterministic local scenarios need no environment variables. The checked-in `.env.example` leaves API keys blank. The demo dashboard selects **Gemini Live** by default; configure `GEMINI_API_KEY` (and optionally `GEMINI_LIVE_MODEL`) in `.env` for live Gemini voice. **Gradium + OpenAI** remains selectable and requires its existing credentials:

- Alternate Gradium + OpenAI voice: `GRADIUM_API_KEY`, `AI_COUSTICS_API_KEY`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.
- Gemini Live voice: `GEMINI_API_KEY`; optional `GEMINI_LIVE_MODEL`.
- Pioneer SLM text triage: `TRIAGE_API_KEY` and `TRIAGE_DECODER_MODEL`.
- Optional transcript cross-check: `DEEPGRAM_API_KEY` or `SONIOX_API_KEY`.
- Optional map/geocoding features: `GOOGLE_MAPS_API_KEY`.

The Live Emergency Assessment includes a natural-language **synthetic-data
prototype**, trained only on 14 authored demo scenarios across 12 labels. It
has no held-out evaluation or clinical validation. Greetings and connection
checks are marked insufficient input. When model inference is unavailable,
the UI separately labels the rules result as `Rule-based demo fallback — not
an ML prediction`. The scikit-learn complaint-code model remains a separate
code-to-chapter classifier and is only run for explicitly supplied codes. See
[`docs/incident-classifier-data.md`](docs/incident-classifier-data.md).

The console checks live voice and triage settings before requesting microphone access and lists missing variable names. The backend also rejects unconfigured WebSocket sessions. No development login is needed. Live integrations are not activated by the text triage exercise.

## Launch

From the repository root:

```powershell
.\.venv312\Scripts\python.exe -m uvicorn emergency_dispatcher.server:app --host 127.0.0.1 --port 8001
```

Open these local pages:

- Dispatch console and text triage: [http://127.0.0.1:8001/demo](http://127.0.0.1:8001/demo)
- Saved session review: [http://127.0.0.1:8001/review](http://127.0.0.1:8001/review)
- Readiness: [http://127.0.0.1:8001/healthz](http://127.0.0.1:8001/healthz)

Stop the server with `Ctrl+C`. Session records are written under `runs/sessions/`.

## Presentation Scenario

1. Open `/demo`. The top of the page contains **Try Triage With Text**; no microphone permission is needed.
2. Paste: `There was a crash near Alexanderplatz. One person is trapped and bleeding. I cannot tell if they are breathing.`
3. Leave **Configured model** selected and click **Analyze text**.
4. With no Pioneer credentials, the result identifies `heuristic` as the engine and shows an advisory structured triage result and any deterministic safety override. This is the local rule based fallback, not a trained ML model.
5. Add valid `TRIAGE_API_KEY` and `TRIAGE_DECODER_MODEL` to `.env`, restart the server, and repeat. **Configured model** then sends the input to the configured Pioneer SLM; the result identifies `slm` and `pioneer`. This network inference was not available for validation without credentials.
6. Point out the explicit advisory warning. This endpoint only returns triage guidance; it does not place calls, assign responders, persist an incident, or alter live sessions.

## Deterministic Emergency Assessment Demo

1. Start the server using the Launch command above and open `http://127.0.0.1:8001/demo`.
2. In **Run a Simulated Scenario**, select a bundled scenario and click **Run Simulated Scenario**. This uses synthetic transcript text, local rules, and the existing session/ledger flow; it does not request microphone access or call a paid model/geocoder.
3. Review **Emergency Assessment** for the incident category, prototype or labeled rule fallback, advisory urgency, evidence, unknown fields, follow-up prompts, and safety guidance.
4. Observe the **SIMULATION MODE** unit assignment, configured Marine Drive demo pin, dashed straight-line route, illustrative ETA, moving ambulance marker, and persisted stage timeline through completion. The location is explicitly not caller-verified; the route is not road navigation.
5. Choose **Reset Demo** to release the simulated unit and clear the current dashboard view. Saved session records are retained.
6. To test live voice, configure the authorized provider credentials listed under Configuration, restart the server, and click **Start Call**. Browser audio, provider stability, and live-provider behavior require a separate manual browser session; the deterministic demo does not verify them.
7. All service planning and ticket state shown in this project are simulated. No real emergency service is contacted and no real dispatch occurs.

The scenario picker includes road-side smoke, road accident with an unconscious casualty, severe bleeding, chest pain/breathing difficulty, multiple casualties, ambiguous location, conflicting breathing reports, complaint-code-unavailable input, and the original worker-collapse and gas-leak examples. Inputs and generated service states are synthetic/simulated.

## Live Gemini voice and assessment

1. Add an authorized `GEMINI_API_KEY` to `.env`, restart the server, open `/demo`, leave **Gemini Live** selected, and click **Start Call**. Allow microphone access and speak one short caller utterance, then pause long enough for the transcript to finalize.
2. The top **Emergency Assessment** panel should change from **Processing transcript** to **Assessment updated**. It shows the current incident/session, reported facts and source utterances, unknown/conflicting facts, follow-up focus, location status, curated safety guidance, and simulation-only responder planning.
3. The same panel always shows a Live ML status. Without an explicitly spoken supported complaint code it says **Not applicable — complaint code unavailable**. When a code is explicitly stated, the existing local model result appears with training coverage and any available uncalibrated score. Symptoms alone are never converted into a complaint code.
4. For a road accident, test a synthetic turn such as: “There has been a road accident. Smoke is coming from a vehicle. I am standing near the road.” Confirm the guidance says to check scene safety, stay away from traffic/smoke/fire/fuel hazards, and ask someone to make a real local emergency call; the app itself cannot place it.
5. Stop the call before switching provider or starting a synthetic scenario. Do not use a real emergency number for testing. The local session record contains the original caller transcript; do not share session files publicly.

This procedure is a manual browser check, not evidence of reliable/live voice performance. The automated checks verify request wiring, presentation contracts, and local scenario processing; actual microphone, Gemini response quality, audible guidance, and playback still need to be checked in the browser.

The Session Events panel records monotonic `voice_latency` markers for capture,
outbound audio, finalized caller turn, first provider response, returned audio,
frontend enqueue, and playback start. A caller-turn's values are relative to
the finalized transcript. Compare those fields from repeated, similar live
sessions; do not use the offline scenario as a Gemini latency measurement. If
the provider does not return a response, the UI reports that state and does not
insert a synthetic assistant reply after the existing 15-second wait.

## Run Checks

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe -m compileall -q src tests
node --test frontend/*.test.js
node --check frontend/app.js
```

The project has no separate frontend build; FastAPI serves the static HTML, CSS, and JavaScript directly.

## Common Issues

- **Python 3.14 or 3.10 selected:** this project currently supports 3.12 and 3.13. Use the Python 3.12 path above or install a supported version.
- **Package install fails on `aic-sdk` or `gradbot`:** confirm Python 3.12/3.13 and 64-bit Windows. These are native audio dependencies with platform-specific wheels.
- **`Address already in use`:** choose another port (for example `--port 8002`) and open the matching URL.
- **Text triage returns 503 for SLM:** configure `TRIAGE_API_KEY` and `TRIAGE_DECODER_MODEL`, or select **Local heuristic**.
- **Start Call reports missing configuration:** Gemini Live requires `GEMINI_API_KEY`; Gradium + OpenAI requires its four listed credentials. Use the deterministic scenario panel for an offline demo.
- **Assessment update reports a network issue:** check the visible report status. Large live reports are sent without Fetch `keepalive`; transient network/timeouts retry once, and automatic sync pauses after three failed requests until you retry.
- **Map marker appears to be retained:** a partial update cannot erase a valid pin. The status identifies when the old pin is retained while a new candidate is unresolved. Starting a new session clears the prior session's pin.
- **Location or response planning looks approximate:** live caller locations retain provider confidence. The deterministic demo pin, fleet, route, and ETA are configured simulations; they are not caller GPS, road navigation, or real service availability.

## Implemented And Limited

- Implemented: FastAPI server, static dispatch console, session review, transcript/session APIs, structured heuristic triage, Pioneer SLM adapter, and safe local text triage endpoint.
- Simulated: demo fleet assignment, straight-line movement, illustrative ETA, lifecycle progression, and incident ticket flow. No emergency services are contacted.
- Not validated here: live voice providers, microphone/WebSocket audio, Pioneer credentials and network inference, live geocoding, PBX or telephone audio, real dispatch, and emergency-service outcomes.
- Safety: triage is advisory and human review remains available. The local text endpoint does not dispatch or claim clinical success.
