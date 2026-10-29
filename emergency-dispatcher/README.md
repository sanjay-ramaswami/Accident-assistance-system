# Emergency Dispatcher

Realtime emergency-dispatch demo for noisy environments.

Originally built during a hackathon, this repo is the working system that the later SLM + LLM grounding playbook was extracted from.

The architectural pattern this project converged on — fast LLM voice lane plus slow SLM grounding lane around a shared ledger — is generalized in a separate playbook: [Reaper2403/slm-llm-grounding-playbook](https://github.com/Reaper2403/slm-llm-grounding-playbook).


https://github.com/user-attachments/assets/f2ee3c60-c74e-4e77-944c-5098a3172fef



The project combines:
- `ai-coustics` for live input enhancement
- `Gradbot + Gradium` for the voice loop
- `Deepgram` as a continuous backend STT cross-check
- `FastAPI` for the demo server and session APIs
- a triage layer that can run in `heuristic`, `llm`, or `slm` mode
- local and Pioneer-oriented dataset tooling for the SLM path

## What Exists Today

### Live demo
- browser-based dispatch console at `/demo`
- live caller / dispatcher transcript streams
- operator-facing handoff desk with severity display and dummy takeover button
- map initially centered on India; it pins an incident only when usable coordinates are available
- tool timeline and session report status
- persisted sessions under `runs/sessions/`

Live voice requires the provider credentials listed below. Without them, the console reports the missing settings and does not request microphone access or claim a connected call. The browser `/healthz` check confirms process liveness only; it does not test provider credentials or services.

### Runtime pipeline
- the dashboard selects `Gemini Live` by default; the existing `Gradium + OpenAI` path remains an optional provider
- mic audio is resampled and sent as provider-compatible PCM
- `ai-coustics` enhancement remains available on the Gradium path; Gemini uses its own audio session
- finalized utterances are synced back to the backend
- finalized utterances update the evidence-linked Emergency Assessment; the Live ML panel reports when a complaint code is unavailable and never invents a code
- the distinct natural-language classifier is a **synthetic-data prototype** trained on 14 authored deterministic demo scenarios across 12 labels; it is not clinically validated and has no held-out accuracy metric
- natural-language inference only runs when finalized speech contains an incident-related cue; greetings and connection checks are marked insufficient input
- if the prototype is missing or fails, the panel shows the separate rule-based assessment as `Rule-based demo fallback — not an ML prediction`
- deterministic first-aid guidance is rendered in the assessment and passed as bounded provider context after the transcript is synchronized
- `Deepgram` continuously compares finalized caller turns and can become the canonical backend transcript provider for the session
- triage state is merged turn by turn and fed into the LLM prompt

### Triage / SLM groundwork
- stable triage contract in `src/emergency_dispatcher/triage.py`
- modes:
  - `heuristic`
  - `llm`
  - `slm`
- SLM data-generation scripts for:
  - dispatch guidance
  - fact-ledger style evidence tracking
- Pioneer-ready decoder dataset export scripts

## Current Runtime Direction

The project now supports two practical live modes:
- `LLM only`
- `SLM + LLM`

What we learned from repeated live calls:
- `LLM only` is currently the smoothest voice path
- `LLM only` can sound natural and recover from noisy turns well
- `LLM only` plateaus on grounding, consistency, and deterministic fact handling
- the next step is a narrower `SLM + LLM` split where the SLM owns hard facts and the LLM owns delivery and judgment

That split is now the intended architecture:
- `SLM / deterministic ledger`: hard facts, location certainty, issue cues, victim count, breathing / bleeding / trapped status
- `LLM`: calm questioning, ambiguity handling, empathy, and spoken instructions
- `server tools`: location resolution, nearby response bases, dispatch simulation, and handoff/ticket generation

## Repo Layout

```text
src/emergency_dispatcher/
  server.py             FastAPI app + websocket voice session
  gradbot_adapter.py    runtime prompt + tool definitions
  dispatch_tools.py     address, priority, and ticket tools
  enhanced_audio.py     ai-coustics live/audio artifact handling
  stt_rescue.py         Deepgram/Soniox rescue and backend transcript selection
  triage.py             triage engine interface + state merge + runtime prompt packet
  slm_guidance.py       guidance-style SLM schema and helpers
  slm_fact_ledger.py    fact-ledger schema and helpers
  berlin_location_lexicon.py  Berlin lexicon helpers for data generation

frontend/
  index.html            live dispatch console
  app.js                browser session logic + operator desk
  review.html           session review page
  review.js             review UI
  styles.css            shared frontend styling

scripts/
  generate_dispatch_training_data.py
  generate_slm_guidance_training_data.py
  generate_slm_fact_ledger_training_data.py
  prepare_* / finalize_* / export_* helpers for dataset workflows

data/
  emergency_scenarios.json
  schema/               checked-in schema snapshots

tests/
  pytest suite for tools, rescue, triage, schemas, and routes
```

## Required Setup

Use Python `3.12` or `3.13` on Windows. Python `3.14` is not supported yet because the native audio packages do not publish compatible wheels for this project.

PowerShell setup from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

The browser text triage exercise and health endpoint work without credentials. Live voice calls need the provider credentials below. Add only the keys you have to `.env`; keep it private.

Required for live voice calls:

```env
GEMINI_API_KEY=...
GRADIUM_API_KEY=...
AI_COUSTICS_API_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

`DEEPGRAM_API_KEY` is optional and enables transcript cross-checking. The live SLM mode additionally needs `TRIAGE_API_KEY` and `TRIAGE_DECODER_MODEL`.

Optional / advanced:

```env
SONIOX_API_KEY=...
TRIAGE_API_KEY=...
TRIAGE_DECODER_MODEL=...
TRIAGE_GLINER_MODEL=...
GOOGLE_MAPS_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
```

Important defaults already supported:
- Gemini Live is the demo default; `GEMINI_API_KEY` is required for real provider sessions
- `TRIAGE_ENGINE=heuristic|llm|slm`
- `LIVE_DISPATCH_MODE=slm|llm_only`
- `STT_RESCUE_CONTINUOUS_COMPARE=true`

## Run The App

```powershell
.\.venv\Scripts\python.exe -m uvicorn emergency_dispatcher.server:app --host 127.0.0.1 --port 8001
```

Open [http://127.0.0.1:8001/demo](http://127.0.0.1:8001/demo). `/healthz` checks process liveness and `/review` shows saved sessions. The server stores session artifacts under `runs/sessions/`; no database setup or seed is required.

For the `.venv312` environment on Windows, run from the repository root:

```powershell
$env:PYTHONPATH = "src"
.\.venv312\Scripts\python.exe -m uvicorn emergency_dispatcher.server:app --host 127.0.0.1 --port 8001
```

The live-call map opens at latitude `22.5`, longitude `79.0`, zoom `5`. Live incident markers require validated coordinates. The deterministic demo runner uses a fixed Marine Drive, Kochi location, labeled `DEMO LOCATION — NOT VERIFIED FROM CALLER`; this is configuration, not caller location or live GPS. It assigns from a small deterministic simulated fleet and advances through an illustrative arrival/completion timeline. The dashed route is straight-line interpolation, not road navigation; its ETA is illustrative. No ambulance or emergency service is contacted.

## Local ML models

The complaint-code model is the checked-in EDEN-RICAPPS code-to-chapter artifact at `data/triage_models/eden_ricapps_code_chapter_v1.joblib`. It accepts an explicitly stated short code (for example `R07.9`); it does not classify arbitrary speech, diagnose, estimate severity, or authorize dispatch. The stored evaluation is for held-out codes only.

The separate natural-language model is a synthetic-data prototype, not clinical evidence. Rebuild it from the bundled authored scenarios with:

```powershell
.\.venv312\Scripts\python.exe scripts\train_incident_text_prototype.py
```

The training script writes `data/triage_models/incident_text_scenario_prototype_v1.joblib`, records the source hash, and deliberately reports no accuracy metrics. The backend validates this artifact at startup and logs only safe model state and metadata.

## Deterministic presentation demo

1. Start the server with the command above and open `/demo`.
2. Under **Test An Emergency Scenario**, select a synthetic case such as **Worker collapse on industrial floor**.
3. Choose **Run Simulated Scenario**. The backend runs the actual local session, assessment, classifier, and simulation services without microphone access or paid model calls.
4. Observe the natural-language prototype result, configured demo location, simulated fleet unit, dashed straight-line route, progress, timeline, and completion state. The demo location is explicitly not caller-verified.
5. Choose **Reset Demo** before another run. Reset releases the demo unit and preserves the session record for review.

These scenarios are authored software fixtures, not clinical training data. The simulation is independent of the live Gemini/Gradium voice path.

Session reports are persisted even when SLM triage configuration is unavailable; the report marks that inference as unavailable and does not fabricate triage output. Service assignment and dispatch shown in the demo are simulated and do not contact emergency services. The human takeover control is also a demo handoff control, not a real operator connection.

Avoid `--reload` while generating datasets inside this repo. The file watcher can restart the server mid-call.

For a text-only triage exercise, enter a sample caller description on the demo page and select `Configured model`. The Pioneer SLM runs when `TRIAGE_API_KEY` and `TRIAGE_DECODER_MODEL` are configured; otherwise the API returns a result from the existing local heuristic and labels it as such. Both outputs are advisory. They do not trigger a real dispatch.

## Smoke Checks

Core provider check:

```powershell
.\.venv\Scripts\python.exe -m emergency_dispatcher.smoke
```

Test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test frontend/*.test.js
node --check frontend/app.js
node --check frontend/live-ml-utils.js
python -m compileall -q src scripts tests
```

## Current Tool Surface

The current live stack is intentionally split between:

### LLM-triggered tools
- `resolve_location_note`
- `checklist_by_incident`
- `validate_address`
- `lookup_address`
- `nearby_context`
- `build_handoff_brief`
- `create_incident_ticket`

### Server-driven deterministic tools
- `plan_response_services`
- `lookup_response_bases`
- `simulate_dispatch_services`

The important design choice is that service planning, dispatch theatrics, and nearby-base resolution should not depend on the LLM remembering to do them.

## LLM-Only Plateau

The current `LLM only` path is good enough to demo:
- the configured Gemini Live voice transport when credentials/browser audio are available
- noisy-call turn recovery hooks
- short dispatcher instructions and rule-grounded guidance
- confidence-labelled map pinning when validated geocoding data is available

But it still plateaus on:
- exact location confirmation and correction handling
- consistent caller-vs-third-person medical phrasing
- duplicate / awkward question forms under pressure
- deterministic fact carry-forward across fragmented STT turns
- knowing when to trust, reject, or defer a location candidate

That is why the next phase is not “make the LLM smarter.”
It is:
- keep the voice loop light
- keep deterministic tools lean
- move hard-fact grounding into the fact-ledger SLM path

## Data Generation Workflows

### Local guidance dataset generation

Requires a local `Ollama` server.

Start Ollama:

```bash
/opt/homebrew/bin/ollama serve
```

Generate a small validation batch first:

```bash
cd path/to/repo
source .venv/bin/activate
python -u scripts/generate_slm_guidance_training_data.py --count 50
```

### Fact-ledger dataset generation

```bash
cd path/to/repo
source .venv/bin/activate
python -u scripts/generate_slm_fact_ledger_training_data.py --count 50
```

### Pioneer export helpers

Use the `prepare_*`, `finalize_*`, and `export_*` scripts in [scripts](./scripts) once the local sample quality looks good.

## Current Documentation

The repo keeps only the active markdown docs:
- [README.md](./README.md): setup, architecture, runtime modes, and workflows
- [DEMO_GUIDE.md](./DEMO_GUIDE.md): Windows launch commands and a text-only presentation scenario
- [prompts/dispatcher_system_prompt.md](./prompts/dispatcher_system_prompt.md): active dispatcher runtime instructions

## Main Known Limits

- `LLM only` is smoother than the current `SLM + LLM` path, but it still tops out on factual grounding
- location flow is better than before, but still not yet a fully strict state machine
- live service planning does not invent an ETA; the isolated demo simulator shows an explicitly illustrative ETA and straight-line route, never a real dispatch
- live session reports are posted without Fetch `keepalive`; large report failures are classified, retried once, and automatic sync pauses after three consecutive failures
- a valid map pin is retained through partial assessment updates and reset only on a new session or explicit clear
- `Soniox` is wired but not yet as trustworthy as `Deepgram`
- the human-takeover button is currently presentation-only
- the next serious quality step is the fact-ledger `SLM + LLM` path, not more prompt growth in `LLM only`

Natural-language classifier data review and the existing complaint-code model's
held-out evaluation are documented in
[`docs/incident-classifier-data.md`](docs/incident-classifier-data.md). The
complaint-code artifact is not applied to free text. Neither model can authorize
dispatch, and the system is not clinically validated.
