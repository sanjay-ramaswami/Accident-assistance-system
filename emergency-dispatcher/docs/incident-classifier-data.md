# Natural-language incident classifier data decision

**Status: synthetic-data prototype only.** There is no suitable real emergency-call
transcript training corpus or clinical validation in this repository. The separate
free-text model is trained only on the 14 authored deterministic scenarios in
`data/emergency_scenarios.json`, covering 12 labels. It has no held-out evaluation
and reports no accuracy or clinical metric. Its output is a demo prototype result,
not a diagnosis, severity estimate, or dispatch recommendation.

Reproduce the artifact with:

```powershell
.\.venv312\Scripts\python.exe scripts\train_incident_text_prototype.py
```

The model artifact is `data/triage_models/incident_text_scenario_prototype_v1.joblib`.
The training script records the source file hash, model version, row/label counts,
and explicitly sets metrics to null and clinical validation to false. Inference
checks the artifact metadata and pipeline before use, runs off the asyncio event
loop, returns an opaque input ID rather than caller text, and marks any scores as
uncalibrated. The prototype is intentionally separate from the EDEN-RICAPPS
complaint-code classifier, which accepts an explicit code and cannot classify
raw speech.

The EDEN-RICAPPS model remains a code-to-chapter model. Its checked-in evaluation
reports 218 held-out complaint codes, 74.31% accuracy, and 0.7413 macro F1. Those
metrics do not apply to transcripts or the synthetic text prototype.

The local scenario demo exercises this prototype without a microphone, paid
model, or external geocoding. Caller transcript text remains in the audit
transcript; classifier outputs do not echo it. Dispatch remains simulation-only.
