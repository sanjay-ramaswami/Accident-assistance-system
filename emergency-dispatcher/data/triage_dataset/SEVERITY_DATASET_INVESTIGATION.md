# Symptom-Severity Model — Dataset Investigation & Blocker

Date: 2026-10-29
Scope: Phase 2/3 of the Emergency Dispatcher prototype — build a **symptom-severity /
urgency** model only if suitable labelled urgency data can be legitimately obtained.

## Decision

**No severity model was trained.** Every candidate dataset that actually contains
urgency/acuity ground-truth labels is gated behind terms-of-service acceptance,
account creation, credentialing, or is restricted. Accepting those terms, creating
accounts, or completing training on the user's behalf was out of scope, so the
pipeline stopped here rather than relabelling an unrelated target (for example
predicting *hospital admission* as a stand-in for *urgency*, which would be a
different and clinically misleading task).

Critical first-aid guidance in this prototype is **never** derived from or downgraded
by an ML severity score; it comes from a deterministic rules engine
(`src/emergency_dispatcher/first_aid.py`). Dispatch is simulated and clearly labelled.

## Candidates examined

| Source | Licence / access | Labels | Outcome |
| --- | --- | --- | --- |
| Mendeley `dd359z6trk` (Scenario Dataset, Nwandikom & Namin) | Public download (CC-style) | Expert vs ChatGPT triage priority per MCI scenario | **Downloaded and inspected.** 111 rows; the "expert" free-text field embeds the answer (e.g. text literally states the priority), so using it as a feature is pure label leakage. No independent input features. Unusable. |
| Hugging Face `olaflaitinen/fedmml-ed-triage` | Requires agreeing to share contact info / access conditions + login | Synthetic ESI 1–5 acuity | **Blocked.** Access-conditions acceptance is a user decision. Not downloaded. |
| Kaggle `ilkeryildiz/emergency-service-triage-application` | Kaggle account + ToS acceptance | Triage labels | **Blocked.** No local Kaggle credentials (`~/.kaggle/kaggle.json` absent); account creation/ToS on user's behalf not permitted. Not accessed. |
| Zenodo `14058516` (Synthetic Emergency Healthcare Services) | Files **Restricted** (record public, files need login/permission) | EMS dispatch fields | **Blocked.** Restricted files not accessed. |
| Zenodo `1308993` (Yale ED dataset) | Open | Hospital **admission** outcome | **Not used.** Target is admission (a post-outcome), not triage urgency. Relabelling it as severity would be a different, misleading task. |
| PhysioNet (MIMIC-IV-ED, eICU, MIETIC) | Requires CITI training + signed DUA | Chief complaint + acuity | **Not accessed.** Credentialing + data-use agreement are user obligations. |

## Exact blocker

Suitable, independently-featured, urgency-labelled emergency data is not obtainable
without the account owner performing one of: accepting dataset terms, completing CITI
training + DUA, or creating a platform account. Until one of those is done by the user,
a defensible severity model cannot be trained, and no inference service was built.

## Reproducing the Mendeley inspection (the one usable download)

```
# List files (public API)
Invoke-RestMethod 'https://data.mendeley.com/public-api/datasets/dd359z6trk/files?folder_id=root&version=2'
# Download the single xlsx and inspect columns; label is embedded in EXPERT_SOLUTION text.
```

The downloaded artefact is stored (unlabelled as usable) under
`data/triage_dataset/mendeley_dd359z6trk_scenarios/` for provenance only. It is **not**
used for training anywhere in this codebase.
