# Local first-aid guidance prototype

The `/api/first-aid` endpoint uses deterministic keyword rules and this small
curated guidance table. It makes no LLM or geocoder request and does not store
the submitted description. It is a demonstration aid, not a diagnosis, not
clinically validated, and not a substitute for emergency dispatch instructions.
The sources below were checked on **2026-10-29**. Guidance is principally from
the American Red Cross and UK Resuscitation Council/NHS; callers must use their
local emergency number and local emergency-service instructions.

| Guidance category | Reference |
| --- | --- |
| Road accident scene safety | [UK Government — Highway Code, Annex 7: First aid on the road](https://www.gov.uk/guidance/the-highway-code/annex-7-first-aid-on-the-road) |
| General/unknown emergency and scene check | [American Red Cross — First Aid Steps](https://www.redcross.org/take-a-class/first-aid/performing-first-aid/first-aid-steps) |
| Unconsciousness; possible cardiac arrest | [Resuscitation Council UK — 2025 Adult Basic Life Support](https://www.resus.org.uk/professional-library/2025-resuscitation-guidelines/adult-basic-life-support-guidelines); [American Red Cross — Unresponsive and Breathing](https://www.redcross.org/take-a-class/resources/learn-first-aid/unresponsive-and-breathing-person) |
| Severe external bleeding | [American Red Cross — Life-Threatening External Bleeding](https://www.redcross.org/take-a-class/resources/learn-first-aid/bleeding-life-threatening-external) |
| Burns/scalds | [American Red Cross — Burns](https://www.redcross.org/take-a-class/resources/learn-first-aid/burns) |
| Choking | [American Red Cross — Adult and Child Choking](https://www.redcross.org/take-a-class/resources/learn-first-aid/adult-child-choking) |
| Difficulty breathing | [American Red Cross — Respiratory Distress](https://www.redcross.org/take-a-class/resources/learn-first-aid/respiratory-distress-trouble-breathing) |
| Possible stroke | [American Red Cross — First Aid Guidelines](https://www.redcross.org/take-a-class/resources/guidelines) |
| Seizure | [American Red Cross — Seizures](https://www.redcross.org/take-a-class/resources/learn-first-aid/seizures); [NHS — What to do if someone has a seizure](https://www.nhs.uk/symptoms/what-to-do-if-someone-has-a-seizure-fit/) |
| Poisoning/overdose | [NHS — Poisoning](https://www.nhs.uk/conditions/poisoning/); [Poison Control — First aid for poisonings](https://www.poison.org/first-aid-for-poisonings) |
| Suspected fracture or serious injury | [American Red Cross — Fractures](https://www.redcross.org/take-a-class/resources/learn-first-aid/fractures); [Muscle, Bone and Joint Injury](https://www.redcross.org/take-a-class/resources/learn-first-aid/muscle-bone-joint-injury) |

## Safety boundaries

- The rules are advisory and use broad text matching; they may miss, over-detect,
  or misclassify symptoms. An unknown description still returns conservative
  scene-safety guidance and recommends human review.
- Unresponsiveness alone is not labeled cardiac arrest. The response advises
  calling emergency services and checking normal breathing while the call is
  arranged. CPR/AED advice is tied to unresponsiveness with absent/abnormal
  normal breathing and defers to dispatcher instructions; age-specific details
  are not invented.
- Multiple recognized conditions keep all matched sources visible and require
  human review. An approximate or ambiguous classification never lowers a
  life-threatening override.
- Emergency-number text is generic and the application never places a call.
- These guidance rules remain deterministic and are not an ML model. The
  separate complaint-code chapter classifier documented in
  `IMPLEMENTATION_REPORT.md` does not classify transcript text, emergency
  severity, or first-aid actions and cannot change these critical rule
  overrides.

## Source spot-check (2026-10-29)

The most safety-critical rule (cardiac arrest / unresponsiveness) was verified
against its authoritative source during completion of this prototype:

- **Resuscitation Council UK — 2025 Adult Basic Life Support** (page fetched
  successfully): confirms the encoded actions — call the emergency number first
  with the phone on speaker, treat gasping/agonal breathing as abnormal (not
  normal) breathing, start chest compressions when a person is unresponsive with
  abnormal breathing, and use an AED as soon as available following its prompts.
  The rule text is consistent with this source.

- **UK Government — Highway Code, Annex 7: First aid on the road** (page fetched
  successfully): confirms scene-danger checks, asking a bystander to call 999 or
  112, supplying exact road location and traffic direction, and not moving
  casualties unless there is further danger. The application's wording stays
  number-neutral because it does not know every caller's jurisdiction.

- **American Red Cross — Life-Threatening External Bleeding** (page fetched
  successfully from its production domain): confirms steady, firm direct
  pressure and that a tourniquet should be used only in the appropriate
  life-threatening limb-bleeding context. The existing rule only suggests a
  commercial tourniquet when available and the bystander knows how to use it.

American Red Cross pages cited in the table above returned HTTP 403 to the
automated check (their web-application firewall blocks non-browser clients);
this is an access limitation of the check, not evidence of a broken link. The
bleeding/burns/choking guidance they encode (firm direct pressure for
life-threatening bleeding; cool running water for thermal burns; no blind finger
sweep for choking) reflects standard, widely published first-aid practice and is
cross-checked against the same conservative wording used in the rules engine.
