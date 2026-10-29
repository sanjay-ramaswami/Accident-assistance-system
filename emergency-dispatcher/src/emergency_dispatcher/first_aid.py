"""Small, deterministic first-aid guidance for a local demo.

This is a retrieval/rule prototype, not medical diagnosis or clinical decision
support. Keep the curated language conservative and each category sourced.
"""

from __future__ import annotations

import re
from typing import Any


ENGINE_ID = "first-aid-rules"
ENGINE_VERSION = "1.0.0"
VERIFIED_ON = "2026-10-29"

_URLS = {
    "general": (
        "American Red Cross",
        "First Aid Steps",
        "https://www.redcross.org/take-a-class/first-aid/performing-first-aid/first-aid-steps",
    ),
    "road_safety": (
        "UK Government, Highway Code (compiled with St John Ambulance, British Heart Foundation, and British Red Cross)",
        "Annex 7: First aid on the road",
        "https://www.gov.uk/guidance/the-highway-code/annex-7-first-aid-on-the-road",
    ),
    "unresponsive": (
        "Resuscitation Council UK",
        "Adult basic life support Guidelines (2025)",
        "https://www.resus.org.uk/professional-library/2025-resuscitation-guidelines/adult-basic-life-support-guidelines",
    ),
    "bleeding": (
        "American Red Cross",
        "Bleeding (Life-Threatening External)",
        "https://production.redcross.org/take-a-class/learn-fa-bleeding-life-threatening-external.html",
    ),
    "burns": (
        "American Red Cross",
        "Burns: Types, Symptoms, and How To Help",
        "https://www.redcross.org/take-a-class/resources/learn-first-aid/burns",
    ),
    "choking": (
        "American Red Cross",
        "Adult & Child Choking: Symptoms and First Aid",
        "https://www.redcross.org/take-a-class/resources/learn-first-aid/adult-child-choking",
    ),
    "breathing": (
        "American Red Cross",
        "Respiratory Distress (Trouble Breathing)",
        "https://www.redcross.org/take-a-class/resources/learn-first-aid/respiratory-distress-trouble-breathing",
    ),
    "stroke": (
        "American Red Cross",
        "Guidelines: Recognition of Stroke in Adults",
        "https://www.redcross.org/take-a-class/resources/guidelines",
    ),
    "seizure": (
        "American Red Cross",
        "Seizures: Causes, Symptoms, and Types",
        "https://www.redcross.org/take-a-class/resources/learn-first-aid/seizures",
    ),
    "fracture": (
        "American Red Cross",
        "Fractures: Types, Symptoms, and Treatment",
        "https://www.redcross.org/take-a-class/resources/learn-first-aid/fractures",
    ),
    "poisoning": (
        "NHS",
        "Poisoning",
        "https://www.nhs.uk/conditions/poisoning/",
    ),
    "overdose": (
        "Poison Control",
        "First aid for poisonings",
        "https://www.poison.org/first-aid-for-poisonings",
    ),
}


def _source(key: str) -> dict[str, str]:
    organization, title, url = _URLS[key]
    return {
        "organization": organization,
        "title": title,
        "url": url,
        "verified_on": VERIFIED_ON,
    }


_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "road_accident_scene_safety",
        "label": "Road accident scene safety",
        "patterns": (r"\broad (?:traffic )?(?:accident|crash|collision)\b", r"\b(?:car|vehicle|truck|van|motorcycle) (?:crash|collision|accident)\b", r"\b(?:crash|collision|accident)\b.{0,50}\b(?:car|vehicle|truck|van|motorcycle)\b"),
        "severity": "high",
        "source": "road_safety",
        "actions": [
            "Check that it is safe to approach; stay clear of traffic, smoke, fire, and leaking fuel.",
            "Ask someone nearby to call your local emergency number and give the exact road location and direction of traffic; this application cannot place the call.",
            "Do not move injured people unless they face immediate danger; keep them warm and follow the real call-taker's instructions.",
        ],
        "avoid": ["Do not enter a dangerous roadway or approach smoke, fire, or a fuel leak.", "Do not move an injured person unless staying there is immediately dangerous."],
        "questions": ["Are you in a safe place away from traffic, smoke, and fire?", "Has someone called the local emergency number?", "What road, direction of travel, or nearby landmark identifies the location?"],
        "call_now": True,
    },
    {
        "id": "cardiac_arrest",
        "label": "Possible cardiac arrest",
        "patterns": (
            r"\bcardiac arrest\b",
            r"\bheart (?:has )?stopped\b",
            r"\bunresponsive\b.{0,60}\b(?:not breathing|no normal breathing|only gasping|abnormal breathing)\b",
            r"\bunconscious\b.{0,60}\b(?:not breathing|no normal breathing|only gasping|abnormal breathing)\b",
            r"\b(?:not breathing|no normal breathing|only gasping|abnormal breathing)\b.{0,60}\b(?:unresponsive|unconscious|collapsed)\b",
        ),
        "severity": "critical",
        "source": "unresponsive",
        "actions": [
            "Call your local emergency number now, put the phone on speaker, and follow the call-taker's CPR instructions.",
            "Check responsiveness and normal breathing while help is being arranged; gasping is not normal breathing.",
            "If the person is unresponsive and not breathing normally, begin CPR as directed by the call-taker; use an AED as soon as available and follow its prompts.",
            "Tell the call-taker the person's approximate age; CPR instructions differ for infants and children.",
        ],
        "avoid": ["Do not delay the emergency call to answer questions.", "Do not give food, drink, or medicine by mouth."],
        "questions": ["Is the person responsive now?", "Are they breathing normally, or only gasping?", "Is the person an adult, child, or infant?"],
        "call_now": True,
    },
    {
        "id": "unconsciousness",
        "label": "Unconsciousness or unresponsiveness",
        "patterns": (r"\bunconscious\b", r"\bunresponsive\b", r"\bnot responding\b", r"\bcan't wake\b", r"\bcannot wake\b", r"\bpassed out\b", r"\bcollapsed\b"),
        "severity": "critical",
        "source": "unresponsive",
        "actions": [
            "Call your local emergency number now and put the phone on speaker.",
            "Check responsiveness and normal breathing while waiting for the call-taker; report exactly what you observe.",
            "If unresponsive and not breathing normally, start CPR and use an AED as directed by the call-taker.",
            "If unresponsive but breathing, follow the call-taker's instructions and keep checking breathing.",
        ],
        "avoid": ["Do not assume cardiac arrest from unresponsiveness alone; check breathing and follow the call-taker.", "Do not give food, drink, or medicine by mouth.", "Do not move the person if a head, neck, or back injury is suspected unless danger requires it."],
        "questions": ["Can you get a response by calling their name and gently tapping their shoulder (or foot for an infant)?", "Are they breathing normally?", "Could they have been injured or taken a substance?"],
        "call_now": True,
    },
    {
        "id": "severe_bleeding",
        "label": "Severe or uncontrolled bleeding",
        "patterns": (r"\bsevere (?:bleeding|blood loss)\b", r"\bheavy bleeding\b", r"\bbleeding (?:severely|heavily|won't stop|does not stop|uncontrollably)\b", r"\b(?:spurting|pouring) blood\b", r"\bblood is (?:spurting|pouring)\b"),
        "severity": "critical",
        "source": "bleeding",
        "actions": [
            "Call your local emergency number now; use gloves or a barrier if available.",
            "Apply firm, steady direct pressure over the wound with clean cloth or dressing.",
            "For life-threatening bleeding from an arm or leg, use a commercial tourniquet only if available and you know how; follow dispatcher instructions.",
            "Keep the person warm and watch their breathing and responsiveness until help arrives.",
        ],
        "avoid": ["Do not remove an embedded object; apply pressure around it.", "Do not stop direct pressure to inspect the wound repeatedly."],
        "questions": ["Where is the bleeding?", "Is blood spurting, flowing continuously, or soaking through?", "Is the person pale, weak, confused, or faint?"],
        "call_now": True,
    },
    {
        "id": "choking",
        "label": "Possible choking",
        "patterns": (r"\bchok(?:e|ing|ed)\b", r"\bfood (?:is )?stuck in (?:the )?(?:throat|airway)\b", r"\bobject (?:is )?blocking (?:the )?(?:throat|airway)\b"),
        "severity": "critical",
        "source": "choking",
        "actions": [
            "If the person cannot speak, cough effectively, or breathe, call your local emergency number and follow the call-taker's instructions.",
            "If they can cough forcefully, encourage coughing and watch closely for worsening.",
            "Tell the call-taker the person's age and whether they are responsive; choking care differs for infants and some adults.",
        ],
        "avoid": ["Do not perform a blind finger sweep; remove an object from the mouth only if you can clearly see it.", "Do not give food or drink while the airway may be blocked."],
        "questions": ["Can they speak, cough forcefully, and breathe?", "How old are they?", "Are they responsive?"],
        "call_now": True,
    },
    {
        "id": "stroke",
        "label": "Possible stroke",
        "patterns": (r"\bstroke\b", r"\bface (?:is )?droop(?:ing)?\b", r"\bslurred speech\b", r"\bsudden (?:one[- ]sided )?(?:weakness|numbness)\b", r"\barm weakness\b"),
        "severity": "critical",
        "source": "stroke",
        "actions": [
            "Call your local emergency number now and report possible stroke symptoms.",
            "Note the time symptoms began or when the person was last known well.",
            "Keep the person safe and with you; follow the call-taker's directions.",
        ],
        "avoid": ["Do not wait to see if symptoms improve.", "Do not give food, drink, or medicine unless a medical professional directs you."],
        "questions": ["When were they last known well?", "Is there face drooping, arm weakness, or speech difficulty?", "Are symptoms changing?"],
        "call_now": True,
    },
    {
        "id": "breathing_difficulty",
        "label": "Difficulty breathing",
        "patterns": (r"\b(?:difficulty|trouble|struggling) breathing\b", r"\b(?:can't|cannot) breathe\b", r"\bshortness of breath\b", r"\bgasping for breath\b", r"\bblue lips\b"),
        "severity": "critical",
        "source": "breathing",
        "actions": [
            "Call your local emergency number now for severe or worsening breathing difficulty, inability to speak normally, blue/gray color, or reduced responsiveness.",
            "If responsive, help them sit in the position that makes breathing easiest and stay with them.",
            "If they have their own prescribed emergency medicine, help them use it only as directed on their plan or by the call-taker.",
            "Keep checking breathing and responsiveness; follow the call-taker's instructions.",
        ],
        "avoid": ["Do not force them to lie flat or walk around.", "Do not give someone else's medication."],
        "questions": ["Can they speak a full sentence?", "When did this begin and is it worsening?", "Are they responsive and what is their age?"],
        "call_now": True,
    },
    {
        "id": "seizure",
        "label": "Seizure",
        "patterns": (r"\bseiz(?:ure|ing|ed)\b", r"\bconvuls(?:ion|ing)\b", r"\bfit episode\b"),
        "severity": "high",
        "source": "seizure",
        "actions": [
            "Move nearby hazards away and protect the person's head if you can do so safely; time the seizure.",
            "Stay with them and check breathing and responsiveness when the movements stop.",
            "Call emergency services for a first seizure, a seizure lasting 5 minutes or more, repeated seizures without recovery, injury, seizure in water, breathing difficulty, or if they do not regain awareness.",
            "Follow their established seizure care plan only if one is known and you are able to follow it safely.",
        ],
        "avoid": ["Do not restrain them.", "Do not put anything in their mouth or give food, drink, or medicine during the seizure."],
        "questions": ["Is the seizure still happening, and when did it start?", "Have they had a seizure before?", "Are they breathing normally and waking afterward?"],
        "call_now": False,
    },
    {
        "id": "burns",
        "label": "Burn or scald",
        "patterns": (r"\bburn(?:ed|t|s|ing)?\b", r"\bscald(?:ed|s|ing)?\b"),
        "severity": "high",
        "source": "burns",
        "actions": [
            "Make sure the scene is safe and stop the burning process if you can do so safely.",
            "For a thermal burn, cool the area with clean, cool running water; remove tight items near it unless stuck to the skin.",
            "Call emergency services for a large, deep, electrical or chemical burn, burns to the face/airway, or any breathing problem; get urgent medical assessment for significant burns.",
            "Watch for changes and follow the call-taker's instructions.",
        ],
        "avoid": ["Do not use ice, butter, or greasy products; do not break blisters or remove clothing stuck to the burn.", "Do not touch a person until an electrical source is confirmed safe."],
        "questions": ["What caused the burn: heat, chemical, or electricity?", "Where is it and how large is it?", "Is there facial injury, smoke exposure, or breathing difficulty?"],
        "call_now": False,
    },
    {
        "id": "poisoning",
        "label": "Possible poisoning or overdose",
        "patterns": (r"\bpoison(?:ed|ing)?\b", r"\boverdose\b", r"\bswallowed (?:too many|a harmful|unknown|poison|chemical|cleaning)\b", r"\btook (?:too many|an overdose|unknown pills)\b", r"\bdrug overdose\b"),
        "severity": "high",
        "source": ("poisoning", "overdose"),
        "actions": [
            "If the person is unconscious, having a seizure, or having trouble breathing, call your local emergency number now.",
            "Otherwise contact your local poison center or urgent medical service immediately for substance-specific advice.",
            "Keep the container or packaging available and tell responders what, how much, and when, if known.",
            "If opioid overdose is suspected and naloxone is available, use it according to its product instructions while emergency help is being called.",
        ],
        "avoid": ["Do not induce vomiting.", "Do not give food, drink, or home remedies unless poison professionals or emergency responders direct you."],
        "questions": ["What substance or medicine was involved?", "How much and when, if known?", "Is the person awake and breathing normally?"],
        "call_now": False,
    },
    {
        "id": "serious_injury_or_fracture",
        "label": "Suspected fracture or serious injury",
        "patterns": (r"\bfractur(?:e|ed)\b", r"\bbroken (?:bone|arm|leg|wrist|ankle|rib|back|neck)\b", r"\bserious injury\b", r"\b(?:head|neck|back|spine) injury\b"),
        "severity": "high",
        "source": "fracture",
        "actions": [
            "Keep the person still and support the injured area in the position found; arrange urgent medical assessment.",
            "Call emergency services now for severe trauma, trouble breathing, signs of shock, major bleeding, or suspected head, neck, or back injury.",
            "Watch breathing and responsiveness and follow the call-taker's directions.",
        ],
        "avoid": ["Do not straighten the injury, push bone back in, or ask the person to move it.", "Do not move someone with a possible head, neck, or back injury unless there is immediate danger."],
        "questions": ["Where is the injury and what happened?", "Is there severe bleeding, numbness, deformity, or trouble breathing?", "Could the head, neck, or back be injured?"],
        "call_now": False,
    },
)


def analyze_first_aid(text: str) -> dict[str, Any]:
    """Return local, auditable, advisory guidance; do not persist the input."""
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    matches = [rule for rule in _RULES if any(re.search(pattern, normalized) for pattern in rule["patterns"])]
    # A no-normal-breathing signal is immediately life-threatening, even if
    # the caller did not use the word unconscious.
    if not matches and re.search(r"\b(?:not breathing|only gasping|no normal breathing)\b", normalized):
        matches = [_RULES[0]]
    selected = matches[0] if matches else None
    ambiguous = selected is None or len(matches) > 1
    if selected is None:
        immediate = [
            "Make sure it is safe to approach; do not put yourself in danger.",
            "If anyone may be in immediate danger or you are unsure, call your local emergency number now and follow the call-taker.",
            "Stay with the person if safe, observe breathing and responsiveness, and report changes.",
        ]
        avoid = ["Do not give food, drink, or medicine to someone who is unconscious or not fully alert.", "Do not move someone with a possible head, neck, or back injury unless there is immediate danger."]
        questions = ["What happened?", "Is the person awake and breathing normally?", "Is there severe bleeding, choking, a burn, poisoning, or another immediate danger?"]
        scenario = {"id": "unknown_or_ambiguous", "label": "Unknown or unclear emergency"}
        severity = "unknown"
        call_now = True
        sources = [_source("general"), _source("unresponsive")]
        related_guidance: list[dict[str, Any]] = []
    else:
        immediate = list(selected["actions"])
        avoid = list(selected["avoid"])
        questions = list(selected["questions"])
        scenario = {"id": selected["id"], "label": selected["label"]}
        severity = selected["severity"]
        call_now = bool(selected["call_now"])
        if selected["id"] == "seizure":
            call_now = bool(re.search(
                r"\b(?:first seizure|more than 5 minutes|over 5 minutes|longer than 5 minutes|multiple seizures|several seizures|not recovering|not waking|injured|in water|difficulty breathing|pregnant|infant under 6 months|overdose|poisoned)\b",
                normalized,
            ))
        elif selected["id"] == "burns":
            call_now = bool(re.search(
                r"\b(?:chemical|electrical|large|deep|face|airway|smoke inhalation|difficulty breathing|hands|feet|genitals|joint)\b",
                normalized,
            ))
        elif selected["id"] == "serious_injury_or_fracture":
            call_now = bool(re.search(
                r"\b(?:head|neck|back|spine|severe trauma|shock|trouble breathing|major bleeding|unconscious|unresponsive)\b",
                normalized,
            ))
        elif selected["id"] == "poisoning":
            call_now = bool(re.search(
                r"\b(?:unconscious|unresponsive|seizure|seizing|trouble breathing|difficulty breathing|can't breathe|cannot breathe|collapsed|can't wake|cannot wake)\b",
                normalized,
            ))
        source_keys = list(dict.fromkeys(
            source_key
            for rule in matches
            for source_key in (rule["source"] if isinstance(rule["source"], (list, tuple)) else [rule["source"]])
        ))
        sources = [_source(key) for key in source_keys]
        related_guidance = [
            {
                "scenario": {"id": rule["id"], "label": rule["label"]},
                "immediate_actions": list(rule["actions"]),
                "actions_to_avoid": list(rule["avoid"]),
                "critical_follow_up_questions": list(rule["questions"]),
            }
            for rule in matches[1:]
        ]
        if len(matches) > 1:
            immediate.insert(0, "More than one emergency sign was detected; treat the most urgent sign first and have a human operator review the situation.")
            safety_flags = [f"additional_scenario:{rule['id']}" for rule in matches[1:]]
        else:
            safety_flags = []
    # Safety override is independent of any classifier or external model.
    critical_override = bool(
        selected and selected["severity"] == "critical"
    ) or bool(re.search(r"\b(?:unconscious|unresponsive|not breathing|only gasping|severe bleeding|spurting blood)\b", normalized))
    if selected is None:
        safety_flags = ["scenario_unknown", "human_review_required"]
    if critical_override:
        severity = "critical"
        call_now = True
        if "life_threatening_indicator" not in safety_flags:
            safety_flags.append("life_threatening_indicator")
    else:
        safety_flags = safety_flags if selected else ["scenario_unknown", "human_review_required"]
    human_review = ambiguous or severity in {"critical", "high", "unknown"}
    if selected and selected["id"] == "poisoning" and not call_now:
        escalation_level = "contact_local_poison_center_or_urgent_medical_service_now"
        escalation_message = "Contact your local poison center or urgent medical service now for substance-specific advice; call emergency services immediately if breathing, responsiveness, or seizures are affected."
    elif call_now:
        escalation_level = "call_local_emergency_services_now"
        escalation_message = "Call your local emergency number; no call is placed by this application."
    elif selected and selected["id"] == "burns":
        escalation_level = "urgent_medical_assessment_if_red_flags"
        escalation_message = "Seek prompt medical assessment for significant burns; call emergency services for the listed burn or breathing red flags."
    elif selected and selected["id"] == "seizure":
        escalation_level = "emergency_services_if_seizure_red_flags"
        escalation_message = "Call emergency services for a first, prolonged, repeated, injurious, water-related seizure, breathing difficulty, or failure to recover."
    else:
        escalation_level = "urgent_medical_assessment_if_red_flags"
        escalation_message = "Seek prompt medical assessment if the condition is significant or worsening; follow local emergency advice."

    return {
        "detected_scenario": scenario,
        "additional_scenarios": [
            {"id": rule["id"], "label": rule["label"]} for rule in matches[1:]
        ],
        "related_guidance": related_guidance,
        "severity": severity,
        "safety_flags": sorted(set(safety_flags)),
        "immediate_actions": immediate,
        "actions_to_avoid": avoid,
        "critical_follow_up_questions": questions,
        "emergency_escalation": {
            "recommended": call_now,
            "professional_assistance_recommended": severity in {"critical", "high", "unknown"},
            "level": escalation_level,
            "message": escalation_message,
        },
        "human_review_required": human_review,
        "sources": sources,
        "engine": {"id": ENGINE_ID, "version": ENGINE_VERSION, "source_type": "deterministic_rules"},
        "ambiguous_or_unsupported": ambiguous,
        "limitations": "Advisory first-aid information only; not a diagnosis, not clinically validated, and not a substitute for local emergency services or dispatcher instructions.",
    }
