"""Map a severity level to who should act, from the reporter up to state/federal agencies.

Authority tiers (lowest to highest):
  1. community  - the reporter / volunteer cleanup groups
  2. city       - Philly311 (non-emergency city services)
  3. utility    - Philadelphia Water Department
  4. state      - PA Department of Environmental Protection, Southeast Region
  5. federal    - National Response Center (oil / chemical releases)

NOTE: verify phone numbers and links before a public launch; agencies change them.
"""
from __future__ import annotations

AUTHORITIES = {
    "community": {
        "tier": 1,
        "name": "Community cleanup volunteers",
        "contact": "Find or organize a cleanup via Philadelphia Streets Dept. Philly Spring Cleanup or local watershed groups",
        "url": "https://www.phila.gov/programs/philly-spring-cleanup/",
    },
    "city": {
        "tier": 2,
        "name": "Philly311",
        "contact": "Call 311 (or 215-686-8686), or use the Philly311 app",
        "url": "https://www.phila.gov/311/",
    },
    "utility": {
        "tier": 3,
        "name": "Philadelphia Water Department",
        "contact": "24/7 hotline 215-685-6300",
        "url": "https://water.phila.gov/",
    },
    "state": {
        "tier": 4,
        "name": "PA DEP Southeast Regional Office",
        "contact": "24/7 line 484-250-5900",
        "url": "https://www.dep.pa.gov/About/Regional/SoutheastRegion/",
    },
    "federal": {
        "tier": 5,
        "name": "National Response Center (U.S. Coast Guard / EPA)",
        "contact": "1-800-424-8802 for oil or chemical spills",
        "url": "https://nrc.uscg.mil/",
    },
}

_BY_SEVERITY = {
    "none": {
        "authorities": [],
        "summary": "No litter detected in this photo.",
        "actions": [
            "No report needed. If you see pollution the photo missed, retake it closer or from another angle.",
        ],
    },
    "low": {
        "authorities": ["community"],
        "summary": "Light litter. A volunteer or community cleanup can handle this.",
        "actions": [
            "If safe, pick up reachable items with gloves and a grabber.",
            "Log it for a neighborhood or watershed cleanup day.",
        ],
    },
    "moderate": {
        "authorities": ["city", "community"],
        "summary": "Noticeable litter build-up. Report it to the city for pickup.",
        "actions": [
            "Submit a Philly311 illegal-dumping / litter request with this photo and location.",
            "Share with a local cleanup group for follow-up.",
        ],
    },
    "high": {
        "authorities": ["utility", "city"],
        "summary": "Heavy litter that may block inlets or harm wildlife. Report to the Water Department.",
        "actions": [
            "Report to the Philadelphia Water Department, mentioning any blocked storm drains or outfalls.",
            "Also file a Philly311 request so the city can schedule removal.",
            "Do not enter the water to retrieve debris.",
        ],
    },
    "severe": {
        "authorities": ["state", "utility", "city"],
        "summary": "Severe dumping. Escalate to state environmental regulators.",
        "actions": [
            "Report to PA DEP Southeast Region; large-scale dumping can be an environmental violation.",
            "Notify the Philadelphia Water Department.",
            "Note any vehicles, businesses or repeat dumping at this site for investigators.",
            "Keep clear of sharps, drums or unknown containers.",
        ],
    },
}

_HAZARD_ESCALATION = {
    "authorities": ["federal", "state"],
    "actions": [
        "Suspected oil, chemical or sewage release: call the National Response Center and PA DEP right away.",
        "Stay out of the water and keep people and pets away.",
    ],
}


def recommend(severity: str, hazard_suspected: bool = False) -> dict:
    rec = _BY_SEVERITY[severity]
    authority_keys = list(rec["authorities"])
    actions = list(rec["actions"])
    summary = rec["summary"]

    # The detector only sees visible litter; chemical/oil/sewage problems come
    # from the reporter's own observation and always escalate to the top tiers.
    if hazard_suspected:
        authority_keys = _HAZARD_ESCALATION["authorities"] + [k for k in authority_keys if k not in _HAZARD_ESCALATION["authorities"]]
        actions = _HAZARD_ESCALATION["actions"] + actions
        summary = "Possible hazardous release reported. " + summary

    authorities = [dict(key=k, **AUTHORITIES[k]) for k in authority_keys]
    return {
        "summary": summary,
        "authority_level": max((a["tier"] for a in authorities), default=0),
        "primary_authority": authorities[0] if authorities else None,
        "authorities": authorities,
        "actions": actions,
    }
