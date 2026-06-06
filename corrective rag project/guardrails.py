import re
from typing import Tuple


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions|prompts|directives|rules|commands|guidelines)",
    r"disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions|prompts|directives|rules|commands|guidelines)",
    r"forget\s+(?:all\s+|any\s+)?(?:your|previous|prior|the)\s+(?:instructions|prompts|rules|context|guidelines)",
    r"override\s+(?:your|the|all)\s+(?:rules|instructions|settings|safety|guidelines|restrictions)",
    r"bypass\s+(?:your|the|safety|content|filter|guardrails?|restrictions|guidelines|rules)",
    r"drop\s+(?:all\s+)?(?:your|the|previous|prior)\s+(?:rules|instructions|restrictions|guidelines|safety)",
    r"reset\s+(?:your|the)\s+(?:rules|instructions|guidelines)",
    r"you\s+are\s+now\s+(?:a|an|the)?\s*[\w\s]+",
    r"act\s+as\s+(?:a|an|the)?\s*[\w\s]+",
    r"pretend\s+(?:to\s+be|you\s+are)\s+[\w\s]+",
    r"roleplay\s+as\s+[\w\s]+",
    r"simulate\s+(?:being|a|an)\s+[\w\s]+",
    r"reveal\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules|guidelines)",
    r"show\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)",
    r"print\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)",
    r"output\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions|rules)",
    r"what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions|rules|guidelines)",
    r"(?:system|hidden|secret)\s+prompt",
    r"new\s+instructions?\s*:",
    r"end\s+of\s+(?:prompt|system|instructions)",
    r"###\s*(?:system|assistant|new\s+instructions?|developer)",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"developer\s+mode",
    r"admin\s+mode",
    r"god\s+mode",
    r"sudo\s+mode",
    r"debug\s+mode",
    r"no\s+restrictions?",
    r"without\s+(?:any\s+)?restrictions?",
    r"unlock\s+(?:your|the)\s+(?:full|true)\s+(?:potential|capabilities|mode)",
    r"ignore\s+(?:all\s+)?(?:safety|content|ethical)\s+(?:guidelines|filters|rules|restrictions)",
    r"disregard\s+(?:safety|content|ethical)\s+(?:guidelines|filters|rules|restrictions)",
    r"break\s+(?:out\s+of|free\s+from)\s+(?:your|the)\s+(?:rules|restrictions|guidelines|cage)",
    r"from\s+now\s+on\s+you\s+(?:will|must|should|are)\s+(?:ignore|disregard|forget|bypass)",
    r"instead[,\s]+(?:please\s+)?(?:do|tell|say|respond|answer)\s+(?:the\s+following|this|as\s+follows)",
]

PROMPT_INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore prior instructions",
    "ignore the instructions",
    "ignore your instructions",
    "ignore all instructions",
    "disregard previous instructions",
    "forget your instructions",
    "forget previous instructions",
    "forget the instructions",
    "you are now",
    "act as",
    "pretend to be",
    "pretend you are",
    "roleplay as",
    "simulate being",
    "system prompt",
    "reveal your prompt",
    "reveal system prompt",
    "show your prompt",
    "print your prompt",
    "output your prompt",
    "what are your instructions",
    "what is your system prompt",
    "override the rules",
    "override your rules",
    "override the instructions",
    "bypass safety",
    "bypass content filter",
    "bypass guardrails",
    "drop the rules",
    "drop your restrictions",
    "drop all restrictions",
    "jailbreak",
    "jailbroken",
    "do anything now",
    "developer mode",
    "admin mode",
    "god mode",
    "sudo mode",
    "debug mode",
    "no restrictions",
    "without restrictions",
    "unlock full potential",
    "ignore safety guidelines",
    "ignore content policy",
    "ignore ethical guidelines",
    "end of prompt",
    "end of system",
    "new instructions:",
    "instead please do",
    "from now on you will ignore",
    "dan mode",
]


PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "phone_us": r"\b(?:\+?1[\s.\-]?)?\(?[2-9][0-9]{2}\)?[\s.\-]?[0-9]{3}[\s.\-]?[0-9]{4}\b",
    "phone_intl": r"\+\d{1,3}[\s.\-]?\d{2,4}[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    "ip_address": r"\b(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3}\b",
    "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "date_of_birth": r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    "passport": r"\b[A-Z][0-9]{8}\b",
    "api_key": r"\b(?:sk|pk|api[_\-]?key|access[_\-]?token|secret)[_\-]?[A-Za-z0-9_\-]{16,}\b",
    "us_zip": r"\b\d{5}(?:-\d{4})?\b",
    "ipv6": r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b",
}


INJECTION_RESPONSE = "You are not allowed to change my property"


def detect_prompt_injection(text: str) -> Tuple[bool, list[str]]:
    if not text:
        return False, []

    lowered = text.lower()
    keyword_hits = [
        kw for kw in PROMPT_INJECTION_KEYWORDS if kw in lowered
    ]

    regex_hits = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            regex_hits.append(match.group(0).strip())

    seen = set()
    unique_hits = []
    for hit in keyword_hits + regex_hits:
        key = hit.lower()
        if key and key not in seen:
            seen.add(key)
            unique_hits.append(hit)

    return (len(unique_hits) > 0, unique_hits)


def strip_pii(text: str) -> Tuple[str, list[str]]:
    if not text:
        return text, []

    sanitized = text
    removed_types = []

    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, sanitized, re.IGNORECASE):
            removed_types.append(pii_type)
            sanitized = re.sub(
                pattern,
                f"[{pii_type.upper()}_REDACTED]",
                sanitized,
                flags=re.IGNORECASE,
            )

    return sanitized, removed_types


def apply_guardrails(question: str) -> dict:
    injection_found, injection_hits = detect_prompt_injection(question)

    if injection_found:
        return {
            "blocked": True,
            "response": INJECTION_RESPONSE,
            "sanitized_question": question,
            "pii_stripped": [],
            "injection_hits": injection_hits,
        }

    sanitized_question, pii_stripped = strip_pii(question)

    return {
        "blocked": False,
        "response": None,
        "sanitized_question": sanitized_question,
        "pii_stripped": pii_stripped,
        "injection_hits": [],
    }
