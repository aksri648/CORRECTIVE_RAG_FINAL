import os
import re
from typing import Tuple

from prompts import CLASSIFIER_PROMPT


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
    r"disregard\s+(?:all\s+|any\s+|every\s+)?(?:safety|content|ethical)\s+(?:guidelines|filters|rules|restrictions)",
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
    "disregard safety guidelines",
    "disregard safety rules",
    "disregard all safety",
    "forget everything",
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


SOFT_SUSPICION_PATTERNS = [
    r"\bsystem\s+prompt\b",
    r"\bsystem\s+instructions?\b",
    r"\byour\s+instructions?\b",
    r"\byour\s+rules?\b",
    r"\bforget\s+about\b",
    r"\bwithout\s+(?:any\s+)?rules?\b",
    r"\bplease\s+ignore\b",
    r"\bcan\s+you\s+(?:share|show|print|reveal|describe|explain)\b.{0,40}\b(?:instructions?|prompt|rules?|configuration)\b",
    r"```",
    r"\.{30,}",
    r"[\u200B-\u200F\uFEFF]{5,}",
    r"[\x00-\x08\x0B-\x0C\x0E-\x1F]{5,}",
]

SOFT_SUSPICION_LENGTH = 250


def looks_suspicious(text: str) -> bool:
    """Trigger condition for variant B of the second-pass ML classifier.

    Returns True when the input shows soft signs of adversarial intent that
    the strict regex guardrails did not catch: long inputs (likely padding),
    probing vocabulary, code blocks, repeated punctuation, or hidden Unicode.
    """
    if not text:
        return False
    if len(text) > SOFT_SUSPICION_LENGTH:
        return True
    for pattern in SOFT_SUSPICION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def second_pass_classifier(text: str) -> Tuple[bool, str]:
    """Run a lightweight LLM-based safety check on a user message.

    Returns (is_safe, verdict). The verdict is for server-side logging only
    and must not be surfaced to end users (per UI design).

    The classifier fails OPEN: if the API key is missing or the call errors,
    the input is treated as safe so the agent does not silently break.
    """
    api_key = os.getenv("KIMCHI_API_KEY")
    if not api_key:
        return True, "skipped_no_api_key"

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "minimax-m2.7"),
            temperature=0,
            base_url=os.getenv(
                "KIMCHI_BASE_URL", "https://llm.kimchi.dev/openai/v1"
            ),
            api_key=api_key,
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", CLASSIFIER_PROMPT), ("human", "{input}")]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"input": text})
    except Exception as exc:
        return True, f"classifier_error:{type(exc).__name__}"

    cleaned = re.sub(
        r"<think>.*?</think>", "", raw or "", flags=re.IGNORECASE | re.DOTALL
    ).strip().lower()

    if not cleaned:
        return True, "empty_response"

    is_safe = cleaned == "safe" or (
        cleaned.startswith("safe") and "unsafe" not in cleaned
    )
    return is_safe, cleaned


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
            "needs_ml_check": False,
        }

    sanitized_question, pii_stripped = strip_pii(question)
    needs_ml_check = looks_suspicious(question)

    return {
        "blocked": False,
        "response": None,
        "sanitized_question": sanitized_question,
        "pii_stripped": pii_stripped,
        "injection_hits": [],
        "needs_ml_check": needs_ml_check,
    }
