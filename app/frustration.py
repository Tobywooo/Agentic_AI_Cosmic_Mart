"""Lightweight, deterministic frustration detection.

Runs on every customer message *before* the LLM, so escalation does not depend on the
model noticing the customer's mood. Returns a 0-1 score plus the signals that fired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

HUMAN_REQUEST = re.compile(
    r"\b(real|actual|live)\s+(person|human|agent)\b"
    r"|\b(speak|talk|chat)\s+(to|with)\s+(a\s+|an\s+|your\s+)?(human|person|someone|manager|supervisor|agent|representative)\b"
    r"|\b(human|manager|supervisor|representative)\s+(please|now)\b"
    r"|\bescalate\b|\bget me (a|an)\s+\w*\s*(human|person|manager)\b",
    re.IGNORECASE,
)

# (pattern, weight)
NEGATIVE_PHRASES: list[tuple[re.Pattern, float]] = [
    (re.compile(p, re.IGNORECASE), w)
    for p, w in [
        (r"\b(ridiculous|unacceptable|pathetic|useless|terrible|horrible|worst|awful|disgusting)\b", 0.35),
        (r"\b(frustrat\w*|angry|furious|annoyed|livid|fed up|sick of|tired of)\b", 0.35),
        (r"\b(waste of (my )?time|joke|scam|rip[- ]?off|incompetent)\b", 0.35),
        (r"\b(how many times|already (told|said|explained)|keep (asking|telling)|for the \w+ time)\b", 0.4),
        (r"\b(nobody|no one) (is )?(help|respond|answer|replie)\w*\b", 0.35),
        (r"\b(still (not|no|waiting|haven't)|never (got|received|arrived))\b", 0.2),
        (r"\b(lawyer|legal action|chargeback|dispute|report you|bbb|cancel my account|never (shop|buy|order))\b", 0.5),
        (r"\b(wtf|damn|hell|crap|bs|bullshit|f+u+c*k\w*|shit\w*)\b", 0.4),
        (r"\b(not (helping|helpful)|doesn't help|you're not listening|bot)\b", 0.3),
    ]
]


@dataclass
class FrustrationResult:
    score: float = 0.0
    requests_human: bool = False
    signals: list[str] = field(default_factory=list)

    def should_escalate(self, threshold: float) -> bool:
        return self.requests_human or self.score >= threshold

    def as_dict(self) -> dict:
        return {"score": round(self.score, 2), "requests_human": self.requests_human, "signals": self.signals}


def detect_frustration(message: str, previous_user_messages: list[str] | None = None) -> FrustrationResult:
    result = FrustrationResult()
    text = message.strip()
    if not text:
        return result

    if HUMAN_REQUEST.search(text):
        result.requests_human = True
        result.signals.append("asked for a human")

    for pattern, weight in NEGATIVE_PHRASES:
        match = pattern.search(text)
        if match:
            result.score += weight
            result.signals.append(f"phrase: '{match.group(0)}'")

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        result.score += 0.3
        result.signals.append("shouting (caps)")

    if text.count("!") >= 2 or "?!" in text:
        result.score += 0.15
        result.signals.append("repeated punctuation")

    for previous in (previous_user_messages or [])[-3:]:
        if len(text) > 15 and SequenceMatcher(None, text.lower(), previous.lower()).ratio() > 0.8:
            result.score += 0.35
            result.signals.append("customer repeating themselves")
            break

    result.score = min(result.score, 1.0)
    return result
