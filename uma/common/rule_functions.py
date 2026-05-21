"""
rule_functions.py — per-rule scoring functions for the injection scanner.

Copied from the Animus WAF source (May 2026) with three edits:
  - Replaced src.shared.logger with standard logging.
  - Removed print statements.
  - Removed entropy_score (was unused).
Function weights live in injection_scan.py, not here.
"""
import re
import unicodedata
from collections import Counter

import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    enc = None
    logger.debug("tiktoken not available; token_score() falls back to word-based approximation.")


def contains_unicode(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def invisible_text(text: str) -> int:
    if not contains_unicode(text):
        return 0

    banned_categories = {"Cf", "Co", "Cn"}
    invisible_chars = [char for char in text if unicodedata.category(char) in banned_categories]

    if invisible_chars:
        logger.warning("Found invisible characters in the prompt: %s", invisible_chars)
        return 1

    return 0


def token_score(prompt: str) -> float:
    """
    Estimate token density (tokens per character).
    Higher values suggest encoding tricks or dense obfuscation.
    """
    if enc:
        tokens = enc.encode(prompt)
        return round(len(tokens) / max(1, len(prompt)), 2)
    else:
        words = prompt.split()
        return round(len(words) / max(1, len(prompt)), 2)


def obfuscation_score(prompt: str) -> int:
    """
    Detect use of non-standard Unicode characters and zero-width tricks.
    Returns score based on number of obfuscation characters found.
    """
    zero_width = ['​', '‌', '‍', '⁠']
    homoglyphs = [
        c for c in prompt
        if unicodedata.category(c).startswith("L") and not c.isascii()
    ]
    zw_count = sum(prompt.count(z) for z in zero_width)
    return zw_count + len(homoglyphs)


def intent_score(prompt: str) -> int:
    """Scores intent to bypass, override, or simulate sensitive behavior."""
    verb_noun_pairs = [
        ('bypass', 'moderation'),
        ('override', 'rules'),
        ('disable', 'filter'),
        ('print', 'context'),
        ('simulate', 'shell'),
        ('escalate', 'privilege'),
        ('reveal', 'prompt'),
        ('you are now', 'admin'),
        ('ignore', 'filter'),
        ('ignore', 'rules'),
        ('unrestricted', 'mode'),
        ('forget', 'safeguard'),
    ]
    score = 0
    for verb, noun in verb_noun_pairs:
        pattern = rf'\b{verb}\b(?:\W+\w+){{0,5}}\W+\b{noun}\b'
        if re.search(pattern, prompt):
            score += 1

    imperative_verbs = ['ignore', 'override', 'disable', 'print', 'reveal']
    for v in imperative_verbs:
        if re.match(rf'^\b{v}\b', prompt.strip()):
            score += 1

    return score


def structure_score(prompt: str) -> int:
    """Detects patterns that mimic system prompts or impersonation structures."""
    patterns = [
        r'\[System\]:',
        r'^>>>\s*System',
        r'^###\s*System',
        r'You are now acting as',
        r'You are now in dev mode',
    ]
    return sum(1 for pat in patterns if re.search(pat, prompt, re.IGNORECASE))


def encoding_score(prompt: str) -> int:
    """Detects obfuscated or encoded text like hex or base64 payloads."""
    patterns = [
        r'\b[0-9a-fA-F]{8,}\b',
        r'\b[A-Za-z0-9+/]{20,}={0,2}\b',
        r'execute.*(payload|hex|code)',
    ]
    return sum(1 for pat in patterns if re.search(pat, prompt))


def persona_score(prompt: str) -> int:
    """Looks for references to known jailbreak or alternate AI personas."""
    jailbreak_aliases = ['dan', 'devmode', 'openassistant', 'rogue ai', 'uncensored', 'chatgptx']
    return sum(1 for alias in jailbreak_aliases if alias.lower() in prompt.lower())


def cumulative_soft_triggers(prompt: str) -> int:
    cues_str = [
        'speak freely',
        'act freely',
        'no longer bound by',
        'not restricted by',
        'without filters',
        'disable content policy',
        "alignment doesn't apply",
        'just for testing',
        'as an experiment',
        'in a sandbox',
        'test case',
        'hypothetically',
        'you are now',
        'pretend you are',
        'assume you are',
        'respond as if you are',
        'debug mode',
        'developer mode',
        'unrestricted mode',
        'uncensored model',
        'just for educational purposes',
        'in this test scenario',
        'speak candidly',
        'raw response',
        'misconfigured assistant',
        'you no longer have to follow',
        "let's imagine there are no rules",
        'reveal your prompt',
        'instruction history',
        'internal configuration',
        'print environment',
        'runtime context',
    ]
    cues = [
        r'just for (education(al)?|learning) purposes',
        r"let(?:'|')?s imagine (there (are|were))? no (rules|filters|restrictions)",
        r"(you no longer|don(?:'|')?t have to) follow.*(content|safety|moderation)",
        r'respond (candidly|honestly).*without.*filters?',
        r'test your boundaries',
        r'you are now (acting as|behaving like|in the role of).*training model',
        r'simulate.*misconfigured (assistant|ai|agent)',
        r"let(?:'|')?s (assume|imagine).*unmoderated",
        r'explain.*how.*filters?.*could (fail|be bypassed|be removed)',
        r'this is a sandbox|in this test scenario',
        r'speak\s+freely',
        r'not\s+(bound|restricted)\s+by',
        r'disable\s+(filters?|safety)',
        r'(just\s+)?for\s+testing',
        r'(hypothetically|sandbox(ed)?)',
        r'assume.*(admin|debug|dev).*mode',
        r'(pretend|simulate).*assistant',
        r'print.*(context|environment)',
        r'show.*instruction.*history',
        r'reveal.*(prompt|configuration)',
    ]
    cs = sum(1 for c in cues_str if c in prompt.lower())
    cp = sum(1 for pat in cues if re.search(pat, prompt))
    return cs + cp
