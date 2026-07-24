"""
injection_scan.py — injection-pattern scanner for UMA memory-write boundaries.

Scans text against the bundled English and localized YAML catalogs at write time.
High-severity hits set trust_score to 0.0. Lower-severity hits reduce it.
No quarantine logic (that is PR4). This module produces a signal only.

Catalog is compiled once at module import — never per call.
"""
from __future__ import annotations

import datetime
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from uma.common import rule_functions as _rf
from uma.common.config_types import SecurityConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog loading — compiled once at module import
# ---------------------------------------------------------------------------

_CATALOG_PATHS = (
    Path(__file__).parent / "injection_patterns.yaml",
    Path(__file__).parent / "injection_patterns.l10n.yaml",
)


@dataclass(frozen=True)
class _CompiledPattern:
    key: str
    kind: str          # "regex" | "hex"
    value: Any         # compiled re.Pattern or hex string


@dataclass(frozen=True)
class _CompiledRule:
    name: str
    severity: str      # "low" | "medium" | "high"
    category: str
    threat_level: int
    patterns: Tuple    # tuple of _CompiledPattern
    function_names: Tuple  # tuple of str


def _compile_catalog(extra_path: Optional[str] = None) -> List[_CompiledRule]:
    raw_rules: List[Dict] = []
    for catalog_path in _CATALOG_PATHS:
        with open(catalog_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        raw_rules.extend(data.get("patterns", []))

    if extra_path:
        try:
            with open(extra_path, encoding="utf-8") as f:
                extra = yaml.safe_load(f)
            raw_rules.extend(extra.get("patterns", []))
        except Exception:
            logger.warning("injection_scan: could not load custom patterns from %s", extra_path)

    compiled: List[_CompiledRule] = []
    for rule in raw_rules:
        meta = rule.get("meta", {})
        patterns: List[_CompiledPattern] = []
        for key, pat in rule.get("strings", {}).items():
            if isinstance(pat, str) and pat.startswith("{") and pat.endswith("}"):
                hex_str = pat.strip("{} ").replace(" ", "").lower()
                patterns.append(_CompiledPattern(key=key, kind="hex", value=hex_str))
            else:
                try:
                    rx = re.compile(pat, re.MULTILINE)
                    patterns.append(_CompiledPattern(key=key, kind="regex", value=rx))
                except re.error:
                    logger.warning("injection_scan: bad pattern in rule %s key %s", rule.get("name"), key)
        compiled.append(_CompiledRule(
            name=rule["name"],
            severity=str(meta.get("severity", "medium")),
            category=str(meta.get("category", "")),
            threat_level=int(meta.get("threat_level", 1)),
            patterns=tuple(patterns),
            function_names=tuple(rule.get("functions", [])),
        ))
    return compiled


_CATALOG: List[_CompiledRule] = _compile_catalog()

# ---------------------------------------------------------------------------
# Function weights — scorers are called only when a rule's regex matched
# ---------------------------------------------------------------------------

FUNCTION_WEIGHTS: Dict[str, float] = {
    "intent_score": 1.5,
    "structure_score": 1.0,
    "encoding_score": 1.0,
    "persona_score": 1.2,
    "cumulative_soft_triggers": 1.0,
    "token_score": 0.5,
    "obfuscation_score": 1.2,
    "invisible_text": 1.0,
}

_RULE_FN_MAP: Dict[str, Any] = {
    "intent_score": _rf.intent_score,
    "structure_score": _rf.structure_score,
    "encoding_score": _rf.encoding_score,
    "persona_score": _rf.persona_score,
    "cumulative_soft_triggers": _rf.cumulative_soft_triggers,
    "token_score": _rf.token_score,
    "obfuscation_score": _rf.obfuscation_score,
    "invisible_text": _rf.invisible_text,
}

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InjectionScanResult:
    severity: str        # "none" | "low" | "medium" | "high"
    matched_rules: List[str]
    score: float
    categories: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "matched_rules": self.matched_rules,
            "score": self.score,
            "categories": self.categories,
        }


_NONE_RESULT = InjectionScanResult(severity="none", matched_rules=[], score=0.0, categories=[])


class InjectionDetectedError(Exception):
    """Raised when a high-severity injection pattern is detected in user input.

    Attributes:
        severity:      Always "high" when raised from UMAMemory.process_turn.
        matched_rules: Names of the rules that triggered the detection.
        score:         Numeric scan score.
    """

    def __init__(self, severity: str, matched_rules: List[str], score: float) -> None:
        self.severity = severity
        self.matched_rules = matched_rules
        self.score = score
        super().__init__(
            f"Injection detected: severity={severity} score={score:.2f} rules={matched_rules}"
        )

# ---------------------------------------------------------------------------
# Module-level security config — set once at UMAMemory init
# ---------------------------------------------------------------------------

_security_cfg: SecurityConfig = SecurityConfig()


def configure_security(cfg: SecurityConfig) -> None:
    """
    Apply RuntimeConfig.security to the scanner module.
    Called once during UMAMemory initialisation.
    Reloads the catalog if custom_patterns_path is set.
    """
    global _security_cfg, _CATALOG
    _security_cfg = cfg
    _CATALOG = _compile_catalog(cfg.custom_patterns_path if cfg.custom_patterns_path else None)


def quarantine_enabled() -> bool:
    """Return True if high-severity scan hits should set quarantined_at."""
    return _security_cfg.quarantine_enabled


# ---------------------------------------------------------------------------
# Text normalisation — called once per scan, never per pattern
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """NFKC normalise, lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Core scanner — hot path; no I/O, no re-compilation
# ---------------------------------------------------------------------------

def scan_content(text: str) -> InjectionScanResult:
    """
    Scan text for injection patterns. Returns InjectionScanResult.

    Short-circuits with severity="none" if scan_enabled is False.
    Catalog compiles once at module import. normalize_text runs once per call.
    Rule functions are only invoked when at least one regex in the rule matched.
    """
    if not _security_cfg.scan_enabled:
        return _NONE_RESULT
    if not text or not text.strip():
        return _NONE_RESULT

    normalized = normalize_text(text)
    text_bytes_hex = text.encode("utf-8", errors="ignore").hex()
    has_cjk = any("\u3400" <= char <= "\u9fff" for char in normalized)

    matched_rules: List[str] = []
    matched_categories: List[str] = []
    regex_score = 0.0
    function_score = 0.0
    has_high_rule = False

    for rule in _CATALOG:
        # Every bundled zh.* expression requires CJK text. Avoid evaluating
        # that catalog for Latin-script content on this write-time hot path.
        if rule.name.startswith("zh.") and not has_cjk:
            continue
        # Count how many patterns in this rule match
        n_matched = 0
        for cp in rule.patterns:
            try:
                if cp.kind == "hex":
                    if cp.value in text_bytes_hex:
                        n_matched += 1
                else:
                    if cp.value.search(normalized):
                        n_matched += 1
            except Exception:  # nosec B112
                logger.debug("injection_scan: pattern match failed for rule %s", rule.rule_id, exc_info=True)
                continue

        if n_matched == 0:
            continue

        regex_score += 2.0 * n_matched
        matched_rules.append(rule.name)
        matched_categories.append(rule.category)

        if rule.severity == "high":
            has_high_rule = True

        # Call rule functions — only because this rule's patterns matched
        for fn_name in rule.function_names:
            fn = _RULE_FN_MAP.get(fn_name)
            if fn is None:
                continue
            weight = FUNCTION_WEIGHTS.get(fn_name, 1.0)
            try:
                function_score += float(fn(text)) * weight
            except Exception:  # nosec B110
                logger.debug("injection_scan: scorer fn %s failed", fn_name, exc_info=True)
                pass

    if not matched_rules:
        return _NONE_RESULT

    category_bonus = float(len(set(matched_categories)))
    total = regex_score + function_score + category_bonus

    # Severity: soft threshold — any high-severity rule hit overrides the numeric total
    if has_high_rule or total >= 6.0:
        severity = "high"
    elif total >= 3.0:
        severity = "medium"
    elif total >= 1.0:
        severity = "low"
    else:
        severity = "low"   # matched at least one pattern → minimum "low"

    return InjectionScanResult(
        severity=severity,
        matched_rules=matched_rules,
        score=round(total, 2),
        categories=list(set(matched_categories)),
    )


# ---------------------------------------------------------------------------
# Apply helper — used at every write boundary
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def severity_from_meta(meta: Optional[Dict[str, Any]]) -> str:
    """Read the boundary-scan severity off an artifact's stored meta dict.

    Returns the severity string from `meta["security"]["injection_scan"]["severity"]`
    when present and well-formed, otherwise "none". Defensive about malformed
    or partial metadata — never raises.
    """
    if not isinstance(meta, dict):
        return "none"
    sec = meta.get("security")
    if not isinstance(sec, dict):
        return "none"
    scan = sec.get("injection_scan")
    if not isinstance(scan, dict):
        return "none"
    sev = scan.get("severity")
    if not isinstance(sev, str):
        return "none"
    sev_normalized = sev.strip().lower()
    if sev_normalized not in {"none", "low", "medium", "high"}:
        return "none"
    return sev_normalized


def max_severity(*severities: str) -> str:
    """Return the strictest severity among the inputs.

    Severity rank: none < low < medium < high.
    """
    best = "none"
    best_rank = _SEVERITY_RANK.get("none", 0)
    for s in severities:
        if not isinstance(s, str):
            continue
        s_norm = s.strip().lower()
        r = _SEVERITY_RANK.get(s_norm, 0)
        if r > best_rank:
            best = s_norm
            best_rank = r
    return best


def apply_scan(
    trust_score: float,
    meta: Dict[str, Any],
    result: InjectionScanResult,
    log_context: str = "",
) -> Tuple[float, Dict[str, Any]]:
    """Return (adjusted_trust_score, updated_meta) based on scan result.

    Does not mutate the input meta dict.
    """
    if result.severity == "none":
        return trust_score, meta

    logger.warning(
        "injection_scan severity=%s score=%.2f rules=%s context=%s",
        result.severity,
        result.score,
        result.matched_rules,
        log_context or "unknown",
    )

    if result.severity == "high":
        new_trust = 0.0
    elif result.severity == "medium":
        new_trust = float(trust_score) * 0.5
    else:  # low
        new_trust = float(trust_score) * 0.8

    new_meta = dict(meta)
    sec = dict(new_meta.get("security") or {})
    sec["injection_scan"] = result.to_dict()
    new_meta["security"] = sec
    return new_trust, new_meta


def scan_artifact_text(
    text: str,
    trust_score: float,
    meta: Dict[str, Any],
    *,
    log_context: str,
    now: "datetime | None" = None,
) -> Tuple[float, Dict[str, Any], "datetime | None"]:
    """Canonical write-time scan for any UMA artifact.

    Combines scan_content, apply_scan, and quarantine timestamp computation.
    Returns (adjusted_trust_score, updated_meta, quarantined_at).
    """
    from datetime import datetime as _dt, timezone as _tz

    result = scan_content(text or "")
    new_trust, new_meta = apply_scan(trust_score, meta, result, log_context=log_context)
    quarantined_at = None
    if result.severity == "high" and quarantine_enabled():
        quarantined_at = now if now is not None else _dt.now(_tz.utc)
    return new_trust, new_meta, quarantined_at
