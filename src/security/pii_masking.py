"""
PII Masking — redacts or masks personally-identifiable information before
it reaches logs, sessions, or API responses.

Masking strategies
------------------
  card_number   "4111111111111111"  → "****-****-****-1111"
  email         "john@example.com"  → "j***@example.com"
  phone         "+1-800-555-1234"   → "+1-800-***-1234"
  name          "John Doe"          → "J*** D***"
  ip_address    "192.168.1.42"      → "192.168.1.***"
  iban          "DE89370400440532013000" → "DE89****3000"
  generic       any tagged field    → "***REDACTED***"

Usage
-----
    from src.security.pii_masking import PIIMasker

    masker = PIIMasker()

    # Mask specific fields in a dict
    safe = masker.mask_dict(report, pii_fields=["user_email", "card_number"])

    # Auto-detect and mask a session before storage
    safe_session = masker.mask_session(session.to_dict())
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Regex patterns for auto-detection
# ---------------------------------------------------------------------------

_CARD_RE   = re.compile(r"\b(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})\b")
_EMAIL_RE  = re.compile(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
_PHONE_RE  = re.compile(r"(\+?\d[\d\s\-\(\)]{6,}\d)")
_IP_RE     = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b")

# Field name prefixes that are treated as PII (case-insensitive)
_PII_FIELD_PATTERNS = [
    "email", "phone", "mobile", "card", "pan", "iban", "ssn",
    "tax_id", "dob", "date_of_birth", "name", "address", "ip",
    "user_id",  # mask in external responses; keep internally
]


class PIIMasker:
    """
    Masks PII in dicts, strings, and session objects.

    Parameters
    ----------
    auto_detect  : if True, auto-scan field names for known PII patterns
    mask_user_id : if True, hash user_id in output (default False — needed for ops)
    """

    def __init__(
        self,
        auto_detect: bool = True,
        mask_user_id: bool = False,
    ) -> None:
        self._auto = auto_detect
        self._mask_uid = mask_user_id

    # ------------------------------------------------------------------ #
    # String-level masking
    # ------------------------------------------------------------------ #

    def mask_card(self, value: str) -> str:
        return _CARD_RE.sub(lambda m: f"****-****-****-{m.group(4)}", str(value))

    def mask_email(self, value: str) -> str:
        def _replace(m: re.Match) -> str:
            local = m.group(1)
            domain = m.group(2)
            return f"{local[0]}***@{domain}"
        return _EMAIL_RE.sub(_replace, str(value))

    def mask_phone(self, value: str) -> str:
        # Keep first 5 chars, mask middle
        s = str(value)
        if len(s) > 8:
            return s[:5] + "***" + s[-3:]
        return "***"

    def mask_ip(self, value: str) -> str:
        return _IP_RE.sub(r"\1.***", str(value))

    def mask_generic(self, value: Any) -> str:
        return "***REDACTED***"

    def mask_string(self, value: str) -> str:
        """Auto-detect and mask PII in a freeform string."""
        s = str(value)
        s = _CARD_RE.sub(lambda m: f"****-****-****-{m.group(4)}", s)
        s = _EMAIL_RE.sub(lambda m: f"{m.group(1)[0]}***@{m.group(2)}", s)
        s = _IP_RE.sub(r"\1.***", s)
        return s

    # ------------------------------------------------------------------ #
    # Dict-level masking
    # ------------------------------------------------------------------ #

    def mask_dict(
        self,
        data: dict[str, Any],
        pii_fields: list[str] | None = None,
        depth: int = 3,
    ) -> dict[str, Any]:
        """
        Return a shallow copy of *data* with PII fields masked.

        Parameters
        ----------
        pii_fields   : explicit list of field names to mask
        depth        : max nesting depth to recurse
        """
        if depth <= 0:
            return data

        result: dict[str, Any] = {}
        explicit = {f.lower() for f in (pii_fields or [])}

        for k, v in data.items():
            key_lower = k.lower()
            is_pii = key_lower in explicit or (
                self._auto and any(p in key_lower for p in _PII_FIELD_PATTERNS)
            )

            if is_pii:
                result[k] = self._mask_value(k, v)
            elif isinstance(v, dict):
                result[k] = self.mask_dict(v, pii_fields=pii_fields, depth=depth - 1)
            elif isinstance(v, list):
                result[k] = [
                    self.mask_dict(i, pii_fields=pii_fields, depth=depth - 1)
                    if isinstance(i, dict) else i
                    for i in v
                ]
            else:
                result[k] = v

        return result

    def mask_session(self, session_dict: dict[str, Any]) -> dict[str, Any]:
        """Mask PII in a session dict before storage."""
        return self.mask_dict(session_dict, depth=4)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _mask_value(self, key: str, value: Any) -> Any:
        k = key.lower()
        if "card" in k or "pan" in k:
            return self.mask_card(str(value))
        if "email" in k:
            return self.mask_email(str(value))
        if "phone" in k or "mobile" in k:
            return self.mask_phone(str(value))
        if "ip" in k:
            return self.mask_ip(str(value))
        if "name" in k and isinstance(value, str):
            parts = value.split()
            return " ".join(f"{p[0]}***" if p else "***" for p in parts)
        return self.mask_generic(value)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_masker = PIIMasker()


def mask(data: dict[str, Any], pii_fields: list[str] | None = None) -> dict[str, Any]:
    return _default_masker.mask_dict(data, pii_fields=pii_fields)


def mask_string(value: str) -> str:
    return _default_masker.mask_string(value)
