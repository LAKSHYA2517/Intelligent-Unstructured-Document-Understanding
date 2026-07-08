"""Fast deterministic PII masking utilities.

The module intentionally uses only precompiled regular expressions and small
deterministic validators. It is suitable for a single FastAPI process where a
stable in-memory mapping is useful across requests handled by that process.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Match


AuditMapping = dict[str, str]


@dataclass
class PiiMasker:
    """Mask common PII with deterministic placeholders.

    Placeholder counters are scoped by PII type and by this masker instance:
    the first unique email becomes ``[EMAIL_1]``, the second unique email
    becomes ``[EMAIL_2]``, and later occurrences of the same normalized value
    reuse the same placeholder. A lock protects the mapping when used by a
    shared FastAPI singleton.
    """

    _value_to_token: dict[str, str] = field(default_factory=dict)
    _token_to_value: dict[str, str] = field(default_factory=dict)
    _counters: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    EMAIL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![A-Za-z0-9._%+\-])"
        r"[A-Za-z0-9._%+\-]{1,64}@"
        r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+"
        r"[A-Za-z]{2,63}"
        r"(?![A-Za-z0-9._%+\-])",
        re.ASCII,
    )

    PHONE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<!\w)"
        r"(?:"
        r"(?:\+|00)\d{1,3}[\s.\-]?)?"
        r"(?:\(?\d{2,5}\)?[\s.\-]?)?"
        r"\d{3,5}[\s.\-]?\d{4,5}"
        r"(?!\w)",
        re.ASCII,
    )

    INDIAN_MOBILE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<!\w)(?:\+91[\s.\-]?|91[\s.\-]?|0)?[6-9]\d{4}[\s.\-]?\d{5}(?!\w)",
        re.ASCII,
    )

    CREDIT_CARD_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)",
        re.ASCII,
    )

    PAN_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Za-z0-9])",
        re.ASCII,
    )

    AADHAAR_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<!\d)[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)",
        re.ASCII,
    )

    GSTIN_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![A-Za-z0-9])"
        r"[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]"
        r"(?![A-Za-z0-9])",
        re.ASCII,
    )

    def mask(self, text: str) -> tuple[str, AuditMapping]:
        """Return sanitized text and a placeholder-to-original audit mapping."""

        if not text:
            return text, {}

        sanitized = text
        sanitized = self._replace(sanitized, "EMAIL", self.EMAIL_RE, self._normalize_email)
        sanitized = self._replace(sanitized, "PAN", self.PAN_RE, self._normalize_compact_upper)
        sanitized = self._replace(sanitized, "GSTIN", self.GSTIN_RE, self._normalize_compact_upper)
        sanitized = self._replace(
            sanitized,
            "CREDIT_CARD",
            self.CREDIT_CARD_RE,
            self._normalize_digits,
            self._is_valid_credit_card,
        )
        sanitized = self._replace(sanitized, "AADHAAR", self.AADHAAR_RE, self._normalize_digits, self._is_valid_aadhaar)
        sanitized = self._replace(
            sanitized,
            "PHONE",
            self.INDIAN_MOBILE_RE,
            self._normalize_phone,
            self._is_valid_phone,
        )
        sanitized = self._replace(
            sanitized,
            "PHONE",
            self.PHONE_RE,
            self._normalize_phone,
            self._is_valid_phone,
        )

        with self._lock:
            audit = {
                token: value
                for token, value in self._token_to_value.items()
                if token in sanitized
            }
        return sanitized, audit

    def _replace(
        self,
        text: str,
        pii_type: str,
        pattern: re.Pattern[str],
        normalizer: Callable[[str], str],
        validator: Callable[[str], bool] | None = None,
    ) -> str:
        def repl(match: Match[str]) -> str:
            raw_value = match.group(0)
            normalized = normalizer(raw_value)
            if not normalized:
                return raw_value
            if validator is not None and not validator(normalized):
                return raw_value
            return self._token_for(pii_type, normalized, raw_value)

        return pattern.sub(repl, text)

    def _token_for(self, pii_type: str, normalized_value: str, display_value: str) -> str:
        key = f"{pii_type}:{normalized_value}"
        with self._lock:
            token = self._value_to_token.get(key)
            if token is not None:
                return token

            next_index = self._counters.get(pii_type, 0) + 1
            self._counters[pii_type] = next_index
            token = f"[{pii_type}_{next_index}]"
            self._value_to_token[key] = token
            self._token_to_value[token] = display_value
            return token

    @staticmethod
    def _normalize_email(value: str) -> str:
        return value.strip().lower()

    @staticmethod
    def _normalize_compact_upper(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", value).upper()

    @staticmethod
    def _normalize_digits(value: str) -> str:
        return re.sub(r"\D", "", value)

    @staticmethod
    def _normalize_phone(value: str) -> str:
        return re.sub(r"[^\d+]", "", value)

    @staticmethod
    def _is_valid_credit_card(digits: str) -> bool:
        if not 13 <= len(digits) <= 19:
            return False
        if len(set(digits)) == 1:
            return False

        total = 0
        double = False
        for char in reversed(digits):
            value = ord(char) - ord("0")
            if double:
                value *= 2
                if value > 9:
                    value -= 9
            total += value
            double = not double
        return total % 10 == 0

    @staticmethod
    def _is_valid_aadhaar(digits: str) -> bool:
        return len(digits) == 12 and digits[0] in "23456789" and len(set(digits)) > 1

    @staticmethod
    def _is_valid_phone(value: str) -> bool:
        digits = re.sub(r"\D", "", value)
        if not 10 <= len(digits) <= 15:
            return False
        if len(set(digits)) == 1:
            return False
        if len(digits) == 10 and digits[0] in "012345":
            return False
        return True
