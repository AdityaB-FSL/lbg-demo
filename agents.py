"""Entity, geo, and name agents for row-level CRM field quality."""

from __future__ import annotations

import re
from typing import Any

from utilities import (
    US_STATE_ABBR,
    digits_only,
    field_str,
    format_zip_us,
    lookup_city_state,
    pad_zip_base5,
    scrub_whitespace,
)

########## entity agent — patterns & rules ##########

_ABBREV_EXPAND = [(re.compile(r"^pac$", re.I), "Palmer Air Charters")]

_JOB_TITLE_EXACT = {
    "operations director", "owner", "manager", "ceo", "cfo", "cto", "president",
    "vice president", "director", "supervisor", "coordinator", "partner", "founder",
    "executive", "solar director", "chief executive officer",
}

_STREETISH = re.compile(r"(?i)\b(\d{1,6}\s+.+\s+(st|street|ave|avenue|dr|drive|rd|road|blvd|boulevard))\b")


def is_job_title(company: str) -> bool:
    """Return True if *company* text looks like a job title rather than an organization name."""
    c = scrub_whitespace(company).lower()
    if not c:
        return False
    if c in _JOB_TITLE_EXACT:
        return True
    if re.search(r"(?i)\b(director|manager|supervisor|coordinator|executive)\b", c) and len(c.split()) <= 4:
        return True
    if re.match(r"(?i)^(chief|vp|vice)\b", c):
        return True
    return False


def looks_like_street_address(text: str) -> bool:
    """Heuristic: street-like pattern or PO box in *text*."""
    t = scrub_whitespace(text)
    return bool(_STREETISH.search(t)) or bool(re.search(r"(?i)\b(po box|p\.o\.)\s*\d+", t))


def run_entity_agent(row: dict[str, Any]) -> dict[str, Any]:
    """Score and optionally fix ``company`` vs ``job_title``; return issues and flags."""
    company = field_str(row, "company")
    name = field_str(row, "name")

    flags: list[str] = []
    if re.search(r"\d", company):
        flags.append("company_contains_digits")
    if looks_like_street_address(company):
        flags.append("company_looks_like_address")

    improved_fields: dict[str, str] = {}
    if company and is_job_title(company):
        improved_fields["job_title"] = " ".join(w.capitalize() for w in company.split())
        improved_fields["company"] = ""
    else:
        norm = re.sub(r"\s+", " ", company).strip()
        for rx, expanded in _ABBREV_EXPAND:
            if rx.match(norm):
                norm = expanded
        norm = " ".join(w.capitalize() if w != "&" else w for w in norm.split())
        if norm and norm != company:
            improved_fields["company"] = norm

    issues: list[str] = []
    score = 100
    if not company:
        issues.append("Company is empty")
        score -= 25
    if company and is_job_title(company):
        issues.append("Company field appears to be a job title rather than organization name")
        score -= 15
    if company and name and len(company.split()) <= 3 and any(p.lower() in company.lower() for p in name.split() if len(p) > 2):
        issues.append("Company may contain a person name")
        score -= 10
    if "company_contains_digits" in flags:
        issues.append("Company field contains digits (verify legal entity name)")
        score -= 5
    if "company_looks_like_address" in flags:
        issues.append("Company field resembles a street address")
        score -= 5

    return {
        "agent": "entity",
        "score": max(0, score),
        "issues": issues,
        "improved_fields": improved_fields,
        "flags": flags,
        "job_title_detected": "job_title" in improved_fields,
    }


########## geo agent ##########

STATE_FULL_TO_ABBR = {
    "north carolina": "NC",
    "illinois": "IL",
    "texas": "TX",
    "washington": "WA",
    "florida": "FL",
    "georgia": "GA",
    "california": "CA",
    "new york": "NY",
}


def _zip_ok(z: str, country_hint: str) -> tuple[bool, str | None]:
    """Return (ok, error_message) for ZIP format given *country_hint* (US vs other)."""
    z = (z or "").strip()
    if not z:
        return False, "ZIP/postal code is empty"
    if country_hint == "US" or re.match(r"^\d{5}(-\d{4})?$", z):
        if re.match(r"^\d{5}(-\d{4})?$", z):
            return True, None
        return False, "US-style ZIP expected (5 digits or 5+4)"
    if len(z) < 3:
        return False, "Postal code seems too short"
    return True, None


def _improve_state(state: str, country_hint: str) -> str:
    """Normalize state: full name to US abbr when US; uppercase 2-letter codes."""
    s = (state or "").strip()
    if not s:
        return s
    if country_hint == "US" and len(s) > 2:
        return STATE_FULL_TO_ABBR.get(s.lower(), s)
    if len(s) == 2:
        return s.upper()
    return s


def _split_suite_from_address(addr1: str, addr2: str) -> tuple[str, str]:
    """Move trailing suite/unit from *addr1* into *addr2* when *addr2* is empty."""
    a1, a2 = (addr1 or "").strip(), (addr2 or "").strip()
    if a2 or not a1:
        return a1, a2
    m = re.search(r"\b((?:suite|ste|unit)\s*[#]?\s*[\w-]+)\s*$", a1, re.I)
    if m:
        unit = m.group(1).strip()
        rest = a1[: m.start()].rstrip(" ,.-")
        if rest:
            return rest, unit
    return a1, a2


def _title_city(city: str) -> str:
    """Title-case city string; leave 2-letter all-caps tokens unchanged."""
    c = re.sub(r"\s+", " ", (city or "").strip())
    if not c:
        return c
    if len(c) == 2 and c.isalpha() and c.upper() == c:
        return c
    return " ".join(w.capitalize() for w in c.split())


def _resolve_zip5(state: str, zip_code: str) -> str:
    """Derive 5-digit ZIP base from state/zip fields (handles misplaced digits in state)."""
    st = field_str({"state": state}, "state")
    zf = field_str({"zip": zip_code}, "zip")
    if re.match(r"^\d{5}$", st):
        return st
    return pad_zip_base5(zf)


def _format_zip_output(zip5: str, zip_original: str) -> str:
    """Format ZIP for display: ``#####-####`` if 9+ digits match *zip5* prefix."""
    d_all = digits_only(zip_original)
    if len(d_all) >= 9 and d_all[:5] == zip5:
        return f"{zip5}-{d_all[5:9]}"
    return format_zip_us(zip5)


def run_geo_agent(row: dict[str, Any]) -> dict[str, Any]:
    """Validate and improve location fields; optional master postal override by ZIP."""
    city = field_str(row, "city")
    state = field_str(row, "state")
    zip_code = field_str(row, "zip")
    addr1 = field_str(row, "address_1")
    addr2 = field_str(row, "address_2")

    zip5 = _resolve_zip5(state, zip_code)
    postal = lookup_city_state(zip5) if zip5 else None
    us_like = state.upper() in US_STATE_ABBR or state.lower() in STATE_FULL_TO_ABBR or bool(postal)
    country_hint = "US" if us_like else "OTHER"

    issues: list[str] = []
    score = 100
    if not city:
        issues.append("City is empty")
        score -= 20
    if not state:
        issues.append("State/region is empty")
        score -= 20
    ok, zip_msg = _zip_ok(zip_code, country_hint)
    if not ok and zip_msg:
        issues.append(zip_msg)
        score -= 20

    improved_fields: dict[str, str] = {}
    zip_out = _format_zip_output(zip5, zip_code)
    if zip_out and zip_out != zip_code:
        improved_fields["zip"] = zip_out

    if postal:
        pcity, pstate = postal
        improved_fields["city"] = pcity
        improved_fields["state"] = pstate
        improved_fields["zip"] = _format_zip_output(zip5, zip_code)
    else:
        new_city = _title_city(city)
        if new_city != city:
            improved_fields["city"] = new_city
        new_state = _improve_state(state, country_hint)
        if new_state != state:
            improved_fields["state"] = new_state

    na1, na2 = _split_suite_from_address(addr1, addr2)
    if (na1, na2) != (addr1, addr2):
        improved_fields["address_1"] = na1
        if na2:
            improved_fields["address_2"] = na2

    return {
        "agent": "geo",
        "score": max(0, score),
        "issues": issues,
        "country_hint": country_hint,
        "improved_fields": improved_fields,
        "postal_lookup_applied": bool(postal),
    }


########## name agent ##########


def _score_name(name: str) -> tuple[int, list[str]]:
    """Return (score 0–100, issue strings) for person-name quality heuristics."""
    issues: list[str] = []
    score = 100
    n = (name or "").strip()
    if not n:
        return 0, ["Name is empty"]
    if len(n) < 2:
        issues.append("Name is unusually short")
        score -= 30
    if re.search(r"\s{2,}", n):
        issues.append("Extra internal whitespace")
        score -= 5
    if re.search(r"[@#]", n):
        issues.append("Name contains @ or # (possible misplaced email/phone)")
        score -= 25
    titles = ("mr.", "mrs.", "ms.", "dr.", "prof.")
    lower = n.lower()
    if any(lower.startswith(t + " ") for t in titles):
        issues.append("Title prefix present — consider normalizing for matching")
        score -= 3
    parts = n.split()
    if len(parts) >= 4:
        issues.append("Long multi-part name — verify against other records for duplicates")
        score -= 5
    return max(0, score), issues


def _improve_name(raw: str) -> str:
    """Strip honorifics, collapse whitespace, title-case tokens."""
    n = re.sub(r"\s+", " ", (raw or "").strip())
    if not n:
        return n
    lower = n.lower()
    for t in ("mr. ", "mrs. ", "ms. ", "dr. ", "prof. "):
        if lower.startswith(t):
            n = n[len(t) :].lstrip()
            lower = n.lower()
            break
    return " ".join(w.capitalize() for w in n.split())


def run_name_agent(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``name`` and return score plus optional improvement patch."""
    name = field_str(row, "name")
    score, issues = _score_name(name)
    improved = _improve_name(name)
    improved_fields: dict[str, str] = {}
    if improved != name:
        improved_fields["name"] = improved

    return {
        "agent": "name",
        "score": score,
        "issues": issues,
        "improved_fields": improved_fields,
    }
