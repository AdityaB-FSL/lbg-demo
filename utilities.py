"""Shared utility helpers for text cleanup, enrichment, and duplicate matching."""

from __future__ import annotations

########## imports ##########
import re
from collections import Counter
from typing import Any

import pandas as pd


########## fuzzy similarity (Levenshtein / Jaro–Winkler) ##########


def token_sort_ratio(a: str, b: str) -> int:
    """
    Backward-compatible fuzzy score (0-100).

    Uses python-Levenshtein directly when available, with a pure-Python
    fallback otherwise.
    """
    s1 = scrub_whitespace(a or "").lower()
    s2 = scrub_whitespace(b or "").lower()
    if not s1 and not s2:
        return 100
    if not s1 or not s2:
        return 0

    try:
        import Levenshtein

        lev = float(Levenshtein.ratio(s1, s2))
        try:
            jw = float(Levenshtein.jaro_winkler(s1, s2))
        except AttributeError:
            jw = lev
        return int(round(((jw + lev) / 2.0) * 100))
    except ImportError:
        jw = _jaro_winkler_similarity(s1, s2)
        lev = _normalized_levenshtein_similarity(s1, s2)
        return int(round(((jw + lev) / 2.0) * 100))


def _normalized_levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1] via two-row dynamic programming."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    # Two-row DP Levenshtein for low memory usage.
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = cur
    distance = prev[lb]
    return 1.0 - (distance / max(la, lb))


def _jaro_winkler_similarity(a: str, b: str, prefix_scale: float = 0.1) -> float:
    """Jaro–Winkler similarity in [0, 1]; used when ``python-Levenshtein`` is unavailable."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0

    match_distance = max(la, lb) // 2 - 1
    a_matches = [False] * la
    b_matches = [False] * lb

    matches = 0
    for i in range(la):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, lb)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    transpositions /= 2.0
    m = float(matches)
    jaro = (m / la + m / lb + (m - transpositions) / m) / 3.0

    prefix = 0
    for ca, cb in zip(a[:4], b[:4]):
        if ca != cb:
            break
        prefix += 1
    return jaro + prefix * prefix_scale * (1.0 - jaro)


########## reference data (postal ZIP → city/state, US states) ##########

POSTAL_MASTER: dict[str, tuple[str, str]] = {
    "28403": ("Wilmington", "NC"),
    "78701": ("Austin", "TX"),
    "98101": ("Seattle", "WA"),
    "33101": ("Miami", "FL"),
    "33132": ("Miami", "FL"),
    "60601": ("Chicago", "IL"),
    "62704": ("Springfield", "IL"),
}

US_STATE_ABBR = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", 
                "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", 
                "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"}


def lookup_city_state(zip5: str) -> tuple[str, str] | None:
    """Map a 5-digit US ZIP to default (city, state) from :data:`POSTAL_MASTER`, or ``None``."""
    z = (zip5 or "").strip()
    if len(z) != 5 or not z.isdigit():
        return None
    return POSTAL_MASTER.get(z)


########## string normalization & formatting ##########


def scrub_whitespace(s: str) -> str:
    """Collapse internal whitespace and strip ends."""
    return re.sub(r"\s+", " ", (s or "").strip())


def title_case_string(s: str) -> str:
    """Title-case tokens; preserve ``&`` and 2-letter US state abbreviations when applicable."""
    s = scrub_whitespace(s)
    if not s:
        return s
    parts: list[str] = []
    for w in s.split():
        if w == "&":
            parts.append("&")
        elif len(w) == 2 and w.isalpha() and w.upper() == w and w.upper() in US_STATE_ABBR:
            parts.append(w.upper())
        else:
            parts.append(w[:1].upper() + w[1:].lower() if len(w) > 1 else w.upper())
    return " ".join(parts)


def normalize_email_case(raw: str) -> str:
    """Lowercase local and domain parts of an email; fix common ``@@`` typo."""
    e = scrub_whitespace(raw).replace("@@", "@")
    if not e or "@" not in e:
        return e
    local, _, domain = e.partition("@")
    return f"{local.lower()}@{domain.lower()}"


def digits_only(s: str) -> str:
    """Keep only ASCII digits from *s*."""
    return re.sub(r"\D", "", s or "")


def pad_zip_base5(zip_raw: str) -> str:
    """Extract digits and return a 5-digit base ZIP (zero-padded) or first 5 if longer."""
    d = digits_only(zip_raw or "")
    if not d:
        return ""
    if len(d) <= 5:
        return d.zfill(5)
    return d[:5]


def format_zip_us(zip_raw: str) -> str:
    """Format as US ``#####`` or ``#####-####`` when 9+ digits are present."""
    d = digits_only(zip_raw or "")
    if not d:
        return scrub_whitespace(zip_raw)
    base = d[:5].zfill(5) if len(d) >= 5 else d.zfill(5)
    if len(d) >= 9:
        return f"{base}-{d[5:9]}"
    return base


def format_phone_us_masked(raw: str) -> str:
    """Format as ``(###) ###-####``; strip leading 1; optional ``ext`` from trailing ``#``."""
    original = scrub_whitespace(raw)
    if not original:
        return original

    ext = ""
    m = re.search(r"\s*[#]\s*(\d+)\s*$", original)
    body = original
    if m:
        ext = m.group(1)
        body = original[: m.start()].strip()

    d = digits_only(body)
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) < 10:
        return original
    if len(d) > 10:
        d = d[:10]

    fmt = f"({d[:3]}) {d[3:6]}-{d[6:]}"
    return f"{fmt} ext {ext}" if ext else fmt


def apply_string_rules_to_text_fields(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Apply email/zip/phone/title-case rules to selected keys; treat pandas NA as empty."""
    out = dict(row)
    for k in keys:
        if k not in out:
            continue
        v = out[k]
        if v is None:
            continue
        try:
            if pd.isna(v):
                out[k] = ""
                continue
        except Exception:
            pass

        s = str(v)
        if k == "email":
            out[k] = normalize_email_case(s)
        elif k == "zip":
            out[k] = format_zip_us(s)
        elif k == "phone":
            out[k] = format_phone_us_masked(s)
        elif k == "state":
            st = scrub_whitespace(s)
            out[k] = st.upper() if len(st) == 2 and st.isalpha() else title_case_string(st)
        else:
            out[k] = title_case_string(scrub_whitespace(s))
    return out


########## cross-row enrichment (clusters, aliases) ##########


def _norm_email(s: Any) -> str:
    """Lowercased trimmed email string, or empty if no ``@``."""
    t = scrub_whitespace(str(s or "")).lower()
    return t if "@" in t else ""


def _phone10(s: Any) -> str:
    """Last 10 digits of a US phone (strip leading country code 1)."""
    d = digits_only(str(s or ""))
    if len(d) >= 11 and d.startswith("1"):
        d = d[1:]
    return d[:10] if len(d) >= 10 else ""


def _cluster_id(df: pd.DataFrame) -> list[str]:
    """Per-row cluster key: email, else phone10, else ``row:<index>``."""
    out: list[str] = []
    for i in range(len(df)):
        e = _norm_email(df.iloc[i].get("email", ""))
        p = _phone10(df.iloc[i].get("phone", ""))
        out.append(f"e:{e}" if e else f"p:{p}" if p else f"row:{i}")
    return out


def _mode_company(vals: list[str]) -> str:
    """Most common non-empty company value, excluding job-title-like strings."""
    cleaned = [scrub_whitespace(v) for v in vals if scrub_whitespace(v) and not _is_job_title(v)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def _is_job_title(company: str) -> bool:
    """Delegate to :func:`agents.is_job_title` (lazy import avoids circular imports)."""
    # Local import avoids circular import (agents ↔ utilities).
    from agents import is_job_title

    return is_job_title(company)


def fix_known_typos_global(df: pd.DataFrame) -> pd.DataFrame:
    """Apply global typo fixes (e.g. ``Sulne`` → ``Suite``) on address columns."""
    out = df.copy()
    for col in ("address_1", "address_2"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.replace(r"(?i)\bSulne\b", "Suite", regex=True)
    return out


def infer_company_from_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing or job-title ``company`` cells from modal company within email/phone clusters."""
    out = df.copy()
    if "company" not in out.columns:
        out["company"] = ""
    cid = _cluster_id(out)
    clusters: dict[str, list[int]] = {}
    for i, c in enumerate(cid):
        clusters.setdefault(c, []).append(i)
    col = out.columns.get_loc("company")
    for idx in clusters.values():
        mode = _mode_company([str(out.iloc[i].get("company", "") or "") for i in idx])
        if not mode:
            continue
        for i in idx:
            cur = scrub_whitespace(str(out.iloc[i].get("company", "") or ""))
            if not cur or _is_job_title(cur):
                out.iloc[i, col] = mode
    return out


def suggest_alias_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``dq_flags`` note when same phone has fuzzy-different names (possible alias)."""
    out = df.copy()
    if "dq_flags" not in out.columns:
        out["dq_flags"] = ""
    phones = [_phone10(out.iloc[i].get("phone", "")) for i in range(len(out))]
    names = [scrub_whitespace(str(out.iloc[i].get("name", "") or "")) for i in range(len(out))]
    fcol = out.columns.get_loc("dq_flags")
    for i in range(len(out)):
        if not phones[i]:
            continue
        for j in range(i + 1, len(out)):
            if phones[i] != phones[j]:
                continue
            ratio = token_sort_ratio(names[i], names[j]) / 100.0
            if ratio >= 0.65 and names[i].lower() != names[j].lower():
                note = "possible_nickname_alias_same_phone"
                for idx in (i, j):
                    cur = scrub_whitespace(str(out.iloc[idx].get("dq_flags", "") or ""))
                    if note not in cur:
                        out.iloc[idx, fcol] = f"{cur}; {note}".strip("; ") if cur else note
    return out


def enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run typo fixes then company inference from clusters."""
    out = fix_known_typos_global(df)
    return infer_company_from_clusters(out)


########## dataframe / row field helpers ##########


def field_str(row: dict[str, Any], key: str, default: str = "") -> str:
    """Scalar string from *row*[*key*]; empty for None, pandas NA, or literal ``nan``."""
    v = row.get(key, default)
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def normalize_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Copy *row* with known CRM keys cleaned via :func:`field_str`."""
    keys = [
        "customer_id",
        "name",
        "company",
        "address_1",
        "address_2",
        "city",
        "state",
        "zip",
        "email",
        "phone",
        "product_id",
        "product_name",
        "account_number",
    ]
    out = dict(row)
    for k in keys:
        if k in out:
            out[k] = field_str(out, k)
    return out


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip, lower-case, and space-to-underscore column names."""
    mapping = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    return df.rename(columns=mapping)


########## duplicate detection (blocking + fuzzy + tie-break) ##########


def _last_initial(name: str) -> str:
    """Uppercase first letter of the last whitespace-separated token, or ``?`` if missing."""
    parts = scrub_whitespace(name).split()
    return parts[-1][0].upper() if parts and parts[-1] else "?"


def blocking_key(row: dict[str, Any]) -> str:
    """Dedupe block: first 3 ZIP digits + last-name initial (``z3:L``)."""
    d = digits_only(str(row.get("zip", "") or ""))
    z3 = d[:5].zfill(5)[:3] if d else "000"
    return f"{z3}:{_last_initial(str(row.get('name', '') or ''))}"


def combined_fuzzy_score(name1: str, addr1: str, name2: str, addr2: str) -> float:
    """Mean of fuzzy name similarity and fuzzy address similarity (0–1)."""
    n = token_sort_ratio(name1 or "", name2 or "") / 100.0
    a = token_sort_ratio(addr1 or "", addr2 or "") / 100.0
    return round((n + a) / 2.0, 4)


def tie_breaker(r1: dict[str, Any], r2: dict[str, Any], fuzzy_score: float) -> dict[str, Any]:
    """When *fuzzy_score* is high, add verdict using normalized email/phone agreement."""
    e1 = normalize_email_case(str(r1.get("email", "") or ""))
    e2 = normalize_email_case(str(r2.get("email", "") or ""))
    p1 = digits_only(str(r1.get("phone", "") or ""))
    p2 = digits_only(str(r2.get("phone", "") or ""))
    if len(p1) >= 11 and p1.startswith("1"):
        p1 = p1[1:]
    if len(p2) >= 11 and p2.startswith("1"):
        p2 = p2[1:]
    p1, p2 = p1[:10], p2[:10]

    same_phone = bool(p1) and p1 == p2
    same_email = bool(e1) and e1 == e2
    verdict = "different_or_unclear"
    reason = ""
    prompt = ""
    if fuzzy_score >= 0.75:
        if same_email or same_phone:
            verdict = "same_person_likely"
            reason = "Shared normalized email or phone with high similarity."
        else:
            verdict = "needs_agent_review"
            reason = "High fuzzy match; identity not confirmed by email/phone."
            prompt = f"Records may refer to same person (score {fuzzy_score:.0%}). Are they same person?"
    return {
        "verdict": verdict,
        "reason": reason,
        "same_phone": same_phone,
        "same_email": same_email,
        "agent_prompt": prompt,
    }


def find_duplicate_candidates(
    df: pd.DataFrame, min_score: float = 0.55, high_band: float = 0.75
) -> pd.DataFrame:
    """Pairwise candidates sharing :func:`blocking_key` with fuzzy score ≥ *min_score*."""
    rows: list[dict[str, Any]] = []
    keys = [blocking_key(df.iloc[i].to_dict()) for i in range(len(df))]
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if keys[i] != keys[j]:
                continue
            r1, r2 = df.iloc[i].to_dict(), df.iloc[j].to_dict()
            score = combined_fuzzy_score(
                scrub_whitespace(str(r1.get("name", "") or "")),
                scrub_whitespace(str(r1.get("address_1", "") or "")),
                scrub_whitespace(str(r2.get("name", "") or "")),
                scrub_whitespace(str(r2.get("address_1", "") or "")),
            )
            if score < min_score:
                continue
            if score >= high_band:
                tb = tie_breaker(r1, r2, score)
            else:
                tb = {
                    "verdict": "",
                    "reason": "",
                    "same_phone": False,
                    "same_email": False,
                    "agent_prompt": "",
                }
            rows.append(
                {
                    "row_i": i,
                    "row_j": j,
                    "id_i": r1.get("customer_id", i),
                    "id_j": r2.get("customer_id", j),
                    "blocking_key": keys[i],
                    "fuzzy_score": score,
                    **tb,
                }
            )
    return pd.DataFrame(rows)
