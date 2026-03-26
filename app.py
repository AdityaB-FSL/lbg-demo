"""Streamlit client: loads ``data.csv`` from the project folder; flags, clean/unify/dedupe, optional LLM review."""

########## imports ##########
import html as html_lib
import json
import os
import re
from io import StringIO
from pathlib import Path
from typing import Any

import markdown as md_lib
import pandas as pd
import streamlit as st

from agents import run_entity_agent
from agents import run_geo_agent
from agents import run_name_agent
from utilities import (
    apply_string_rules_to_text_fields,
    digits_only,
    enrich_dataset,
    find_duplicate_candidates,
    scrub_whitespace,
    suggest_alias_notes,
)


########## constants ##########

_DATA_CSV_PATH = Path(__file__).resolve().parent / "data.csv"
_THEME_CSS_PATH = Path(__file__).resolve().parent / "theme.css"

TEXT_KEYS = ["name", "company", "address_1", "address_2", "city", "state", "zip", "email", "phone", "product_name"]

# Default text shown in the AI review rules box (also set in init_state on load).
_DEFAULT_AI_REVIEW_RULES = """1. For name, use clean name without any prefix

2. For Address, use the latest updated address

3. For email, it should always be picked from CRM data.

4. For customer_id, use the core banking data id for merging."""

_ADDRESS_MERGE_COLS = ("address_1", "address_2", "city", "state", "zip")


def _data_file_token() -> str:
    """Change when ``data.csv`` is replaced or edited (reload session)."""
    p = _DATA_CSV_PATH
    if not p.is_file():
        return "missing"
    meta = p.stat()
    return f"{meta.st_mtime_ns}:{meta.st_size}"


def load_default_dataset() -> pd.DataFrame:
    """Read :data:`_DATA_CSV_PATH` as strings with empty NA."""
    return pd.read_csv(_DATA_CSV_PATH, dtype=str).fillna("")

########## UI theme (client-facing) ##########


def _load_theme_css() -> str:
    """Read ``theme.css`` next to ``app.py`` and wrap for ``st.markdown``."""
    if not _THEME_CSS_PATH.is_file():
        return f"<style>/* missing: {_THEME_CSS_PATH.name} */</style>"
    raw = _THEME_CSS_PATH.read_text(encoding="utf-8")
    return f"<style>\n{raw}\n</style>"


########## UI helpers ##########

def inject_theme() -> None:
    """Inject global CSS (fonts, hero, buttons, metrics) from ``theme.css``."""
    st.markdown(_load_theme_css(), unsafe_allow_html=True)


def _markdown_to_html_fragment(text: str) -> str:
    """Convert trusted markdown (bold, code, paragraphs) to HTML for caption blocks."""
    return md_lib.markdown(
        text.strip(),
        extensions=["nl2br"],
        output_format="html5",
    ).strip()


def hero(title: str, subtitle: str) -> None:
    """Render the gradient header card."""
    st.markdown(
        f'<div class="dq-hero">'
        f'<div class="dq-hero-eyebrow">✨ Data quality workspace</div>'
        f"<h2>{html_lib.escape(title)}</h2>"
        f'<div class="dq-hero-sub">{_markdown_to_html_fragment(subtitle)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def section_heading(title: str, *, first: bool = False) -> None:
    """Main section title (H3) with bottom rule."""
    cls = "dq-h3" + (" dq-section-gap" if not first else "")
    safe = html_lib.escape(title)
    st.markdown(f'<h3 class="{cls}">{safe}</h3>', unsafe_allow_html=True)


def subheading(title: str, *, tight: bool = False) -> None:
    """Subsection label (H4) under a main section — use before a control block or table."""
    cls = "dq-h4" + (" dq-h4--tight" if tight else "")
    safe = html_lib.escape(title)
    st.markdown(f'<h4 class="{cls}">{safe}</h4>', unsafe_allow_html=True)


def section_divider() -> None:
    """Subtle horizontal rule between major sections."""
    st.markdown('<hr class="dq-divider"/>', unsafe_allow_html=True)


def section_caption(text: str) -> None:
    """Muted helper text under a section heading (markdown: **bold**, `code`)."""
    inner = _markdown_to_html_fragment(text)
    st.markdown(f'<div class="dq-muted">{inner}</div>', unsafe_allow_html=True)


def widget_caption(text: str) -> None:
    """Short hint under a subheading (markdown supported)."""
    inner = _markdown_to_html_fragment(text)
    st.markdown(f'<div class="dq-caption">{inner}</div>', unsafe_allow_html=True)


########## session state ##########


def init_state(loaded: pd.DataFrame) -> None:
    """Initialize Streamlit session from the loaded dataframe."""
    st.session_state.df = loaded
    st.session_state.original_df = loaded.copy(deep=True)
    st.session_state.history = [("Initial load", loaded.copy(deep=True))]
    st.session_state.latest_report = {}
    st.session_state.matches = pd.DataFrame()
    st.session_state.dedupe_ran = False
    st.session_state.clean_unify_ran = False
    st.session_state.group_ai_results = {}
    st.session_state.df_before_ai_consolidation = None
    st.session_state.ai_consolidation_log = []
    st.session_state.ai_review_rules = _DEFAULT_AI_REVIEW_RULES
    st.session_state.last_ai_merge = None


def save_snapshot(action_name: str, df: pd.DataFrame, report: dict[str, Any] | None = None) -> None:
    """Persist current dataframe and optional profiling report to session."""
    st.session_state.df = df.copy(deep=True)
    st.session_state.history.append((action_name, st.session_state.df.copy(deep=True)))
    if report is not None:
        st.session_state.latest_report = report


def _append_flag(existing: str, new_flag: str) -> str:
    """Append *new_flag* to semicolon-separated flags without duplicates."""
    parts = [p.strip() for p in str(existing or "").split(";") if p.strip()]
    if new_flag not in parts:
        parts.append(new_flag)
    return "; ".join(parts)


def _contact_dq_issues(row: dict[str, Any]) -> list[str]:
    """Email / phone / transaction checks aligned with :func:`compute_quality_score`."""
    issues: list[str] = []
    email = str(row.get("email") or "").strip()
    if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        issues.append("Email format does not match expected pattern")

    phone = digits_only(str(row.get("phone") or ""))
    if phone and len(phone) < 10:
        issues.append("Phone number has fewer than 10 digits (US)")

    txn_raw = str(row.get("transaction_amount") or "").strip()
    if txn_raw:
        try:
            if float(txn_raw) < 0:
                issues.append("Negative transaction amount")
        except ValueError:
            issues.append("Transaction amount is not numeric")

    return issues


########## profiling & pipelines ##########


def profile_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Build a simple quality summary (counts, invalid email/phone/zip, etc.)."""
    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_by_col": {},
        "invalid_email_rows": 0,
        "invalid_phone_rows": 0,
        "invalid_us_zip_rows": 0,
        "negative_txn_rows": 0,
        "duplicate_customer_id_rows": 0,
    }

    for col in df.columns:
        report["missing_by_col"][col] = int((df[col].astype(str).str.strip() == "").sum())

    if "email" in df.columns:
        email_ok = df["email"].astype(str).str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)
        report["invalid_email_rows"] = int((~email_ok & (df["email"].astype(str).str.strip() != "")).sum())

    if "phone" in df.columns:
        phone_digits = df["phone"].astype(str).apply(digits_only)
        report["invalid_phone_rows"] = int(((phone_digits.str.len() < 10) & (phone_digits != "")).sum())

    if "zip" in df.columns:
        zip_ok = df["zip"].astype(str).str.match(r"^\d{5}(-\d{4})?$", na=False)
        report["invalid_us_zip_rows"] = int((~zip_ok & (df["zip"].astype(str).str.strip() != "")).sum())

    if "transaction_amount" in df.columns:
        txn = pd.to_numeric(df["transaction_amount"], errors="coerce")
        report["negative_txn_rows"] = int((txn < 0).sum())

    if "customer_id" in df.columns:
        dup_mask = df["customer_id"].astype(str).duplicated(keep=False) & (df["customer_id"].astype(str) != "")
        report["duplicate_customer_id_rows"] = int(dup_mask.sum())

    return report


def run_standardization(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply :func:`apply_string_rules_to_text_fields` to :data:`TEXT_KEYS` per row."""
    out = df.copy(deep=True)
    available_keys = [k for k in TEXT_KEYS if k in out.columns]
    touched_rows = 0

    for i in range(len(out)):
        before = out.iloc[i].to_dict()
        after = apply_string_rules_to_text_fields(before, available_keys)
        if after != before:
            touched_rows += 1
        for k, v in after.items():
            out.at[i, k] = v

    return out, {"rows_standardized": touched_rows}


def run_entity_geo_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per-row entity + geo agents, merge improvements, enrich, alias notes."""
    out = df.copy(deep=True)
    entity_changes = 0
    geo_changes = 0
    dq_flag_rows = 0

    if "dq_flags" not in out.columns:
        out["dq_flags"] = ""
    if "job_title" not in out.columns:
        out["job_title"] = ""

    for i in range(len(out)):
        row = out.iloc[i].to_dict()
        ent = run_entity_agent(row)
        geo = run_geo_agent(row)
        name = run_name_agent(row)

        for k, v in ent.get("improved_fields", {}).items():
            if k in out.columns and str(out.at[i, k]) != str(v):
                out.at[i, k] = v
                entity_changes += 1

        if "job_title" in ent.get("improved_fields", {}) and str(out.at[i, "job_title"]).strip() == "":
            out.at[i, "job_title"] = ent["improved_fields"]["job_title"]

        for k, v in geo.get("improved_fields", {}).items():
            if k in out.columns and str(out.at[i, k]) != str(v):
                out.at[i, k] = v
                geo_changes += 1

        combined = (
            ent.get("issues", [])
            + geo.get("issues", [])
            + name.get("issues", [])
            + _contact_dq_issues(row)
        )
        for issue in combined:
            out.at[i, "dq_flags"] = _append_flag(str(out.at[i, "dq_flags"]), issue)

    out = enrich_dataset(out)
    out = suggest_alias_notes(out)

    for i in range(len(out)):
        if str(out.at[i, "dq_flags"]).strip():
            dq_flag_rows += 1

    return out, {
        "entity_field_updates": entity_changes,
        "geo_field_updates": geo_changes,
        "rows_with_dq_flags": dq_flag_rows,
    }


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``dq_score`` column (heuristic 0–100 from flags and field checks)."""
    out = df.copy(deep=True)
    scores: list[int] = []

    for i in range(len(out)):
        score = 100
        flags = str(out.iloc[i].get("dq_flags", ""))
        flag_count = len([f for f in flags.split(";") if f.strip()])
        score -= flag_count * 7

        email = str(out.iloc[i].get("email", ""))
        if email and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            score -= 10

        phone = digits_only(str(out.iloc[i].get("phone", "")))
        if phone and len(phone) < 10:
            score -= 10

        txn_raw = str(out.iloc[i].get("transaction_amount", "")).strip()
        if txn_raw:
            try:
                if float(txn_raw) < 0:
                    score -= 10
            except ValueError:
                score -= 5

        scores.append(max(0, score))

    out["dq_score"] = scores
    return out


########## flags ##########


def generate_dq_flags_only(df: pd.DataFrame) -> pd.DataFrame:
    """Populate ``dq_flags`` from name, entity, geo, and contact checks (no enrichment)."""
    out = df.copy(deep=True)
    flags: list[str] = []
    for i in range(len(out)):
        row = out.iloc[i].to_dict()
        ent = run_entity_agent(row)
        geo = run_geo_agent(row)
        name = run_name_agent(row)
        issues = [
            str(x).strip()
            for x in (
                ent.get("issues", [])
                + geo.get("issues", [])
                + name.get("issues", [])
                + _contact_dq_issues(row)
            )
            if str(x).strip()
        ]
        deduped = list(dict.fromkeys(issues))
        flags.append("; ".join(deduped))
    out["dq_flags"] = flags
    return out


########## LLM (Gemini via LangChain) ##########

_LLM_RECORD_KEYS = [
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
]


def _record_snapshot_for_llm(df: pd.DataFrame, idx: int) -> dict[str, str]:
    """Extract comparable field dict for row *idx* (string values, empty if missing)."""
    if idx < 0 or idx >= len(df):
        return {}
    row = df.iloc[int(idx)]
    out: dict[str, str] = {}
    for k in _LLM_RECORD_KEYS:
        if k not in df.columns:
            continue
        v = row.get(k, "")
        try:
            if v is None or pd.isna(v):
                out[k] = ""
                continue
        except Exception:
            pass
        out[k] = str(v).strip()
    return out


def _as_bool_clean(val: Any) -> bool:
    """Coerce to bool; treat missing/NaN as False (avoids ``bool(nan)`` being True)."""
    try:
        if val is None or pd.isna(val):
            return False
    except Exception:
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "y")
    return bool(val)


def _build_llm_duplicate_payload(df: pd.DataFrame, ambiguous_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Attach ``record_a`` / ``record_b`` plus rule metadata so the model can judge identity."""
    out: list[dict[str, Any]] = []
    for _, pair in ambiguous_df.iterrows():
        d = pair.to_dict()
        ri = int(d.get("row_i", -1))
        rj = int(d.get("row_j", -1))
        fs = d.get("fuzzy_score")
        try:
            fs_f = float(fs) if fs is not None and str(fs) != "nan" else None
        except (TypeError, ValueError):
            fs_f = None
        out.append(
            {
                "row_i": ri,
                "row_j": rj,
                "fuzzy_score": fs_f,
                "blocking_key": d.get("blocking_key"),
                "same_email_normalized": _as_bool_clean(d.get("same_email")),
                "same_phone_normalized": _as_bool_clean(d.get("same_phone")),
                "rule_reason": str(d.get("reason") or ""),
                "record_a": _record_snapshot_for_llm(df, ri),
                "record_b": _record_snapshot_for_llm(df, rj),
            }
        )
    return out


def _same_person_truthy(val: Any) -> bool:
    """Parse model output for same-person yes/no (robust to wording)."""
    if val is True:
        return True
    if val is False:
        return False
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y", "same", "duplicate", "dup", "match"):
        return True
    if s in ("false", "0", "no", "n", "different", "not_duplicate", "not duplicate", "distinct"):
        return False
    return False


_LLM_DEDUPE_INSTRUCTIONS = """You are a data-steward assistant. For EACH object below, decide whether **record_a** and **record_b** describe the **same real-world person** (duplicate contact) or **two different people**.

Use the full fields in record_a and record_b (name, address, email, phone, company). Minor typos, nicknames, punctuation, line order, or formatting differences often still mean **same person**. Same city/ZIP region with very similar names often indicates a duplicate unless evidence suggests two distinct people.

The pair already has high name/address similarity (fuzzy_score) but email/phone may not match — mismatched contact info can still occur for the same person (old email, work vs personal phone).

**Downstream merge (for your context):** **Per-group AI review** merges **all** rows in that group into one record (forced combination); address columns come from the **latest** row. **Batch “all pairs”** merges clusters where you mark **same_person**; address mismatch does **not** block merging, and address fields still follow the **latest** row in each cluster.

Return ONLY a JSON array. Each element must be an object with exactly these keys:
- row_i (number, must match the input)
- row_j (number, must match the input)
- same_person (boolean true or false only)
- reason (one short sentence citing specific fields, e.g. "Same name and address; emails differ but phone matches")

Do not include markdown, code fences, or any text outside the JSON array."""


def _build_duplicate_review_system_prompt(extra_rules: str) -> str:
    """Append user-editable business rules to the base duplicate-review system prompt."""
    extra = (extra_rules or "").strip()
    if not extra:
        return _LLM_DEDUPE_INSTRUCTIONS
    return (
        f"{_LLM_DEDUPE_INSTRUCTIONS}\n\n"
        "**Additional business rules (must follow when deciding and reasoning):**\n\n"
        f"{extra}"
    )


class GeminiApiQuotaError(Exception):
    """Raised when Gemini hits quota, rate limits, or temporary overload (ResourceExhausted / 429)."""


def _iter_exception_chain(exc: BaseException | None) -> list[BaseException]:
    """Walk __cause__ / __context__ without infinite loops."""
    out: list[BaseException] = []
    seen: set[int] = set()

    def walk(e: BaseException | None) -> None:
        if e is None or id(e) in seen:
            return
        seen.add(id(e))
        out.append(e)
        walk(e.__cause__)
        ctx = getattr(e, "__context__", None)
        if ctx is not None and ctx is not e.__cause__:
            walk(ctx)

    walk(exc)
    return out


def _exception_is_quota_or_exhausted(exc: BaseException) -> bool:
    """Detect Google / HTTP quota, rate limit, and resource exhaustion."""
    try:
        from google.api_core import exceptions as gexc

        for e in _iter_exception_chain(exc):
            if isinstance(e, (gexc.ResourceExhausted, gexc.TooManyRequests)):
                return True
    except ImportError:
        pass

    try:
        import httpx

        for e in _iter_exception_chain(exc):
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                if e.response.status_code == 429:
                    return True
    except ImportError:
        pass

    for e in _iter_exception_chain(exc):
        name = type(e).__name__
        if "ResourceExhausted" in name or "TooManyRequests" in name:
            return True
        msg = f"{name} {e}".lower()
        if any(
            token in msg
            for token in (
                "resource exhausted",
                "resource_exhausted",
                "quota",
                "rate limit",
                "rate_limit",
                "429",
                "too many requests",
                "exceeded your current quota",
                "billing",
            )
        ):
            return True
    return False


def _friendly_gemini_quota_message(original: BaseException | None) -> str:
    """User-facing copy; optional short detail from the API for debugging."""
    detail = ""
    if original is not None:
        s = str(original).strip()
        if s and len(s) < 400:
            detail = f" Details: {s}"
    return (
        "The Gemini API reported a quota limit, rate limit, or temporary overload. "
        "Wait a few minutes, try fewer pairs in one request, or check usage and billing "
        "in Google AI Studio or Google Cloud Console."
        + detail
    )


def call_llm_agent(
    user_content: str,
    api_key: str,
    model: str,
    *,
    system_content: str = "You are a careful data quality assistant.",
) -> str:
    """Call Gemini via LangChain; return assistant message text."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.15,
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]
        )
    except Exception as e:
        if _exception_is_quota_or_exhausted(e):
            raise GeminiApiQuotaError(_friendly_gemini_quota_message(e)) from e
        raise
    return str(response.content)


def _render_llm_call_error(exc: BaseException) -> None:
    """Streamlit message for LLM failures; special handling for quota / exhaustion."""
    if isinstance(exc, GeminiApiQuotaError):
        st.error(str(exc))
        return
    st.error(f"LLM error: {exc}")


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Parse JSON array/object from model output; strip ```json``` fences if present."""
    text = (raw or "").strip()
    if not text:
        return []
    if "```" in text:
        chunks = [c.strip() for c in text.split("```") if c.strip()]
        text = chunks[1] if len(chunks) >= 2 else chunks[0]
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except Exception:
        return []

    if isinstance(parsed, dict):
        if isinstance(parsed.get("results"), list):
            return [x for x in parsed["results"] if isinstance(x, dict)]
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def _normalize_llm_duplicate_decisions(parsed: list[dict[str, Any]]) -> pd.DataFrame:
    """Build decisions dataframe with ``same_person`` as ``\"true\"`` / ``\"false\"`` strings."""
    decisions = pd.DataFrame(parsed)
    for req in ("row_i", "row_j", "same_person", "reason"):
        if req not in decisions.columns:
            decisions[req] = ""
    decisions = decisions[["row_i", "row_j", "same_person", "reason"]]
    sb = decisions["same_person"].map(_same_person_truthy)
    decisions["same_person"] = sb.map(lambda x: "true" if x else "false")
    return decisions


def _llm_duplicate_pair_decisions(
    df: pd.DataFrame,
    ambiguous_df: pd.DataFrame,
    api_key: str,
    model: str,
    *,
    review_rules: str = "",
) -> tuple[pd.DataFrame | None, str | None]:
    """Returns ``(decisions, None)`` on success, or ``(None, raw_response)`` if JSON parse fails."""
    payload = _build_llm_duplicate_payload(df, ambiguous_df)
    user_msg = (
        "Decide each pair below. Each item includes record_a, record_b, fuzzy_score, "
        "and rule-based same_email_normalized / same_phone_normalized flags.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    response = call_llm_agent(
        user_msg,
        api_key=api_key,
        model=model,
        system_content=_build_duplicate_review_system_prompt(review_rules),
    )
    parsed = _parse_llm_json(response)
    if not parsed:
        return None, response
    return _normalize_llm_duplicate_decisions(parsed), None


########## AI consolidation (merge rows) ##########


def _cell_at_pos(df: pd.DataFrame, pos: int, col: str) -> str:
    if pos < 0 or pos >= len(df):
        return ""
    v = df.iloc[pos].get(col, "")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _normalized_full_address_key(df: pd.DataFrame, pos: int) -> str:
    """Single comparable key for address_1, address_2, city, state, zip."""
    parts: list[str] = []
    for c in ("address_1", "address_2", "city", "state", "zip"):
        if c in df.columns:
            parts.append(scrub_whitespace(str(df.iloc[pos].get(c, "") or "")))
    return " | ".join(parts).lower()


def _format_full_address_line(df: pd.DataFrame, pos: int) -> str:
    """Human-readable full address from row *pos* (for ``final_address``)."""
    a1 = _cell_at_pos(df, pos, "address_1") if "address_1" in df.columns else ""
    a2 = _cell_at_pos(df, pos, "address_2") if "address_2" in df.columns else ""
    city = _cell_at_pos(df, pos, "city") if "city" in df.columns else ""
    state = _cell_at_pos(df, pos, "state") if "state" in df.columns else ""
    zipc = _cell_at_pos(df, pos, "zip") if "zip" in df.columns else ""
    line1 = ", ".join(x for x in (a1, a2) if x)
    line2_parts = [x for x in (city, state, zipc) if x]
    line2 = ", ".join(line2_parts) if line2_parts else ""
    if line1 and line2:
        return f"{line1}; {line2}"
    return line1 or line2


def _strip_name_prefix(name: str) -> str:
    """Remove common honorific prefixes for merged canonical names."""
    n = re.sub(r"\s+", " ", (name or "").strip())
    if not n:
        return n
    lower = n.lower()
    for t in ("mr. ", "mrs. ", "ms. ", "dr. ", "prof. ", "mr ", "mrs ", "ms ", "dr "):
        if lower.startswith(t):
            n = n[len(t) :].lstrip()
            break
    return " ".join(n.split())


def _crm_data_source_row(df: pd.DataFrame, comp: list[int]) -> int | None:
    """Row index whose ``data_source`` indicates CRM (for preferred email)."""
    if "data_source" not in df.columns:
        return None
    for p in sorted(comp):
        ds = _cell_at_pos(df, p, "data_source")
        if ds and re.search(r"\bCRM\b", ds, re.I):
            return p
    return None


def _core_banking_data_source_row(df: pd.DataFrame, comp: list[int]) -> int | None:
    """Row index whose ``data_source`` indicates core banking (for preferred ``customer_id``)."""
    if "data_source" not in df.columns:
        return None
    for p in sorted(comp):
        ds = _cell_at_pos(df, p, "data_source").lower()
        if "core banking" in ds:
            return p
    return None


def _union_find_components(nodes: list[int], edges: list[tuple[int, int]]) -> list[list[int]]:
    """Return connected components as lists of nodes (each sorted)."""
    if not nodes:
        return []
    parent: dict[int, int] = {n: n for n in nodes}
    rank: dict[int, int] = {n: 0 for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    buckets: dict[int, list[int]] = {}
    for n in nodes:
        buckets.setdefault(find(n), []).append(n)
    return [sorted(v) for v in buckets.values() if len(v) > 1]


def _merge_component_into_survivor(
    df: pd.DataFrame, comp: list[int], survivor: int, latest_pos: int
) -> tuple[dict[str, Any], list[str]]:
    """Build merged row dict at *survivor*; address from *latest_pos*; CRM/core-banking picks where applicable."""
    comp = sorted(comp)
    explanations: list[str] = []
    merged: dict[str, Any] = {}
    crm_row = _crm_data_source_row(df, comp)
    cb_row = _core_banking_data_source_row(df, comp)

    for col in df.columns:
        if col == "final_address":
            merged[col] = ""
            continue
        if col in _ADDRESS_MERGE_COLS:
            merged[col] = df.iloc[latest_pos][col]
            explanations.append(f"**{col}**: from **latest row {latest_pos}** (rule: latest address).")
            continue
        if col == "email" and crm_row is not None:
            merged[col] = df.iloc[crm_row][col]
            explanations.append(f"**{col}**: from **row {crm_row}** (`data_source` = CRM).")
            continue
        if col == "customer_id" and cb_row is not None:
            merged[col] = df.iloc[cb_row][col]
            explanations.append(f"**{col}**: from **row {cb_row}** (`data_source` = core banking).")
            continue
        if col == "dq_flags":
            parts: list[str] = []
            seen: set[str] = set()
            for p in comp:
                raw = _cell_at_pos(df, p, col)
                for seg in raw.split(";"):
                    s = seg.strip()
                    if s and s not in seen:
                        seen.add(s)
                        parts.append(s)
            val = "; ".join(parts)
            merged[col] = val
            if parts:
                explanations.append(
                    f"**{col}**: merged unique flags from rows {comp} ({len(parts)} segment(s))."
                )
            else:
                explanations.append(f"**{col}**: empty across merged rows.")
            continue

        if col == "dq_score":
            nums: list[float] = []
            for p in comp:
                s = _cell_at_pos(df, p, col)
                if not s:
                    continue
                try:
                    nums.append(float(s))
                except ValueError:
                    nums.append(float("nan"))
            nums = [x for x in nums if x == x]  # drop nan
            if nums:
                worst = min(nums)
                merged[col] = str(int(worst)) if worst == int(worst) else str(worst)
                explanations.append(
                    f"**{col}**: kept **worst** (min) score {merged[col]} across rows {comp}."
                )
            else:
                merged[col] = _cell_at_pos(df, survivor, col)
                explanations.append(f"**{col}**: no numeric values; left as canonical row {survivor}.")
            continue

        if col == "transaction_amount":
            total = 0.0
            any_num = False
            for p in comp:
                s = _cell_at_pos(df, p, col)
                if not s:
                    continue
                try:
                    total += float(s)
                    any_num = True
                except ValueError:
                    pass
            if any_num:
                merged[col] = str(int(total)) if total == int(total) else str(total)
                explanations.append(
                    f"**{col}**: **summed** values from rows {comp} → **{merged[col]}**."
                )
            else:
                sv = _cell_at_pos(df, survivor, col)
                merged[col] = sv
                explanations.append(f"**{col}**: no numeric values; kept from **row {survivor}**.")
            continue

        sv = _cell_at_pos(df, survivor, col)
        if sv:
            merged[col] = df.iloc[survivor][col]
            explanations.append(f"**{col}**: kept from **row {survivor}** (canonical survivor).")
            continue

        for p in comp:
            if p == survivor:
                continue
            ov = _cell_at_pos(df, p, col)
            if ov:
                merged[col] = df.iloc[p][col]
                explanations.append(f"**{col}**: taken from **row {p}** (canonical row was empty).")
                break
        else:
            merged[col] = df.iloc[survivor][col]
            explanations.append(f"**{col}**: empty across merged rows.")

    if "name" in merged:
        before = str(merged.get("name", "") or "")
        after = _strip_name_prefix(before)
        if after != before:
            merged["name"] = after
            explanations.append("**name**: honorific prefixes removed (clean name).")

    return merged, explanations


def consolidate_rows_from_ai_decisions(
    df: pd.DataFrame,
    group_row_indices: list[int],
    decisions_df: pd.DataFrame,
    *,
    force_entire_group: bool = False,
) -> tuple[pd.DataFrame, str, int, list[dict[str, Any]]]:
    """
    Merge rows in *group_row_indices* using same-person edges from *decisions_df*,
    or (when *force_entire_group*) merge **all** rows in the group into one record.

    Address mismatch no longer blocks a merge; address columns follow the **latest** row.

    Returns (new dataframe, markdown explanation, number of rows removed, preview chunks for UI).
    """
    if df.empty or not group_row_indices:
        return df, "", 0, []

    group_set = set(group_row_indices)
    components: list[list[int]]

    if force_entire_group:
        comp = sorted(group_row_indices)
        if len(comp) < 2:
            return df, "_Group has fewer than 2 rows — nothing to merge._", 0, []
        components = [comp]
        intro = (
            "**Merge mode:** **Forced combination** — every row in this group is merged into **one** survivor record "
            "(independent of pairwise same/different model output).\n\n"
            "**Canonical row:** smallest row index. **`final_address`:** formatted from the **latest** row (highest index).\n\n"
            "**Address fields:** taken from the **latest** row. "
            "**Email:** prefer `data_source` **CRM** when present. "
            "**customer_id:** prefer **Core Banking** when present. "
            "**Name:** honorific prefixes stripped. "
            "**Other fields:** survivor wins when non-empty; else first non-empty peer; "
            "`transaction_amount` **summed**; `dq_flags` **deduplicated**; `dq_score` **minimum** (worst).\n\n"
        )
    else:
        edges: list[tuple[int, int]] = []
        for _, r in decisions_df.iterrows():
            if not _same_person_truthy(r.get("same_person")):
                continue
            try:
                i, j = int(r["row_i"]), int(r["row_j"])
            except (TypeError, ValueError):
                continue
            if i in group_set and j in group_set:
                edges.append((i, j))

        if not edges:
            return df, "_No pairs marked as the same person — nothing to consolidate._", 0, []

        components = _union_find_components(list(group_set), edges)
        if not components:
            return df, "_No multi-row clusters to merge._", 0, []

        intro = (
            "**Merge rule:** clusters are built from pairs the model marked **same person**. "
            "**Address mismatch does not block** the merge; address columns use the **latest** row in the cluster.\n\n"
            "**Canonical row:** smallest row index. **`final_address`:** from the **latest** row.\n\n"
            "**Address fields:** from the **latest** row. "
            "**Email:** CRM when present. **customer_id:** Core Banking when present. "
            "**Name:** prefixes stripped. "
            "**Other fields:** standard merge rules; `transaction_amount` **summed**; "
            "`dq_flags` **deduplicated**; `dq_score` **minimum**.\n\n"
        )

    out = df.copy()
    if "final_address" not in out.columns:
        out["final_address"] = ""

    drop_mask = [False] * len(out)
    all_expl: list[str] = []
    removed = 0
    preview_chunks: list[dict[str, Any]] = []

    for comp in components:
        addr_keys = [_normalized_full_address_key(out, p) for p in comp]
        uniq_addr = set(addr_keys)
        n_distinct_addr = len(uniq_addr)

        surv = min(comp)
        latest_pos = max(comp)
        merged, expl_lines = _merge_component_into_survivor(out, comp, surv, latest_pos)
        merged["final_address"] = _format_full_address_line(out, latest_pos)
        if n_distinct_addr <= 1:
            expl_lines.append(
                f"**final_address**: formatted from **latest row {latest_pos}**; "
                "normalized address aligned across the cluster."
            )
        else:
            expl_lines.append(
                f"**final_address**: formatted from **latest row {latest_pos}**; "
                f"cluster had **{n_distinct_addr}** distinct normalized addresses — **latest row** kept (forced merge)."
            )
        merged_record = {c: merged.get(c, "") for c in out.columns}
        preview_chunks.append(
            {
                "status": "merged",
                "cluster": comp,
                "survivor_index": surv,
                "latest_row_index": latest_pos,
                "merged_record": merged_record,
                "field_explanations": list(expl_lines),
            }
        )
        for col in out.columns:
            out.iat[surv, out.columns.get_loc(col)] = merged[col]
        for p in comp:
            if p != surv:
                drop_mask[p] = True
                removed += 1
        all_expl.append(f"**Cluster {comp}** → survivor **row {surv}** (canonical); **latest** row index = {latest_pos}")
        all_expl.extend(expl_lines)

    out = out.loc[[not x for x in drop_mask]].reset_index(drop=True)
    md = intro + "\n\n".join(all_expl)
    return out, md, removed, preview_chunks


def _apply_group_consolidation_from_decisions(
    group_label: int | str,
    group_row_indices: list[int],
    dec: pd.DataFrame,
    *,
    force_entire_group: bool = False,
) -> tuple[int, str, list[dict[str, Any]]]:
    """
    Merge duplicate rows per AI decisions; update session state and dedupe matches.
    Removes cached per-group decisions for this group when *group_label* is int.
    """
    if st.session_state.df_before_ai_consolidation is None:
        st.session_state.df_before_ai_consolidation = st.session_state.df.copy(deep=True)
    new_df, md, removed, previews = consolidate_rows_from_ai_decisions(
        st.session_state.df,
        list(group_row_indices),
        dec,
        force_entire_group=force_entire_group,
    )
    st.session_state.df = new_df
    st.session_state.matches = find_duplicate_candidates(
        new_df, min_score=0.55, high_band=0.75
    )
    st.session_state.ai_consolidation_log.append(
        {
            "group_num": group_label,
            "explanation": md,
            "rows_removed": removed,
        }
    )
    if isinstance(group_label, int):
        st.session_state.group_ai_results.pop(group_label, None)
    return removed, md, previews


def _merged_record_to_dataframe(merged: dict[str, Any], column_order: list[str]) -> pd.DataFrame:
    """Single-row dataframe with stable column order for display."""
    row: dict[str, Any] = {c: merged.get(c, "") for c in column_order}
    for k, v in merged.items():
        if k not in row:
            row[k] = v
    return pd.DataFrame([row])


def _build_ai_insight_bullets(
    field_explanations: list[str],
    decisions: pd.DataFrame | None,
    *,
    max_bullets: int = 5,
) -> list[str]:
    """3–5 short bullets: prefer Gemini pairwise ``reason`` lines, then merge rule lines."""
    bullets: list[str] = []
    seen_lower: set[str] = set()

    def add_sentence(s: str) -> None:
        s = re.sub(r"\s+", " ", (s or "").strip())
        if len(s) > 260:
            s = s[:257].rsplit(" ", 1)[0] + "…"
        if len(s) < 2:
            return
        low = s.lower()
        if low in seen_lower:
            return
        seen_lower.add(low)
        bullets.append(s)

    if decisions is not None and not decisions.empty and "reason" in decisions.columns:
        for r in decisions["reason"].dropna().astype(str).str.strip():
            if r and r.lower() not in ("nan", "none", ""):
                add_sentence(r)
            if len(bullets) >= max_bullets:
                return bullets[:max_bullets]

    for line in field_explanations:
        if len(bullets) >= max_bullets:
            break
        t = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        t = re.sub(r"`([^`]+)`", r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            add_sentence(t)

    return bullets[:max_bullets]


def _render_merge_outcome_previews(
    previews: list[dict[str, Any]],
    md: str,
    *,
    column_order: list[str],
    decisions: pd.DataFrame | None = None,
    show_full_merge_report: bool = False,
) -> None:
    """Show combined row(s), 3–5 AI insight bullets, field prefs in expander; full md only for batch when enabled."""
    if not previews:
        if md.strip() and show_full_merge_report:
            with st.expander("📄 Full merge report", expanded=False):
                st.markdown(
                    f'<div class="dq-muted">{_markdown_to_html_fragment(md)}</div>',
                    unsafe_allow_html=True,
                )
        return

    for i, p in enumerate(previews):
        if p.get("status") == "skipped":
            st.warning(f"Cluster {p.get('cluster')} — {p.get('message', 'Not merged.')}")
            det = p.get("detail_md") or ""
            if det:
                skip_bullets = _build_ai_insight_bullets([det], None, max_bullets=3)
                if skip_bullets:
                    subheading("💡 AI insights")
                    for b in skip_bullets:
                        st.markdown(f"- {b}")
            continue

        if p.get("status") == "merged":
            lab = (
                f"**Cluster** `{p.get('cluster')}` — survivor row **{p.get('survivor_index')}** · "
                f"latest row **{p.get('latest_row_index')}**"
            )
            st.markdown(
                f'<div class="dq-muted">{_markdown_to_html_fragment(lab)}</div>',
                unsafe_allow_html=True,
            )
            merged = p.get("merged_record") or {}
            subheading("📋 Combined record")
            st.dataframe(
                _merged_record_to_dataframe(merged, column_order),
                width='stretch',
                height=min(280, max(160, 80 + len(column_order) * 8)),
            )
            expl = p.get("field_explanations") or []
            insight_bullets = _build_ai_insight_bullets(expl, decisions, max_bullets=5)
            if insight_bullets:
                subheading("💡 AI insights")
                for b in insight_bullets:
                    st.markdown(f"- {b}")
            if expl:
                with st.expander("🔍 Field preferences (merge rules)", expanded=False):
                    st.markdown(
                        f'<div class="dq-muted">{_markdown_to_html_fragment("\n\n".join(expl))}</div>',
                        unsafe_allow_html=True,
                    )
            if i < len(previews) - 1:
                st.markdown("---")

    if md.strip() and show_full_merge_report:
        with st.expander("📄 Full merge report (technical)", expanded=False):
            st.markdown(
                f'<div class="dq-muted">{_markdown_to_html_fragment(md)}</div>',
                unsafe_allow_html=True,
            )


def _set_last_ai_merge(
    *,
    decisions: pd.DataFrame,
    previews: list[dict[str, Any]],
    md: str,
    no_merge_reason: str | None = None,
    show_full_merge_report: bool = False,
) -> None:
    """Store latest AI review outcome for the persistent results panel."""
    st.session_state.last_ai_merge = {
        "decisions": decisions,
        "previews": previews,
        "md": md,
        "no_merge_reason": no_merge_reason,
        "show_full_merge_report": show_full_merge_report,
    }


def _render_last_ai_merge_panel() -> None:
    """Persistent section: combined record + field notes after the most recent AI review."""
    lm = st.session_state.get("last_ai_merge")
    if not lm:
        return

    section_divider()
    section_heading("✨ Latest AI review result")
    if lm.get("no_merge_reason") == "no_same_person":
        dec = lm.get("decisions")
        nb = _build_ai_insight_bullets([], dec if dec is not None else None, max_bullets=5)
        if nb:
            subheading("💡 AI insights")
            for b in nb:
                st.markdown(f"- {b}")
        else:
            st.info(
                "No same-person merge was applied for **all pairs** — the dataset was not changed. "
                "(Per-group review still merges the full group.)"
            )
        return

    _render_merge_outcome_previews(
        lm.get("previews") or [],
        lm.get("md") or "",
        column_order=list(st.session_state.df.columns),
        decisions=lm.get("decisions"),
        show_full_merge_report=bool(lm.get("show_full_merge_report")),
    )
    if st.button("Clear latest AI result", key="btn_clear_last_ai_merge", type="secondary"):
        st.session_state.last_ai_merge = None
        st.toast("Cleared.", icon="✅")


_PAIR_DISPLAY_COLS = ["id_i", "id_j", "fuzzy_score", "verdict", "reason", "prompt_suggestion"]


def _prepare_matches_pairs_view(matches: pd.DataFrame) -> pd.DataFrame:
    """Sort and ensure columns exist for the duplicate-pairs table."""
    v = matches.sort_values("fuzzy_score", ascending=False).copy()
    if "prompt_suggestion" not in v.columns:
        v["prompt_suggestion"] = v.get("agent_prompt", "")
    for c in _PAIR_DISPLAY_COLS:
        if c not in v.columns:
            v[c] = ""
    return v[_PAIR_DISPLAY_COLS]


########## styled dataframes ##########


def _style_unified_dataframe(df: pd.DataFrame, baseline: pd.DataFrame):
    """Highlight cells that differ from *baseline* (blue); severity colors on ``dq_flags`` when set."""
    aligned_base = baseline.reindex(index=df.index, columns=df.columns, fill_value="")
    changed_mask = df.astype(str) != aligned_base.astype(str)

    def _row_styles(row: pd.Series) -> list[str]:
        ridx = row.name
        styles: list[str] = []
        for col in df.columns:
            if col == "dq_flags":
                txt = str(row[col] or "").strip()
                if txt:
                    n = len([x for x in txt.split(";") if x.strip()])
                    if n >= 4:
                        styles.append("background-color: #fecaca; color: #7f1d1d;")
                    elif n >= 2:
                        styles.append("background-color: #fed7aa; color: #7c2d12;")
                    else:
                        styles.append("background-color: #fef3c7; color: #78350f;")
                elif bool(changed_mask.loc[ridx, col]):
                    styles.append("background-color: #dbeafe; color: #0f172a; font-weight: 500;")
                else:
                    styles.append("")
                continue
            if bool(changed_mask.loc[ridx, col]):
                styles.append("background-color: #dbeafe; color: #0f172a; font-weight: 500;")
            else:
                styles.append("")
        return styles

    return df.style.apply(_row_styles, axis=1)


def _build_unification_groups(matches: pd.DataFrame) -> list[list[int]]:
    """Build connected row groups from duplicate candidate pairs."""
    if matches.empty or "row_i" not in matches.columns or "row_j" not in matches.columns:
        return []

    # Keep only valid integer row indices.
    edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    for _, r in matches.iterrows():
        try:
            a = int(r.get("row_i"))
            b = int(r.get("row_j"))
        except Exception:
            continue
        edges.append((a, b))
        nodes.add(a)
        nodes.add(b)

    if not edges:
        return []

    # Union-find over candidate-pair graph.
    parent: dict[int, int] = {n: n for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    buckets: dict[int, list[int]] = {}
    for n in nodes:
        buckets.setdefault(find(n), []).append(n)

    groups = [sorted(v) for v in buckets.values() if len(v) > 1]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def _render_unification_banners(
    df: pd.DataFrame,
    matches: pd.DataFrame,
    api_key: str,
    model: str,
    review_rules: str,
) -> None:
    """Show clickable expanders for each connected unification group."""
    groups = _build_unification_groups(matches)
    if not groups:
        st.caption("📭 No multi-record groups at this threshold — try lowering **🎯 Minimum pair score**.")
        return

    COLS = 1
    for row_start in range(0, len(groups), COLS):
        chunk = groups[row_start : row_start + COLS]
        cols = st.columns(COLS)
        for col_idx, group in enumerate(chunk):
            gnum = row_start + col_idx + 1
            with cols[col_idx]:
                ids: list[str] = []
                for ridx in group:
                    if 0 <= ridx < len(df):
                        cid = str(df.iloc[ridx].get("customer_id", "") or "").strip()
                        ids.append(cid if cid else f"row:{ridx}")
                    else:
                        ids.append(f"row:{ridx}")
                sample = ", ".join(ids[:3]) + (" …" if len(ids) > 3 else "")
                title = f"🧩 Group {gnum} · {len(group)} records · {sample}"

                with st.expander(title):
                    subheading("📋 Rows in this group", tight=True)
                    group_df = df.iloc[group].copy()
                    group_df.insert(0, "row_index", group_df.index)
                    st.dataframe(group_df, width='stretch', height=200)

                    sub_pairs = matches[
                        matches["row_i"].isin(group) & matches["row_j"].isin(group)
                    ].copy()
                    if not sub_pairs.empty:
                        pair_cols = [
                            "row_i",
                            "row_j",
                            "id_i",
                            "id_j",
                            "fuzzy_score",
                            "verdict",
                            "reason",
                            "prompt_suggestion",
                        ]
                        if "prompt_suggestion" not in sub_pairs.columns:
                            sub_pairs["prompt_suggestion"] = sub_pairs.get("agent_prompt", "")
                        for c in pair_cols:
                            if c not in sub_pairs.columns:
                                sub_pairs[c] = ""
                        subheading("🔗 Pair scores", tight=True)
                        st.dataframe(
                            sub_pairs.sort_values("fuzzy_score", ascending=False)[pair_cols],
                            width='stretch',
                            height=160,
                        )

                    ambiguous = sub_pairs.copy()
                    if "verdict" in ambiguous.columns:
                        ambiguous = ambiguous[ambiguous["verdict"].astype(str).str.strip().str.lower() == "needs_agent_review"]
                    if not api_key.strip():
                        st.caption("🔑 Set `GOOGLE_API_KEY` to run AI review.")
                    elif ambiguous.empty:
                        st.caption("✅ No ambiguous pairs in this group.")
                    else:
                        if st.button("✨ Run AI review", key=f"btn_group_ai_{gnum}", type="primary"):
                            try:
                                decisions, raw_bad = _llm_duplicate_pair_decisions(
                                    df,
                                    ambiguous,
                                    api_key,
                                    model,
                                    review_rules=review_rules,
                                )
                                if decisions is None:
                                    st.warning("Could not parse JSON from model.")
                                    st.code(raw_bad or "", language="json")
                                else:
                                    st.toast(f"Group {gnum}: AI review complete — merging group", icon="✅")
                                    dec_copy = decisions.copy()
                                    removed, md_prev, previews = _apply_group_consolidation_from_decisions(
                                        gnum,
                                        list(group),
                                        dec_copy,
                                        force_entire_group=True,
                                    )
                                    _set_last_ai_merge(
                                        decisions=dec_copy,
                                        previews=previews,
                                        md=md_prev,
                                        show_full_merge_report=False,
                                    )
                                    st.success(
                                        f"✅ {removed} row(s) merged into one. "
                                        "See **✨ Latest AI review result** for **AI insights** and field preferences."
                                    )
                                    st.toast(
                                        f"Dataset updated: {removed} duplicate row(s) removed.",
                                        icon="✅",
                                    )
                            except Exception as exc:
                                _render_llm_call_error(exc)


def _render_data_quality_tab() -> None:
    """Tab 1: KPIs, dataset preview, consolidation log, duplicate pairs, AI review rules."""
    df_loaded = st.session_state.df
    m1, m2, m3 = st.columns(3)
    m1.metric("📊 Rows", f"{len(df_loaded):,}")
    m2.metric("📋 Columns", len(df_loaded.columns))
    m3.metric("📄 Source file", _DATA_CSV_PATH.name)

    section_heading("📊 Dataset", first=True)
    section_caption(
        """**🏷️ Generate flags** — adds `dq_flags` and related columns. 
           **🧹 Clean & unify** — standardizes and enriches rows in this table. 
           **↩️ Reset data** — restores the original load. Blue cells differ from the file; `dq_flags` uses severity colours when set."""
    )
    subheading("⚡ Actions")
    bf1, bf2, bf3 = st.columns(3)
    if bf1.button("🏷️ Generate flags", type="primary", key="btn_flags"):
        flagged = generate_dq_flags_only(st.session_state.df.copy(deep=True))
        save_snapshot("Generate dq_flags", flagged)
        st.toast("Flags added.", icon="✅")
    if bf2.button("🧹 Clean & unify", type="primary", key="btn_clean_unify"):
        snap = st.session_state.df.copy(deep=True)
        cleaned, _ = run_standardization(snap)
        enriched, _ = run_entity_geo_pipeline(cleaned)
        scored = compute_quality_score(enriched)
        report = {
            "before": profile_data_quality(snap),
            "after": profile_data_quality(scored),
        }
        save_snapshot("Full quality pipeline", scored, report=report)
        st.session_state.matches = find_duplicate_candidates(st.session_state.df, min_score=0.55, high_band=0.75)
        st.session_state.dedupe_ran = True
        st.session_state.clean_unify_ran = True
        st.session_state.group_ai_results = {}
        st.session_state.df_before_ai_consolidation = None
        st.session_state.ai_consolidation_log = []
        st.toast("Standardized, enriched, and dedupe candidates computed.", icon="✅")
    if bf3.button("↩️ Reset data", type="secondary", key="btn_reset"):
        st.session_state.df = st.session_state.original_df.copy(deep=True)
        st.session_state.history = [("Initial load", st.session_state.df.copy(deep=True))]
        st.session_state.matches = pd.DataFrame()
        st.session_state.dedupe_ran = False
        st.session_state.clean_unify_ran = False
        st.session_state.group_ai_results = {}
        st.session_state.df_before_ai_consolidation = None
        st.session_state.ai_consolidation_log = []
        st.session_state.ai_review_rules = _DEFAULT_AI_REVIEW_RULES
        st.session_state.last_ai_merge = None
        st.toast(f"Restored original {_DATA_CSV_PATH.name}.", icon="↩️")

    df_work = st.session_state.df
    n_rows = len(df_work)
    preview_max = max(1, min(200, n_rows))
    preview_default = min(50, preview_max)
    preview_n = st.slider(
        "👁️ Rows",
        min_value=1,
        max_value=preview_max,
        value=preview_default,
        step=1,
        key="preview_rows",
    )
    if "dq_score" in df_work.columns:
        _dq = pd.to_numeric(df_work["dq_score"], errors="coerce")
        view_df = (
            df_work.assign(_dq_score_sort=_dq)
            .sort_values("_dq_score_sort", ascending=True, na_position="last")
            .drop(columns=["_dq_score_sort"])
            .head(preview_n)
        )
    else:
        view_df = df_work.head(preview_n)
    base_aligned = st.session_state.original_df.reindex(
        index=view_df.index, columns=view_df.columns, fill_value=""
    )
    st.dataframe(
        _style_unified_dataframe(view_df, base_aligned),
        width='stretch',
        height=480,
    )

    if st.session_state.get("df_before_ai_consolidation") is not None:
        subheading("↩️ AI consolidation")
        widget_caption(
            "💾 A snapshot was saved before AI-driven merges. Revert restores the full table to that point."
        )
        if st.button("↩️ Revert AI consolidation", type="secondary", key="btn_revert_ai_consolidation"):
            st.session_state.df = st.session_state.df_before_ai_consolidation.copy(deep=True)
            st.session_state.df_before_ai_consolidation = None
            st.session_state.matches = find_duplicate_candidates(
                st.session_state.df, min_score=0.55, high_band=0.75
            )
            st.session_state.group_ai_results = {}
            st.session_state.ai_consolidation_log = []
            st.toast("Reverted AI consolidation.", icon="↩️")

    log = st.session_state.get("ai_consolidation_log") or []
    if log:
        with st.expander("📋 Consolidation log (field-level)", expanded=False):
            for entry in reversed(log):
                st.markdown(
                    f"**Group {entry['group_num']}** — removed **{entry['rows_removed']}** row(s)"
                )
                st.markdown(entry["explanation"])
                st.markdown("---")

    section_divider()
    section_heading("🔎 Duplicate detection")
    section_caption(
        """🔗 **Blocking + fuzzy matching** on candidate pairs. 
        Run **🧹 Clean & unify** first so pairs are computed. 
        Open the **🔎 Duplicate & AI review** tab for Gemini review and merges."""
    )
    matches = st.session_state.get("matches", pd.DataFrame())
    if not st.session_state.get("dedupe_ran"):
        st.info("👆 Run **🧹 Clean & unify** in **📊 Dataset** to compute candidate pairs and unlock duplicate review.")
    elif matches.empty:
        st.success("✅ No candidate pairs at the current thresholds.")
    else:
        subheading("🔗 Candidate pairs")
        widget_caption("🟡 Highlighted rows need **agent review** (`needs_agent_review`).")
        _pairs_view = _prepare_matches_pairs_view(matches)
        _dedupe_cols = _PAIR_DISPLAY_COLS

        def _dedupe_row_style(row: pd.Series) -> list[str]:
            verdict = str(row.get("verdict", "")).strip().lower()
            if verdict == "needs_agent_review":
                return ["background-color: #fef3c7; color: #78350f;"] * len(_dedupe_cols)
            return [""] * len(_dedupe_cols)

        st.dataframe(
            _pairs_view.style.apply(_dedupe_row_style, axis=1),
            width='stretch',
            height=320,
        )
        subheading("📈 Pair summary")
        m_a, m_b = st.columns(2)
        m_a.metric("🔗 Pairs", len(matches))
        m_b.metric("🟡 Needs review", int((matches["verdict"] == "needs_agent_review").sum()) if "verdict" in matches.columns else 0)

    section_divider()



def _render_duplicate_ai_review_tab(api_key: str, model: str) -> None:
    """Tab 2: similar record groups, batch Gemini review, latest merge summary."""

    section_heading("📝 AI review instructions")
    section_caption(
        """🔁 Defaults load on each new session or when the file reloads 
        **↩️ Reset data** restores them too). """
    )
    subheading("✏️ Business rules")
    st.text_area("",
        key="ai_review_rules",
        height=156,
        help="Extends the base Gemini instructions for duplicate review.",
    )
    matches = st.session_state.get("matches", pd.DataFrame())
    if (
        st.session_state.get("dedupe_ran")
        and not matches.empty
        and st.session_state.get("clean_unify_ran", False)
    ):
        section_divider()
        section_heading("🧩 Similar record groups")
        section_caption(
            "Clusters of **connected** candidate pairs. Tune the threshold, then expand a group for **per-group AI review**."
        )
        subheading("📐 Similarity threshold")
        widget_caption("Only pairs at or above this fuzzy score are used to build groups.")
        group_score = st.slider(
            "🎯 Minimum pair score",
            min_value=0.55,
            max_value=1.00,
            value=0.75,
            step=0.01,
            key="group_similarity_threshold",
        )
        similar_matches = matches[matches["fuzzy_score"] >= float(group_score)].copy()
        subheading("👥 Groups")
        widget_caption(
            f"Showing clusters where at least one pair has score **≥ {group_score:.2f}**."
        )
        _render_unification_banners(
            st.session_state.df,
            similar_matches,
            api_key,
            model,
            str(st.session_state.get("ai_review_rules", _DEFAULT_AI_REVIEW_RULES)),
        )

    if st.session_state.get("dedupe_ran"):
        section_divider()
        section_heading("🤖 AI review — all pairs")
        section_caption(
            "One batch call to Gemini for up to **30** ambiguous pairs. For smaller scope, use **🧩 Similar record groups** in this tab."
        )
        if not api_key:
            st.warning("🔑 Set `GOOGLE_API_KEY` in `.env` (or your environment) to enable Gemini.")
        else:
            candidates = st.session_state.get("matches", pd.DataFrame())
            if candidates.empty or "verdict" not in candidates.columns:
                st.caption("📭 No pairs to send — run **🧹 Clean & unify** first.")
            else:
                ambiguous = candidates[candidates["verdict"] == "needs_agent_review"].head(30)
                if ambiguous.empty:
                    st.caption("✅ No ambiguous pairs in this run — nothing to send.")
                else:
                    subheading("⚡ Batch decision")
                    if st.button("✨ Get AI decisions (all pairs)", type="primary", key="btn_llm"):
                        try:
                            decisions, raw_bad = _llm_duplicate_pair_decisions(
                                st.session_state.df,
                                ambiguous,
                                api_key,
                                model,
                                review_rules=str(
                                    st.session_state.get("ai_review_rules", _DEFAULT_AI_REVIEW_RULES)
                                ),
                            )
                            if decisions is None:
                                st.warning("Could not parse JSON from model.")
                                st.code(raw_bad or "", language="json")
                            else:
                                sp = decisions["same_person"].map(_same_person_truthy)
                                yes = int(sp.sum())
                                no = int((~sp).sum())
                                c1, c2 = st.columns(2)
                                c1.metric("✅ Same person", yes)
                                c2.metric("➖ Different", no)
                                st.dataframe(decisions, width='stretch')
                                st.toast(f"{len(decisions)} AI decisions", icon="✅")
                                dec_copy = decisions.copy()
                                idx_set: set[int] = set()
                                for _, r in ambiguous.iterrows():
                                    try:
                                        idx_set.add(int(r["row_i"]))
                                        idx_set.add(int(r["row_j"]))
                                    except (TypeError, ValueError):
                                        pass
                                group_rows = sorted(idx_set)
                                removed, md_prev, previews = _apply_group_consolidation_from_decisions(
                                    "all pairs",
                                    group_rows,
                                    dec_copy,
                                    force_entire_group=True,
                                )
                                _set_last_ai_merge(
                                    decisions=dec_copy,
                                    previews=previews,
                                    md=md_prev,
                                    show_full_merge_report=False,
                                )
                                st.success(
                                    f"**✅ Forced combination:** {removed} row(s) merged into one. "
                                    "See **✨ Latest AI review result** for **AI insights** and field preferences."
                                )
                                st.toast(
                                    f"Dataset updated: {removed} duplicate row(s) removed.",
                                    icon="✅",
                                )
                                buf = StringIO()
                                decisions.to_csv(buf, index=False)
                                st.download_button(
                                    "⬇️ Export decisions (CSV)",
                                    data=buf.getvalue().encode("utf-8"),
                                    file_name="llm_duplicate_decisions.csv",
                                    mime="text/csv",
                                )
                        except Exception as exc:
                            _render_llm_call_error(exc)

    _render_last_ai_merge_panel()


def main() -> None:
    """Streamlit entry: ``data.csv`` single view — one dataset table, dedupe + LLM."""
    st.set_page_config(
        page_title="Data Quality Studio",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_theme()

    hero(
        "Data Quality Studio",
        "📊 Profile your dataset · 🧹 standardize & enrich · 🔎 review duplicates · ✨ optional Gemini for ambiguous pairs.",
    )

    if not _DATA_CSV_PATH.is_file():
        st.error(f"Dataset not found: `{_DATA_CSV_PATH}`. Add `data.csv` next to `app.py`.")
        return

    file_token = _data_file_token()

    if st.session_state.get("data_file_token") != file_token:
        try:
            loaded = load_default_dataset()
        except Exception as exc:
            st.error(f"Could not read `{_DATA_CSV_PATH}`: {exc}")
            return
        init_state(loaded)
        st.session_state.data_file_token = file_token

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    if "ai_review_rules" not in st.session_state:
        st.session_state.ai_review_rules = _DEFAULT_AI_REVIEW_RULES
    if "last_ai_merge" not in st.session_state:
        st.session_state.last_ai_merge = None

    tab_quality, tab_review = st.tabs(["📊 Data quality", "🔎 Duplicate & AI review"])

    with tab_quality:
        _render_data_quality_tab()

    with tab_review:
        _render_duplicate_ai_review_tab(api_key, model)


if __name__ == "__main__":
    main()

