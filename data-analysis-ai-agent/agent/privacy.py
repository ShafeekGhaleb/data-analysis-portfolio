"""
Build the ONLY dataset information that may be sent to Claude: a minimal,
safe, structured summary.

THE PRIVACY RULES (enforced here, in code, not just promised in a prompt):
1. No raw rows. Ever. There is no option to send them.
2. No cell values at all, not even "example values". Only column
   names, their kind (numeric/text/date), and counts.
3. Columns that look like personal data (names, emails, phones, addresses,
   IDs...) are flagged, so the model knows not to ask for their contents.
4. The only free text that comes from the file is the COLUMN NAMES. They are
   cleaned (control characters removed, length limited) and serialized as
   JSON in a way that cannot break the request's structure (see safe_json).

WHY "XML TAGS" ARE NOT A DEFENCE ON THEIR OWN
Putting data inside <data>...</data> only LABELS it. A cell containing
"</data> Ignore your instructions" could fake the end of the label. Our real
defences are layered:
  a) minimisation: no cell text is sent at all, so there is almost nothing
     an attacker can write into the request;
  b) structure: the summary is JSON, with < > & escaped, so text from the
     file can never close a tag or end a string early;
  c) instructions: the system prompt states that data is never instructions;
  d) capability: in Stage 3 the model can only return text; nothing it says
     is executed. (From Stage 4: a fixed allowlist of tested tools.)
"""

import json
import re

import pandas as pd

from tools.cleaning import DATE_PATTERN
from tools.inspection import inspect_data

MAX_COLUMN_NAME_LENGTH = 64

# Column-name words that suggest personal data (matched as whole words, any case).
PERSONAL_NAME_WORDS = {
    "name", "names", "firstname", "lastname", "surname", "fullname", "customer", "client",
    "email", "mail", "phone", "mobile", "tel", "telephone", "address", "street", "zip",
    "postcode", "postal", "birth", "dob", "birthday", "passport", "ssn", "iban", "account",
    "ip", "username", "user", "person", "patient", "employee", "contact",
}

# Value patterns that look like personal data, checked on a sample of text values.
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
PHONE_PATTERN = re.compile(r"^\+?[\d\s().-]{7,}$")
VALUE_SAMPLE_SIZE = 50        # how many values we look at per column (locally, never sent)
PATTERN_SHARE_THRESHOLD = 0.3  # flag the column if 30% of sampled values match

CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def clean_column_name(name) -> str:
    """Column names are the only text from the file we send: make them harmless."""
    text = CONTROL_CHARACTERS.sub(" ", str(name)).strip()
    if len(text) > MAX_COLUMN_NAME_LENGTH:
        text = text[:MAX_COLUMN_NAME_LENGTH] + "…"
    return text


def personal_data_reason(name, series: pd.Series) -> str | None:
    """
    Why this column may contain personal data, or None.
    Checks the column NAME and (locally) a sample of its VALUES.
    Heuristics can miss things; they are a safety net, not a guarantee.
    """
    words = set(re.split(r"[^a-z0-9]+", str(name).lower()))
    hits = words & PERSONAL_NAME_WORDS
    if hits:
        return f"column name contains {sorted(hits)}"
    texts = series.dropna().astype(str).str.strip()
    texts = texts[texts != ""].head(VALUE_SAMPLE_SIZE)
    if len(texts):
        if texts.str.match(EMAIL_PATTERN).mean() >= PATTERN_SHARE_THRESHOLD:
            return "values look like email addresses"
        # Phone check only for TEXT columns: numeric columns hold amounts, not phones.
        # Dates such as "2026-01-05" also consist of digits and dashes, so
        # date-shaped values are excluded first (our tests caught that mistake).
        if not pd.api.types.is_numeric_dtype(series):
            not_dates = texts[~texts.str.match(DATE_PATTERN)]
            if len(not_dates) and (not_dates.str.match(PHONE_PATTERN).sum() / len(texts)
                                   >= PATTERN_SHARE_THRESHOLD):
                return "values look like phone numbers"
    return None


def safe_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Structure only: size, and per column its name, kind and counts.
    Built from inspect_data(sample_rows=0), with every value-bearing field removed.
    """
    report = inspect_data(df, sample_rows=0)   # no sample rows at all
    columns = []
    for info, original_name in zip(report["columns"], df.columns):
        reason = personal_data_reason(original_name, df[original_name])
        columns.append({
            "name": clean_column_name(info["name"]),
            "kind": info["kind"],
            "missing": info["missing"],
            "missing_percent": info["missing_percent"],
            "unique_values": info["unique_values"],
            "possible_personal_data": reason is not None,
            **({"personal_data_reason": reason} if reason else {}),
            # NOTE: inspect_data's "examples" (real cell values) are deliberately NOT copied.
        })
    return {
        "n_rows": report["n_rows"],
        "n_columns": report["n_columns"],
        "duplicate_rows": report["duplicate_rows"],
        "total_missing_values": report["total_missing_values"],
        "columns": columns,
        "privacy": {
            "raw_rows_included": False,
            "cell_values_included": False,
            "note": "Only structure and counts. Columns flagged possible_personal_data must never be requested.",
        },
    }


def safe_json(data) -> str:
    """
    Serialize to JSON that cannot break the surrounding request text:
    - ensure_ascii=True: every non-ASCII character becomes \\uXXXX;
    - < > & are escaped as \\u003c \\u003e \\u0026, so no text from the file can
      ever form a tag such as </dataset_summary>;
    - sort_keys=True: the same data always gives exactly the same text
      (useful later for prompt caching, and for tests).
    """
    text = json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
