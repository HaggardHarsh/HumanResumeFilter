"""
data_loader.py — Step 1: Data Ingestion & Preprocessing.

Parses Job Descriptions, loads candidate data (CSV/JSON),
cleans missing values, and assembles candidate documents.
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Section headings we try to extract from a JD ──────────────────────
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("core_responsibilities", re.compile(
        r"(?:core\s+)?responsibilities|key\s+duties|what\s+you.?ll\s+do",
        re.IGNORECASE,
    )),
    ("required_skills", re.compile(
        r"required\s+skills|technical\s+requirements|must[\s-]+have|qualifications",
        re.IGNORECASE,
    )),
    ("preferred_skills", re.compile(
        r"preferred|nice[\s-]+to[\s-]+have|bonus",
        re.IGNORECASE,
    )),
    ("soft_skills", re.compile(
        r"soft\s+skills|behavioral|interpersonal|culture",
        re.IGNORECASE,
    )),
    ("about", re.compile(
        r"about\s+(?:the\s+)?(?:role|position|company|us|team)",
        re.IGNORECASE,
    )),
]

# Columns we expect (or will create) in candidate data
_EXPECTED_COLUMNS = [
    "candidate_id", "name", "title", "skills", "experience_years",
    "education", "career_history", "certifications",
    "platform_activity", "bio",
]


# ======================================================================
#  JD Parsing
# ======================================================================

def parse_job_description(path: str) -> dict:
    """
    Parse a Job Description file (.txt / .md / .json) into structured sections.

    Returns
    -------
    dict with keys:
        raw_text          – full original text
        title             – extracted job title (first non-blank line)
        sections          – dict of section_name → text
        full_document     – cleaned concatenation for embedding
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"JD file not found: {path}")

    raw = filepath.read_text(encoding="utf-8")

    # If it's JSON, treat it as pre-structured
    if filepath.suffix.lower() == ".json":
        data = json.loads(raw)
        return {
            "raw_text": raw,
            "title": data.get("title", ""),
            "sections": {k: v for k, v in data.items() if k != "title"},
            "full_document": _flatten_dict(data),
        }

    # Otherwise parse free text
    lines = raw.strip().splitlines()
    title = next((l.strip() for l in lines if l.strip()), "Unknown Position")
    sections = _extract_sections(raw)

    # Build a clean document for embedding
    full_document = _clean_text(raw)

    logger.info(
        "Parsed JD: title='%s', sections=%s",
        title,
        list(sections.keys()),
    )

    return {
        "raw_text": raw,
        "title": title,
        "sections": sections,
        "full_document": full_document,
    }


def _extract_sections(text: str) -> dict[str, str]:
    """Heuristically split text into named sections."""
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current_section: Optional[str] = None
    buffer: list[str] = []

    for line in lines:
        matched = False
        for section_name, pattern in _SECTION_PATTERNS:
            if pattern.search(line):
                # Save previous section
                if current_section and buffer:
                    sections[current_section] = "\n".join(buffer).strip()
                current_section = section_name
                buffer = []
                matched = True
                break
        if not matched:
            buffer.append(line)

    # Save last section
    if current_section and buffer:
        sections[current_section] = "\n".join(buffer).strip()

    # If no sections found, put everything under "general"
    if not sections:
        sections["general"] = _clean_text(text)

    return sections


def _flatten_dict(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, list):
            v = ", ".join(str(i) for i in v)
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


# ======================================================================
#  Candidate Loading
# ======================================================================

def load_candidates(path: str) -> pd.DataFrame:
    """
    Load candidate data from CSV or JSON.

    Handles missing values, normalises text columns,
    and ensures expected columns exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    ext = filepath.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(filepath, dtype=str)
    elif ext == ".json":
        df = pd.read_json(filepath, dtype=str)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")

    logger.info("Loaded %d candidates from %s", len(df), path)

    # Ensure expected columns exist (fill with empty string if missing)
    for col in _EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            logger.warning("Column '%s' not found — created with empty values.", col)

    # Generate candidate_id if missing
    if df["candidate_id"].isna().all() or (df["candidate_id"] == "").all():
        df["candidate_id"] = [f"CAND-{i+1:04d}" for i in range(len(df))]

    # Clean text columns
    text_cols = [c for c in _EXPECTED_COLUMNS if c not in ("candidate_id", "experience_years")]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).apply(_clean_text)

    # Normalise experience_years to numeric
    df["experience_years"] = pd.to_numeric(
        df["experience_years"], errors="coerce"
    ).fillna(0).astype(int)

    return df


# ======================================================================
#  Candidate Document Assembly
# ======================================================================

def build_candidate_documents(df: pd.DataFrame) -> list[str]:
    """
    Assemble a single text document per candidate by concatenating
    all relevant fields. This document is what gets embedded.
    """
    documents = []
    for _, row in df.iterrows():
        parts = [
            f"Name: {row.get('name', '')}",
            f"Current Title: {row.get('title', '')}",
            f"Experience: {row.get('experience_years', 0)} years",
            f"Education: {row.get('education', '')}",
            f"Skills: {row.get('skills', '')}",
            f"Certifications: {row.get('certifications', '')}",
            f"Career History: {row.get('career_history', '')}",
            f"Platform Activity: {row.get('platform_activity', '')}",
            f"Bio: {row.get('bio', '')}",
        ]
        doc = "\n".join(p for p in parts if not p.endswith(": ") and not p.endswith(": 0 years"))
        documents.append(doc)

    logger.info("Built %d candidate documents.", len(documents))
    return documents


# ======================================================================
#  Text Utilities
# ======================================================================

def _clean_text(text: str) -> str:
    """Basic text cleaning: collapse whitespace, strip control chars."""
    text = re.sub(r"[^\S\n]+", " ", text)   # collapse spaces (keep newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive newlines
    text = text.strip()
    return text
