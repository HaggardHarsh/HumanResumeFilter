"""
llm_scorer.py — Step 4: LLM Re-Ranking ("Recruiter Intuition").

Constructs a structured prompt, calls an LLM (Grok/xAI, OpenAI, or
Google Gemini) to score each shortlisted candidate 0-100 with a
2-sentence justification, and returns sorted results.
"""

import json
import logging
import time
from typing import Optional

from config import PipelineConfig

logger = logging.getLogger(__name__)

# ── Prompt Template ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert technical recruiter with 15 years of experience \
evaluating engineering candidates. You evaluate candidates holistically \
— not just by keyword matches, but by career trajectory, proven impact, \
behavioral signals, and cultural fit."""

USER_PROMPT_TEMPLATE = """\
Compare the following Job Description with the Candidate Profile below.

**Evaluation criteria (in order of importance):**
1. Technical skill alignment with the role's requirements
2. Relevant experience depth and career trajectory
3. Demonstrated impact and leadership signals
4. Cultural/behavioral fit indicators

**Job Description:**
{jd_text}

**Candidate Profile:**
{candidate_doc}

**Respond ONLY with valid JSON — no markdown, no code fences, no extra text:**
{{"score": <integer 0-100>, "justification": "<exactly 2 sentences>"}}
"""


# ======================================================================
#  LLM Client Factory
# ======================================================================

def _call_openai_compatible(
    config: PipelineConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """Call an OpenAI-compatible API (works for OpenAI and xAI/Grok)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )

    response = client.chat.completions.create(
        model=config.resolved_model,
        temperature=config.llm_temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)


def _call_gemini(
    config: PipelineConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """Call Google Gemini via the google-generativeai SDK."""
    import google.generativeai as genai

    genai.configure(api_key=config.api_key)
    model = genai.GenerativeModel(
        model_name=config.resolved_model,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=config.llm_temperature,
            response_mime_type="application/json",
        ),
    )

    response = model.generate_content(user_prompt)
    content = response.text.strip()
    return json.loads(content)


# ======================================================================
#  Score a Single Candidate
# ======================================================================

def _score_candidate(
    config: PipelineConfig,
    jd_text: str,
    candidate_doc: str,
) -> dict:
    """
    Score one candidate against the JD via LLM.

    Returns
    -------
    dict with keys: score (int), justification (str)
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        jd_text=jd_text,
        candidate_doc=candidate_doc,
    )

    if config.resolved_provider in ("xai", "openai", "groq"):
        result = _call_openai_compatible(config, SYSTEM_PROMPT, user_prompt)
    elif config.resolved_provider == "gemini":
        result = _call_gemini(config, SYSTEM_PROMPT, user_prompt)
    else:
        raise ValueError(f"Unsupported provider: {config.resolved_provider}")

    # Validate response structure
    score = int(result.get("score", 0))
    score = max(0, min(100, score))  # clamp
    justification = str(result.get("justification", "No justification provided."))

    return {"score": score, "justification": justification}


# ======================================================================
#  Batch Scoring with Retry
# ======================================================================

def score_candidates(
    config: PipelineConfig,
    jd_text: str,
    candidate_docs: list[str],
    candidate_indices: list[int],
    retrieval_scores: Optional[list[float]] = None,
) -> list[dict]:
    """
    Score a shortlist of candidates using the configured LLM.

    Parameters
    ----------
    config : PipelineConfig
    jd_text : str
        Full JD text for the prompt.
    candidate_docs : list[str]
        Candidate document strings (only the shortlisted ones).
    candidate_indices : list[int]
        Original DataFrame row indices for each candidate.
    retrieval_scores : list[float], optional
        The first-pass retrieval scores (for inclusion in output).

    Returns
    -------
    list[dict] sorted descending by LLM score. Each dict contains:
        index, llm_score, justification, retrieval_score
    """
    if not config.has_llm:
        logger.warning(
            "No LLM API key configured. Returning retrieval scores only."
        )
        results = []
        for i, idx in enumerate(candidate_indices):
            ret_score = retrieval_scores[i] if retrieval_scores else 0.0
            results.append({
                "index": idx,
                "llm_score": None,
                "justification": "LLM scoring skipped — no API key configured.",
                "retrieval_score": ret_score,
            })
        return results

    logger.info(
        "Scoring %d candidates with %s (%s) …",
        len(candidate_docs),
        config.resolved_provider.upper(),
        config.resolved_model,
    )

    results = []
    for i, (idx, doc) in enumerate(zip(candidate_indices, candidate_docs)):
        ret_score = retrieval_scores[i] if retrieval_scores else 0.0
        attempt = 0
        scored = False

        while attempt < config.llm_max_retries and not scored:
            try:
                result = _score_candidate(config, jd_text, doc)
                results.append({
                    "index": idx,
                    "llm_score": result["score"],
                    "justification": result["justification"],
                    "retrieval_score": ret_score,
                })
                scored = True
                logger.info(
                    "  [%d/%d] Index %d → score %d",
                    i + 1,
                    len(candidate_docs),
                    idx,
                    result["score"],
                )
            except Exception as e:
                attempt += 1
                logger.warning(
                    "  [%d/%d] Attempt %d failed: %s",
                    i + 1,
                    len(candidate_docs),
                    attempt,
                    str(e),
                )
                if attempt < config.llm_max_retries:
                    time.sleep(config.llm_retry_delay)

        if not scored:
            logger.error(
                "  [%d/%d] All %d attempts failed for index %d. "
                "Using retrieval score as fallback.",
                i + 1,
                len(candidate_docs),
                config.llm_max_retries,
                idx,
            )
            results.append({
                "index": idx,
                "llm_score": None,
                "justification": f"LLM scoring failed after {config.llm_max_retries} attempts.",
                "retrieval_score": ret_score,
            })

    # Sort by LLM score descending (None → -1 so failures sort last)
    results.sort(key=lambda r: r["llm_score"] if r["llm_score"] is not None else -1, reverse=True)

    logger.info("LLM scoring complete. %d candidates scored.", len(results))
    return results
