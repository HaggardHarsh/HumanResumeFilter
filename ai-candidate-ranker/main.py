"""
main.py — Step 5: Pipeline Orchestration & Output Generation.

Wires together all modules (data_loader, embedder, retriever, llm_scorer)
into a single CLI-driven pipeline that produces ranked candidate results.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from config import PipelineConfig
from data_loader import parse_job_description, load_candidates, build_candidate_documents
from embedder import EmbeddingEngine
from retriever import semantic_search, bm25_search, hybrid_search
from llm_scorer import score_candidates

# ── Logging Setup ─────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-14s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

logger = logging.getLogger("pipeline")


# ======================================================================
#  Main Pipeline
# ======================================================================

def run_pipeline(
    jd_path: str,
    candidates_path: str,
    output_path: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Execute the full candidate ranking pipeline.

    Returns the final ranked DataFrame.
    """

    # ── Step 1: Data Ingestion ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Data Ingestion & Preprocessing")
    logger.info("=" * 60)

    jd = parse_job_description(jd_path)
    logger.info("JD Title: %s", jd["title"])
    logger.info("JD Sections: %s", list(jd["sections"].keys()))

    df = load_candidates(candidates_path)
    logger.info("Candidates loaded: %d", len(df))

    candidate_docs = build_candidate_documents(df)

    # ── Step 2: Semantic Representation ───────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Semantic Representation (Embedding)")
    logger.info("=" * 60)

    engine = EmbeddingEngine(model_name=config.embedding_model)

    # Encode JD
    jd_vector = engine.encode([jd["full_document"]], show_progress=False)

    # Encode all candidates
    candidate_vectors = engine.encode(candidate_docs)

    # Build FAISS index
    index = engine.build_index(candidate_vectors)

    # ── Step 3: First-Pass Retrieval ──────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: First-Pass Retrieval")
    logger.info("=" * 60)

    # Semantic search
    sem_results = semantic_search(index, jd_vector[0], top_n=config.retrieval_top_n)

    if config.enable_hybrid_search:
        # BM25 search
        bm25_results = bm25_search(
            candidate_docs, jd["full_document"], top_n=config.retrieval_top_n
        )
        # Hybrid fusion
        retrieval_results = hybrid_search(
            sem_results, bm25_results,
            alpha=config.hybrid_alpha,
            top_n=config.retrieval_top_n,
        )
        logger.info("Using HYBRID search (α=%.2f).", config.hybrid_alpha)
    else:
        retrieval_results = sem_results
        logger.info("Using PURE SEMANTIC search.")

    logger.info("Shortlisted %d candidates for re-ranking.", len(retrieval_results))

    # Print shortlist preview
    logger.info("─" * 50)
    logger.info("Retrieval Shortlist (top 10 preview):")
    for rank, (idx, score) in enumerate(retrieval_results[:10], 1):
        name = df.iloc[idx]["name"] if idx < len(df) else "?"
        logger.info("  %2d. [%s] %s  (score: %.4f)", rank, df.iloc[idx]["candidate_id"], name, score)

    # ── Step 4: LLM Re-Ranking ────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: LLM Re-Ranking")
    logger.info("=" * 60)

    # Prepare shortlist for LLM
    llm_top_n = min(config.llm_top_n, len(retrieval_results))
    shortlist = retrieval_results[:llm_top_n]

    shortlist_indices = [idx for idx, _ in shortlist]
    shortlist_scores = [score for _, score in shortlist]
    shortlist_docs = [candidate_docs[idx] for idx in shortlist_indices]

    llm_results = score_candidates(
        config=config,
        jd_text=jd["full_document"],
        candidate_docs=shortlist_docs,
        candidate_indices=shortlist_indices,
        retrieval_scores=shortlist_scores,
    )

    # ── Step 5: Output Generation ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Output Generation")
    logger.info("=" * 60)

    output_df = _build_output_dataframe(df, llm_results)
    _export_results(output_df, output_path, config.output_format)

    # Print final ranking
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RANKING")
    logger.info("=" * 60)
    for _, row in output_df.iterrows():
        score_str = (
            f"LLM: {row['llm_score']:3d}"
            if pd.notna(row["llm_score"])
            else "LLM:  N/A"
        )
        logger.info(
            "  #%d │ %-20s │ %s │ Retrieval: %.4f │ %s",
            row["rank"],
            row["name"],
            score_str,
            row["retrieval_score"],
            row["justification"][:80],
        )

    logger.info("")
    logger.info("Results exported to: %s", output_path)
    logger.info("Pipeline complete.")

    return output_df


# ======================================================================
#  Output Helpers
# ======================================================================

def _build_output_dataframe(
    candidates_df: pd.DataFrame,
    llm_results: list[dict],
) -> pd.DataFrame:
    """Build the final ranked output DataFrame."""
    rows = []
    for rank, result in enumerate(llm_results, 1):
        idx = result["index"]
        candidate = candidates_df.iloc[idx]
        rows.append({
            "rank": rank,
            "candidate_id": candidate["candidate_id"],
            "name": candidate["name"],
            "title": candidate.get("title", ""),
            "final_score": (
                result["llm_score"]
                if result["llm_score"] is not None
                else round(result["retrieval_score"] * 100, 1)
            ),
            "llm_score": result["llm_score"],
            "retrieval_score": round(result["retrieval_score"], 4),
            "justification": result["justification"],
        })

    return pd.DataFrame(rows)


def _export_results(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Export results to CSV or JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        records = df.to_dict(orient="records")
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        df.to_csv(out, index=False, encoding="utf-8")

    logger.info("Exported %d ranked candidates to %s (%s).", len(df), path, fmt.upper())


# ======================================================================
#  CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Candidate Ranking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic run with sample data
  python main.py --jd sample_data/sample_jd.txt --candidates sample_data/sample_candidates.csv

  # Output as JSON, top 30 retrieval, top 10 LLM
  python main.py --jd jd.txt --candidates data.csv -o results.json --format json --retrieval-top-n 30 --llm-top-n 10

  # Force a specific LLM provider
  python main.py --jd jd.txt --candidates data.csv --llm-provider xai --llm-model grok-3-mini-fast
""",
    )

    parser.add_argument(
        "--jd", required=True,
        help="Path to the Job Description file (.txt, .md, or .json).",
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to the candidate data file (.csv or .json).",
    )
    parser.add_argument(
        "-o", "--output", default="ranked_results.csv",
        help="Output file path (default: ranked_results.csv).",
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default=None,
        help="Output format. If omitted, inferred from output file extension.",
    )

    # Retrieval options
    retrieval = parser.add_argument_group("retrieval options")
    retrieval.add_argument(
        "--retrieval-top-n", type=int, default=50,
        help="Number of candidates for first-pass retrieval (default: 50).",
    )
    retrieval.add_argument(
        "--no-hybrid", action="store_true",
        help="Disable hybrid search; use pure semantic search only.",
    )
    retrieval.add_argument(
        "--alpha", type=float, default=0.70,
        help="Semantic weight in hybrid search, 0-1 (default: 0.70).",
    )

    # LLM options
    llm = parser.add_argument_group("LLM re-ranking options")
    llm.add_argument(
        "--llm-top-n", type=int, default=20,
        help="Number of candidates to send to LLM for scoring (default: 20).",
    )
    llm.add_argument(
        "--llm-provider", choices=["auto", "groq", "xai", "openai", "gemini"], default="auto",
        help="LLM provider to use (default: auto-detect from env vars).",
    )
    llm.add_argument(
        "--llm-model", default="",
        help="Override the default model for the chosen provider.",
    )

    # General
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    # Infer output format from extension if not specified
    output_format = args.format
    if output_format is None:
        ext = Path(args.output).suffix.lower()
        output_format = "json" if ext == ".json" else "csv"

    config = PipelineConfig(
        retrieval_top_n=args.retrieval_top_n,
        hybrid_alpha=args.alpha,
        enable_hybrid_search=not args.no_hybrid,
        llm_top_n=args.llm_top_n,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        output_format=output_format,
    )

    logger.info("Pipeline configuration:")
    logger.info("  Embedding model:  %s", config.embedding_model)
    logger.info("  Retrieval top-N:  %d", config.retrieval_top_n)
    logger.info("  Hybrid search:    %s (α=%.2f)", config.enable_hybrid_search, config.hybrid_alpha)
    logger.info("  LLM provider:     %s", config.resolved_provider.upper())
    logger.info("  LLM model:        %s", config.resolved_model or "(none)")
    logger.info("  LLM top-N:        %d", config.llm_top_n)
    logger.info("  Output format:    %s", output_format.upper())
    logger.info("")

    try:
        run_pipeline(
            jd_path=args.jd,
            candidates_path=args.candidates,
            output_path=args.output,
            config=config,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
