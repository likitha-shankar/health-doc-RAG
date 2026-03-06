"""
Evaluation pipeline for the healthcare RAG system.

Two evaluation modes:
1. LOCAL (default): Fast, fully offline evaluation using heuristic checks and
   local embeddings. No API key needed. Runs in ~2 minutes.
2. RAGAS + GEMINI: Full Ragas evaluation using Google Gemini as the LLM judge.
   Set GOOGLE_API_KEY env var to enable. More accurate — extracts claims,
   verifies faithfulness, and scores relevance using an actual LLM.

LOCAL METRICS (what we measure without an API):
- Faithfulness (approximate): Citation coverage + n-gram context grounding
- Answer Relevancy: Cosine similarity between question and answer embeddings
- Context Precision: Fuzzy text overlap between retrieved and ground truth context

These are approximations of the Ragas metrics, but they're reliable enough to
catch regressions. In production, you'd run the full Ragas suite with an API model.
"""

import os
import re
import sys
import json
import logging
import warnings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ── Suppress noisy library warnings before importing anything else ────
os.environ["GRPC_VERBOSITY"] = "ERROR"                       # gRPC C++ absl log spam
os.environ["GLOG_minloglevel"] = "2"                         # gRPC glog warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"                 # ChromaDB telemetry opt-out
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

from app.generation.generator import generate_answer
from app.embeddings.embedder import embed_texts
from app.evaluation.test_dataset import TEST_DATASET

load_dotenv()

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "eval_results"


# ── Metric Functions ──────────────────────────────────────────────────────


def _faithfulness(answer: str, contexts: list[str]) -> float:
    """Approximate faithfulness: checks citation coverage AND context grounding.

    Two components:
    1. Citation check (45% weight): Does the answer contain [Source N] markers?
       No citations = the model isn't attributing its claims.
    2. N-gram grounding (55% weight): What fraction of the answer's 3-grams
       appear in the context? Higher overlap = more grounded in source material.

    NOTE: This is a LOCAL HEURISTIC approximation. Real faithfulness requires
    an LLM judge to extract claims and verify each against context. 3-gram
    overlap is a reasonable proxy — it allows paraphrasing of medical terms
    while still catching grounded content. Full Ragas with an LLM judge would
    use stricter thresholds.
    """
    # Strip Qwen3 /think tags before scoring
    clean_answer = re.sub(r"</?think>", "", answer)

    # Component 1: Citation coverage
    has_citations = 1.0 if re.search(r"\[Source \d+\]", clean_answer) else 0.0

    # Component 2: N-gram grounding (3-grams — less strict than 4-grams,
    # accommodates paraphrasing of medical terminology)
    answer_text = clean_answer.split("SOURCES:")[0].replace("ANSWER:", "").strip()
    answer_clean = re.sub(r"\[Source \d+(?:,\s*Source \d+)*\]", "", answer_text).strip().lower()

    combined_context = " ".join(contexts).lower()
    words = answer_clean.split()

    if len(words) < 3:
        ngram_score = 1.0  # Too short to meaningfully check
    else:
        ngram_hits = 0
        total = len(words) - 2
        for j in range(total):
            ngram = " ".join(words[j : j + 3])
            if ngram in combined_context:
                ngram_hits += 1
        ngram_score = ngram_hits / total

    return 0.45 * has_citations + 0.55 * ngram_score


def _answer_relevancy(question: str, answer: str) -> float:
    """Answer relevancy via local embedding cosine similarity.

    Embeds the question and the answer content, measures how semantically
    similar they are. A relevant answer should be "about" the same thing
    as the question.
    """
    answer_text = answer.split("SOURCES:")[0].replace("ANSWER:", "").strip()
    if not answer_text:
        return 0.0

    vecs = embed_texts([question, answer_text])
    q_vec = np.array(vecs[0])
    a_vec = np.array(vecs[1])

    similarity = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec)))
    return max(0.0, similarity)


def _context_precision(retrieved_contexts: list[str], ground_truth_context: str) -> float:
    """Context precision: is the ground truth information in the retrieved chunks?

    Uses substring-containment: splits the ground truth into key phrases
    (sentences/clauses) and checks what fraction appear in the retrieved chunks.
    This is much better than SequenceMatcher.ratio() which compares full text
    similarity — ground truth snippets are short while chunks are long, making
    ratio() inherently produce low scores.

    NOTE: Local heuristic approximation. Full Ragas context_precision uses an
    LLM judge and would use higher thresholds.
    """
    gt_lower = ground_truth_context.lower().strip()
    combined_context = " ".join(ctx.lower() for ctx in retrieved_contexts)

    # Split ground truth into key phrases (by sentence-ending punctuation or semicolons)
    phrases = re.split(r"[.;!\n]+", gt_lower)
    phrases = [p.strip() for p in phrases if len(p.strip()) > 10]

    if not phrases:
        # Fallback: check if the whole ground truth appears as substring
        return 1.0 if gt_lower in combined_context else 0.0

    # Check what fraction of ground truth phrases appear in retrieved chunks
    hits = sum(1 for phrase in phrases if phrase in combined_context)
    containment_score = hits / len(phrases)

    # Rank bonus: check which chunk contains the most phrases — earlier is better
    best_rank = len(retrieved_contexts)
    best_chunk_hits = 0
    for rank, ctx in enumerate(retrieved_contexts):
        ctx_lower = ctx.lower()
        chunk_hits = sum(1 for phrase in phrases if phrase in ctx_lower)
        if chunk_hits > best_chunk_hits:
            best_chunk_hits = chunk_hits
            best_rank = rank

    rank_factor = 1.0 / (1 + best_rank * 0.2)

    return containment_score * rank_factor


# ── Main Evaluation Pipeline ─────────────────────────────────────────────


def run_evaluation(save_results: bool = True) -> dict:
    """Run the full evaluation pipeline.

    For each test question:
    1. Run our RAG pipeline (retrieve → rerank → generate)
    2. Compute faithfulness, answer relevancy, and context precision
    3. Report scores and save results

    Returns dict of average scores.
    """
    print("=" * 70)
    print("RUNNING EVALUATION (local, no API required)")
    print("=" * 70)

    all_scores = []

    for i, test_item in enumerate(TEST_DATASET):
        question = test_item["question"]
        print(f"\n[Eval {i+1}/{len(TEST_DATASET)}] {question}")

        # Run full RAG pipeline
        result = generate_answer(question, retrieval_top_k=10, rerank_top_k=5)
        contexts = [chunk["text"] for chunk in result["context_chunks"]]

        # Compute all three metrics
        faith = _faithfulness(result["answer"], contexts)
        relevancy = _answer_relevancy(question, result["answer"])
        precision = _context_precision(contexts, test_item["ground_truth_context"])

        scores = {
            "question": question,
            "faithfulness": round(faith, 4),
            "answer_relevancy": round(relevancy, 4),
            "context_precision": round(precision, 4),
        }
        all_scores.append(scores)

        print(f"  Faithfulness:      {scores['faithfulness']}")
        print(f"  Answer Relevancy:  {scores['answer_relevancy']}")
        print(f"  Context Precision: {scores['context_precision']}")

    # Compute averages
    avg_scores = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        values = [s[metric] for s in all_scores]
        avg_scores[metric] = round(sum(values) / len(values), 4)

    print(f"\n{'='*70}")
    print("AVERAGE SCORES")
    print(f"{'='*70}")
    # Local heuristic thresholds — lower than full Ragas w/ LLM judge
    # Full Ragas would use: faithfulness≥0.8, context_precision≥0.7
    thresholds = {"faithfulness": 0.65, "answer_relevancy": 0.7, "context_precision": 0.5}
    for metric, score in avg_scores.items():
        threshold = thresholds[metric]
        status = "PASS" if score >= threshold else "NEEDS IMPROVEMENT"
        print(f"  {metric:25s}: {score:.4f}  [{status}]")

    # Save results
    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = {"average_scores": avg_scores, "per_sample": all_scores}
        output_path = RESULTS_DIR / "ragas_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return avg_scores


# ── Full Ragas Evaluation with Gemini ─────────────────────────────────


def run_ragas_evaluation(save_results: bool = True) -> dict:
    """Run full Ragas evaluation using Google Gemini as the LLM judge.

    Requires GOOGLE_API_KEY env var. Uses Ragas 0.2.x API with:
    - SingleTurnSample for each test question
    - EvaluationDataset to batch samples
    - evaluate() with faithfulness, answer_relevancy, context_precision, context_recall

    Returns dict of average scores.
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    api_key = os.environ["GOOGLE_API_KEY"]

    print("=" * 70)
    print("RUNNING EVALUATION (full Ragas with Google Gemini)")
    print("=" * 70)

    # Set up Gemini as the LLM judge
    gemini_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    )
    gemini_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    )

    # Build samples by running the RAG pipeline on each test question
    samples = []
    for i, test_item in enumerate(TEST_DATASET):
        question = test_item["question"]
        print(f"\n[Eval {i+1}/{len(TEST_DATASET)}] {question}")

        # Run full RAG pipeline
        result = generate_answer(question, retrieval_top_k=10, rerank_top_k=5)
        contexts = [chunk["text"] for chunk in result["context_chunks"]]

        sample = SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=test_item["ground_truth"],
        )
        samples.append(sample)
        print(f"  Retrieved {len(contexts)} chunks, answer length: {len(result['answer'])} chars")

    # Evaluate with Ragas
    dataset = EvaluationDataset(samples=samples)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print(f"\nRunning Ragas evaluation with {len(metrics)} metrics...")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=gemini_llm,
        embeddings=gemini_embeddings,
    )

    # Extract scores
    scores_df = ragas_result.to_pandas()
    avg_scores = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric_name in scores_df.columns:
            avg_scores[metric_name] = round(float(scores_df[metric_name].mean()), 4)

    # Report results
    print(f"\n{'='*70}")
    print("AVERAGE SCORES (full Ragas with Gemini)")
    print(f"{'='*70}")
    thresholds = {
        "faithfulness": 0.80,
        "answer_relevancy": 0.70,
        "context_precision": 0.70,
        "context_recall": 0.70,
    }
    for metric, score in avg_scores.items():
        threshold = thresholds.get(metric, 0.70)
        status = "PASS" if score >= threshold else "NEEDS IMPROVEMENT"
        print(f"  {metric:25s}: {score:.4f}  [{status}]")

    # Print per-sample scores
    print(f"\n{'='*70}")
    print("PER-SAMPLE SCORES")
    print(f"{'='*70}")
    for idx, row in scores_df.iterrows():
        print(f"\n  Q{idx+1}: {TEST_DATASET[idx]['question'][:60]}...")
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric_name in scores_df.columns:
                print(f"    {metric_name:25s}: {row[metric_name]:.4f}")

    # Save results
    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        per_sample = []
        for idx, row in scores_df.iterrows():
            sample_scores = {"question": TEST_DATASET[idx]["question"]}
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric_name in scores_df.columns:
                    sample_scores[metric_name] = round(float(row[metric_name]), 4)
            per_sample.append(sample_scores)

        output = {
            "mode": "ragas_gemini",
            "average_scores": avg_scores,
            "per_sample": per_sample,
        }
        output_path = RESULTS_DIR / "ragas_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return avg_scores


if __name__ == "__main__":
    # Auto-select evaluation mode based on API key availability
    if os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY detected — running full Ragas with Gemini\n")
        scores = run_ragas_evaluation()
        thresholds = {
            "faithfulness": 0.80,
            "answer_relevancy": 0.70,
            "context_precision": 0.70,
            "context_recall": 0.70,
        }
    else:
        print("No GOOGLE_API_KEY — running local heuristic evaluation\n")
        scores = run_evaluation()
        thresholds = {"faithfulness": 0.65, "answer_relevancy": 0.7, "context_precision": 0.5}

    failed = any(scores.get(m, 0) < t for m, t in thresholds.items())

    for m, t in thresholds.items():
        if scores.get(m, 0) < t:
            print(f"\nFAILED: {m} ({scores[m]:.4f}) < threshold ({t})")

    if not failed:
        print("\nAll metrics PASSED thresholds.")

    exit(1 if failed else 0)
