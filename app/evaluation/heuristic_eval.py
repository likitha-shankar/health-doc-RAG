"""
Heuristic (non-LLM) evaluation for fast offline testing.

WHY THIS EXISTS:
Ragas uses an LLM to judge answer quality — it extracts claims, checks if they're
grounded, generates hypothetical questions, etc. This requires a capable, fast LLM.
Local models like Qwen3 are too slow for the complex multi-step reasoning Ragas
needs (each of 18 evaluations can take 3-5 minutes locally → 60-90 min total).

This heuristic evaluator provides APPROXIMATE scores using simple text analysis.
It's not as accurate as Ragas, but it runs in seconds and gives you a useful
signal during development. Use Ragas with an API model (Claude, GPT-4) for
official evaluation.

WHAT IT MEASURES:
1. Citation Coverage: Does the answer contain [Source N] citations?
2. Context Grounding: What fraction of answer sentences appear in the context?
3. Answer Completeness: Does the answer contain key terms from the ground truth?
4. Retrieval Quality: Is the ground truth context found in the retrieved chunks?
"""

import re
import json
from pathlib import Path
from difflib import SequenceMatcher

from app.generation.generator import generate_answer
from app.evaluation.test_dataset import TEST_DATASET

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "eval_results"


def _has_citations(answer: str) -> float:
    """Check if the answer contains [Source N] citations.

    Returns 1.0 if citations are present, 0.0 if not.
    This is a binary check — either the model cited its sources or it didn't.
    """
    citations = re.findall(r"\[Source \d+\]", answer)
    return 1.0 if citations else 0.0


def _context_grounding(answer: str, contexts: list[str]) -> float:
    """Estimate what fraction of the answer is grounded in the context.

    Approach: For each sentence in the answer, check if it has a high
    overlap with any context chunk. This is a crude approximation of
    faithfulness — real faithfulness checking requires semantic understanding.

    Uses SequenceMatcher for fuzzy string matching (handles minor
    paraphrasing and word reordering).
    """
    # Extract the ANSWER section (before SOURCES)
    answer_part = answer.split("SOURCES:")[0] if "SOURCES:" in answer else answer
    answer_part = answer_part.replace("ANSWER:", "").strip()

    # Split into sentences
    sentences = re.split(r"[.!?]\s+", answer_part)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return 1.0  # No substantial sentences to check

    combined_context = " ".join(contexts).lower()
    grounded_count = 0

    for sentence in sentences:
        # Remove citation markers for comparison
        clean = re.sub(r"\[Source \d+(?:,\s*Source \d+)*\]", "", sentence).strip().lower()
        if not clean:
            grounded_count += 1
            continue

        # Check fuzzy overlap with context
        ratio = SequenceMatcher(None, clean, combined_context).ratio()
        # Also check if key phrases from the sentence appear in context
        words = clean.split()
        # Check 4-grams (4 consecutive words) — if a 4-gram from the answer
        # appears in the context, the claim is likely grounded
        ngram_hits = 0
        ngram_total = max(1, len(words) - 3)
        for j in range(len(words) - 3):
            ngram = " ".join(words[j : j + 4])
            if ngram in combined_context:
                ngram_hits += 1

        ngram_score = ngram_hits / ngram_total
        # Consider grounded if either fuzzy match or n-gram match is good
        if ratio > 0.3 or ngram_score > 0.3:
            grounded_count += 1

    return grounded_count / len(sentences)


def _answer_completeness(answer: str, ground_truth: str) -> float:
    """How much of the ground truth information appears in the answer.

    Extracts key terms (numbers, medical terms, drug names) from the
    ground truth and checks what fraction appear in the answer.
    """
    # Extract key terms: numbers, capitalized words, medical terms
    gt_terms = set(re.findall(r"\b\d+(?:\.\d+)?(?:mg|%|mL|mmol|units?)\b", ground_truth.lower()))
    gt_terms.update(re.findall(r"\b\d+(?:\.\d+)?(?:mg|%|mL|mmol|units?)\b", ground_truth))

    # Also extract drug names and medical terms (capitalized multi-char words)
    medical_terms = re.findall(r"\b[A-Z][a-z]{3,}\b", ground_truth)
    gt_terms.update(t.lower() for t in medical_terms)

    # Add specific numbers that appear
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", ground_truth)
    gt_terms.update(numbers)

    if not gt_terms:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for term in gt_terms if term.lower() in answer_lower)
    return found / len(gt_terms)


def _retrieval_quality(retrieved_contexts: list[str], ground_truth_context: str) -> float:
    """Check if the ground truth context was found by retrieval.

    Measures the best overlap between any retrieved chunk and the
    ground truth context. If the right chunk is in the top results,
    this score is high.
    """
    gt_lower = ground_truth_context.lower()
    best_score = 0.0

    for ctx in retrieved_contexts:
        ctx_lower = ctx.lower()
        ratio = SequenceMatcher(None, gt_lower, ctx_lower).ratio()
        best_score = max(best_score, ratio)

    return best_score


def run_heuristic_evaluation(save_results: bool = True) -> dict:
    """Run fast heuristic evaluation.

    Returns average scores for each metric.
    """
    print("=" * 70)
    print("RUNNING HEURISTIC EVALUATION (fast, offline)")
    print("=" * 70)

    all_scores = []

    for i, test_item in enumerate(TEST_DATASET):
        question = test_item["question"]
        print(f"\n[Eval {i+1}/{len(TEST_DATASET)}] {question}")

        result = generate_answer(question, retrieval_top_k=10, rerank_top_k=5)
        contexts = [chunk["text"] for chunk in result["context_chunks"]]

        scores = {
            "question": question,
            "citation_coverage": _has_citations(result["answer"]),
            "context_grounding": _context_grounding(result["answer"], contexts),
            "answer_completeness": _answer_completeness(
                result["answer"], test_item["ground_truth"]
            ),
            "retrieval_quality": _retrieval_quality(
                contexts, test_item["ground_truth_context"]
            ),
        }
        all_scores.append(scores)

        print(f"  Citation Coverage:   {scores['citation_coverage']:.2f}")
        print(f"  Context Grounding:   {scores['context_grounding']:.2f}")
        print(f"  Answer Completeness: {scores['answer_completeness']:.2f}")
        print(f"  Retrieval Quality:   {scores['retrieval_quality']:.2f}")

    # Compute averages
    metrics = ["citation_coverage", "context_grounding", "answer_completeness", "retrieval_quality"]
    avg_scores = {}
    for metric in metrics:
        avg_scores[metric] = sum(s[metric] for s in all_scores) / len(all_scores)

    print(f"\n{'='*70}")
    print("AVERAGE SCORES")
    print(f"{'='*70}")
    for metric, score in avg_scores.items():
        threshold = 0.8
        status = "PASS" if score >= threshold else "NEEDS IMPROVEMENT"
        print(f"  {metric:25s}: {score:.4f}  [{status}]")

    # Approximate Ragas-equivalent scores for CI pipeline compatibility
    ragas_equivalent = {
        "faithfulness": (avg_scores["context_grounding"] + avg_scores["citation_coverage"]) / 2,
        "answer_relevancy": avg_scores["answer_completeness"],
        "context_precision": avg_scores["retrieval_quality"],
    }

    print(f"\n  --- Approximate Ragas-equivalent scores ---")
    for metric, score in ragas_equivalent.items():
        status = "PASS" if score >= 0.8 else "NEEDS IMPROVEMENT"
        print(f"  {metric:25s}: {score:.4f}  [{status}]")

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            "average_scores": avg_scores,
            "ragas_equivalent": ragas_equivalent,
            "per_sample": all_scores,
        }
        output_path = RESULTS_DIR / "heuristic_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return ragas_equivalent


if __name__ == "__main__":
    scores = run_heuristic_evaluation()

    thresholds = {
        "faithfulness": 0.8,
        "answer_relevancy": 0.7,
        "context_precision": 0.7,
    }
    failed = False
    for metric, threshold in thresholds.items():
        if scores.get(metric, 0) < threshold:
            print(f"\nFAILED: {metric} ({scores[metric]:.4f}) < threshold ({threshold})")
            failed = True

    if not failed:
        print("\nAll metrics PASSED thresholds.")

    exit(1 if failed else 0)
