"""
Citation-enforced answer generation using an LLM.

This module ties together the full RAG pipeline:
1. Hybrid retrieval (Phase 3) → candidate chunks
2. Cross-encoder reranking (Phase 4) → top-K best chunks
3. Prompt construction → system prompt + labeled context + question
4. LLM call → citation-grounded answer

WHY WE USE A SYSTEM PROMPT (not just instructions in the user message):
System prompts have a privileged position — the model treats them as
persistent behavioral constraints, not just suggestions. Putting citation
rules in the system prompt makes them harder for the model to "forget" or
override, even with adversarial user inputs.

WHY WE LABEL CHUNKS AS [Source N]:
The model needs a referencing mechanism. By giving each chunk a numbered
label and requiring citations like [Source 1], we create a closed system:
- The model can only cite sources we provided
- We can verify that [Source 1] matches what we actually gave it
- The SOURCES section creates a human-readable audit trail

LLM BACKEND:
We use Ollama (local, free) via the OpenAI-compatible API. This keeps
everything offline and avoids API costs. The code structure makes it
trivial to swap to Anthropic Claude or OpenAI GPT by changing the
base_url and model name.
"""

import os

from openai import OpenAI

from app.retrieval.hybrid_retriever import hybrid_search
from app.reranking.reranker import rerank
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt

# Ollama exposes an OpenAI-compatible API on localhost:11434.
# We use the OpenAI SDK pointed at Ollama — same interface, local model.
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:latest"


def _get_client() -> OpenAI:
    """Create an OpenAI client pointed at the local Ollama server.

    Why OpenAI SDK for Ollama? Ollama implements the OpenAI API spec,
    so the same client code works with Ollama, OpenAI, Azure OpenAI,
    or any OpenAI-compatible provider. One code path, many backends.
    """
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # Ollama doesn't need a real key, but the SDK requires one
    )


def generate_answer(
    query: str,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Full RAG pipeline: retrieve → rerank → generate cited answer.

    Args:
        query: The user's natural language question.
        retrieval_top_k: How many candidates to pull from hybrid retrieval.
        rerank_top_k: How many top candidates to keep after reranking.
        model: Which LLM model to use.

    Returns:
        Dict with:
        - "answer": The full model response (ANSWER + SOURCES sections)
        - "context_chunks": The reranked chunks that were fed to the model
        - "query": The original query (for evaluation/logging)

    The pipeline in action:
        Query → hybrid_search() → 10 candidates
             → rerank() → 5 best candidates
             → build_user_prompt() → structured prompt with [Source N] labels
             → LLM → citation-grounded answer
    """
    if not query or not query.strip():
        return {"answer": "Please enter a question.", "context_chunks": [], "query": query}

    # Stage 1: Retrieve candidates (vector + BM25 + RRF)
    print(f"[Generator] Retrieving candidates for: {query}")
    candidates = hybrid_search(query, top_k=retrieval_top_k)
    print(f"[Generator] Got {len(candidates)} candidates from hybrid search")

    # Stage 2: Rerank to find the best chunks
    reranked = rerank(query, candidates, top_k=rerank_top_k)
    print(f"[Generator] Reranked to top {len(reranked)} chunks")
    for i, chunk in enumerate(reranked):
        score = chunk.get("rerank_score", "N/A")
        section = chunk["metadata"]["section_title"]
        print(f"  [{i+1}] score={score:.2f}  section='{section}'")

    # Stage 3: Build the prompt with labeled context
    user_prompt = build_user_prompt(query, reranked)

    # Stage 4: Call the LLM
    print(f"[Generator] Calling {model} via Ollama...")
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # Low temperature for factual, deterministic answers.
                          # High temperature = more creative = more hallucination risk.
    )

    answer_text = response.choices[0].message.content
    print(f"[Generator] Response received ({len(answer_text)} chars)")

    return {
        "answer": answer_text,
        "context_chunks": reranked,
        "query": query,
    }
