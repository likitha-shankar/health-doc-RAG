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
from app.generation.citation_parser import parse_llm_output
from app.embeddings.vector_store import get_collection

# LLM backend configuration — works with any OpenAI-compatible API.
# Local: Ollama on localhost:11434 (default, fully offline)
# Cloud: Groq, Together AI, OpenRouter, etc. (set via env vars)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:latest")


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Create an OpenAI client pointed at the configured LLM backend.

    Why OpenAI SDK? Most LLM providers (Ollama, Groq, Together AI,
    OpenRouter) implement the OpenAI API spec. One client, many backends.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _client


def generate_answer(
    query: str,
    retrieval_top_k: int = 10,
    rerank_top_k: int = 5,
    model: str = LLM_MODEL,
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
        return {"answer": "Please enter a question.", "context_chunks": [], "citations": [], "query": query}

    if get_collection().count() == 0:
        return {"answer": "No documents loaded. Please upload a document first.", "context_chunks": [], "citations": [], "query": query}

    # Stage 1: Retrieve candidates (vector + BM25 + RRF)
    print(f"[Generator] Retrieving candidates for: {query}")
    candidates = hybrid_search(query, top_k=retrieval_top_k)
    print(f"[Generator] Got {len(candidates)} candidates from hybrid search")

    # Stage 2: Rerank to find the best chunks
    reranked = rerank(query, candidates, top_k=rerank_top_k)
    print(f"[Generator] Reranked to top {len(reranked)} chunks")
    for i, chunk in enumerate(reranked):
        score = chunk.get("rerank_score")
        section = chunk["metadata"]["section_title"]
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
        print(f"  [{i+1}] score={score_str}  section='{section}'")

    # Stage 3: Build the prompt with labeled context
    user_prompt = build_user_prompt(query, reranked)

    # Stage 4: Call the LLM
    # /no_think disables Qwen3's thinking mode to avoid wasting tokens.
    # Only append it for Qwen3 models; other models ignore or don't need it.
    prompt_content = user_prompt
    if "qwen3" in model.lower():
        prompt_content += "\n\n/no_think"

    print(f"[Generator] Calling {model} via {LLM_BASE_URL}...")
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content},
        ],
        temperature=0.1,  # Low temperature for factual, deterministic answers.
                          # High temperature = more creative = more hallucination risk.
        max_tokens=512,
    )

    raw_answer = response.choices[0].message.content
    print(f"[Generator] Response received ({len(raw_answer)} chars)")

    parsed = parse_llm_output(raw_answer, reranked)

    return {
        "answer": raw_answer,
        "answer_text": parsed["answer_text"],
        "citations": parsed["citations"],
        "context_chunks": reranked,
        "query": query,
    }
