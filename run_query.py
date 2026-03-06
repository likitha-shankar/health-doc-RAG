"""
Quick test script for the full RAG pipeline.

Usage:
    python run_query.py "What are the contraindications for metformin?"
"""

import sys

from app.generation.generator import generate_answer


def main():
    if len(sys.argv) < 2:
        query = "What are the renal contraindications for metformin?"
        print(f"No query provided. Using default: \"{query}\"\n")
    else:
        query = " ".join(sys.argv[1:])

    print(f"{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}\n")

    result = generate_answer(query)

    print(f"\n{'='*70}")
    print("GENERATED ANSWER")
    print(f"{'='*70}\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
