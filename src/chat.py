from pathlib import Path

from retriever import Retriever
from generator import Generator
from query_normalizer import (
    normalize_query,
    classify_query_category,
)
from reranker import get_reranker

INDEX_DIR = "../data/processed"

print("\nLoading RAKA components...\n")

retriever = Retriever.load(Path(INDEX_DIR))
reranker = get_reranker(retriever._embedder)
generator = Generator()

print("RAKA Chat Ready!")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    normalized_query = normalize_query(question)
    category = classify_query_category(normalized_query)

    print(f"\n[Normalized] {normalized_query}")
    print(f"[Category]   {category}")

    results = retriever.retrieve(
        normalized_query,
        k=5,
    )

    results = reranker.rerank(
        normalized_query,
        results,
        top_k=5,
    )

    response = generator.generate(
        normalized_query,
        results,
    )

    print("\nAnswer:\n")
    print(response.answer)
    print("\nSources:\n")

    for i, src in enumerate(response.sources, 1):
        print(
            f"[{i}] {src['source']} "
            f"(pp. {src['page_start']}-{src['page_end']}) "
            f"score={src['score']:.3f}"
        )
    print("\n" + "=" * 70 + "\n")