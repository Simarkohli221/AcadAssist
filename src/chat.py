from pathlib import Path

from retriever import Retriever
from generator import Generator
from faithfulness import FaithfulnessVerifier
from reranker import get_reranker
from pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_DIR = PROJECT_ROOT / "data" / "processed"
FAITHFULNESS_MODEL = PROJECT_ROOT / "models" / "faithfulness_lr.joblib"


print("\nLoading RAKA components...\n")


# ---------------------------------------------------------
# Load RAG components
# ---------------------------------------------------------

retriever = Retriever.load(INDEX_DIR)

reranker = get_reranker(
    retriever._embedder
)

generator = Generator()


# ---------------------------------------------------------
# Load Faithfulness Verifier
#
# Reuses retriever._embedder (already loaded for retrieval) instead of
# lazily loading a second SBERT instance, per faithfulness.py's own
# documented usage guidance.
# ---------------------------------------------------------

verifier = FaithfulnessVerifier(
    model_path=FAITHFULNESS_MODEL,
    embedder=retriever._embedder,
    index_dir=INDEX_DIR,
)


print("RAKA Chat Ready!")
print("Type 'exit' to quit.\n")


# ---------------------------------------------------------
# Chat loop
# ---------------------------------------------------------

while True:

    question = input("Ask: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        continue

    # -----------------------------------------------------
    # Run the shared normalize -> retrieve -> rerank ->
    # generate -> verify pipeline (same code path the web
    # API uses).
    # -----------------------------------------------------

    result = run_pipeline(question, retriever, reranker, generator, verifier)

    print(f"\n[Normalized] {result.normalized_query}")
    print(f"[Category]   {result.category}")

    # -----------------------------------------------------
    # Final answer
    # -----------------------------------------------------

    print("\nAnswer:\n")
    print(result.answer)

    if result.warning:
        print(f"\n⚠ Warning: {result.warning}")

    # -----------------------------------------------------
    # Faithfulness information
    # -----------------------------------------------------

    print("\nFaithfulness:")
    print(
        f"Status: {result.faithfulness_status}"
    )

    print(
        f"Score:  {result.faithfulness_score:.3f}"
    )

    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    print("\nSources:\n")

    for i, src in enumerate(
        result.sources,
        1
    ):

        print(
            f"[{i}] {src['source']} "
            f"(pp. {src['page_start']}-"
            f"{src['page_end']}) "
            f"score={src['score']:.3f}"
        )

    print("\n" + "=" * 70 + "\n")
