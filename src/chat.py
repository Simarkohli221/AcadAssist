from pathlib import Path

from retriever import Retriever
from generator import Generator
from faithfulness import FaithfulnessVerifier

from query_normalizer import (
    normalize_query,
    classify_query_category,
    get_category_sources,
)

from reranker import get_reranker


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
# ---------------------------------------------------------

verifier = FaithfulnessVerifier(
    model_path=FAITHFULNESS_MODEL,
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
    # 1. Normalize query
    # -----------------------------------------------------

    normalized_query = normalize_query(question)

    category = classify_query_category(
        normalized_query
    )
    allowed_sources = get_category_sources(category)

    print(f"\n[Normalized] {normalized_query}")
    print(f"[Category]   {category}")

    # -----------------------------------------------------
    # 2. Retrieve relevant chunks (candidate set k=10)
    # -----------------------------------------------------

    results = retriever.retrieve(
        normalized_query,
        k=10,
        allowed_sources=allowed_sources,
    )

    # -----------------------------------------------------
    # 3. Rerank retrieved chunks (best 3 chunks)
    # -----------------------------------------------------

    results = reranker.rerank(
        normalized_query,
        results,
        top_k=3,
    )


    # -----------------------------------------------------
    # 4. Generate answer using retrieved context
    # -----------------------------------------------------

    response = generator.generate(
        normalized_query,
        results,
    )


    # -----------------------------------------------------
    # 5. Build verification context
    #
    # IMPORTANT:
    # This must be the same retrieved chunk text
    # that was provided to the generator.
    # -----------------------------------------------------

    context = (
        response.context_used
        if response.context_used
        else "\n\n".join(result.text for result in results)
    )


    # -----------------------------------------------------
    # 6. Faithfulness verification
    # -----------------------------------------------------

    verification = verifier.predict(
        response.answer,
        context,
    )


    # -----------------------------------------------------
    # 7. Final answer
    # -----------------------------------------------------

    print("\nAnswer:\n")

    if verification["status"] == "unfaithful":

        print(
            "Unable to verify the answer from "
            "the available sources."
        )

    else:

        print(response.answer)

        if verification["status"] == "uncertain":

            print(
                "\n⚠ Warning: "
                "The answer may not be fully supported "
                "by the retrieved sources."
            )


    # -----------------------------------------------------
    # 8. Faithfulness information
    # -----------------------------------------------------

    print("\nFaithfulness:")
    print(
        f"Status: {verification['status']}"
    )

    print(
        f"Score:  {verification['score']:.3f}"
    )


    # -----------------------------------------------------
    # 9. Sources
    # -----------------------------------------------------

    print("\nSources:\n")

    for i, src in enumerate(
        response.sources,
        1
    ):

        print(
            f"[{i}] {src['source']} "
            f"(pp. {src['page_start']}-"
            f"{src['page_end']}) "
            f"score={src['score']:.3f}"
        )


    print("\n" + "=" * 70 + "\n")   