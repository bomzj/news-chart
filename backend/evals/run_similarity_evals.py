from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from langfuse import Evaluation
from langfuse.experiment import ExperimentResult

from src.config import app_config
from src.shared.embeddings import cosine_similarity, embed_texts
from src.shared.observability import langfuse_client, shutdown_langfuse

DEFAULT_DATASET = "similar-news"


async def news_similarity_task(
    *,
    item: Any,
    threshold: float | None = None,
    **kwargs: Any,
) -> dict[str, bool | float]:
    """Embed one dataset pair and classify it using the requested threshold."""
    del kwargs
    threshold_used = (
        app_config().dedup.cosine_threshold if threshold is None else threshold
    )
    news_1, news_2 = item.input["news_1"], item.input["news_2"]
    embeddings = await embed_texts([news_1, news_2])

    if len(embeddings) != 2:
        raise ValueError(
            f"Expected two embeddings for a news pair, received {len(embeddings)}"
        )

    cos_sim = cosine_similarity(embeddings[0], embeddings[1])
    return {
        "similar": cos_sim >= threshold_used,
        "cosine_similarity": cos_sim,
        "threshold_used": threshold_used,
    }


def _expected_similarity(expected_output: Any) -> bool:
    if isinstance(expected_output, bool):
        return expected_output
    if isinstance(expected_output, Mapping) and isinstance(
        expected_output.get("similar"), bool
    ):
        return expected_output["similar"]
    raise TypeError(
        "Dataset expected_output must be a boolean or a mapping with a boolean "
        "'similar' field"
    )


def eval_semantic_similarity(
    *,
    input: Any,
    output: Any,
    expected_output: Any = None,
    **kwargs: Any,
) -> Evaluation:
    """Score whether the threshold-based classification matches the human label."""
    del input, kwargs
    if not isinstance(output, Mapping) or not isinstance(output.get("similar"), bool):
        raise TypeError("Task output must be a mapping with a boolean 'similar' field")

    expected = _expected_similarity(expected_output)
    predicted = output["similar"]
    correct = predicted == expected

    return Evaluation(
        name="similar",
        value=1.0 if correct else 0.0,
        comment=(
            f"Predicted similar={predicted}; expected similar={expected}"
        ),
        metadata={
            "expected_similar": expected,
            "predicted_similar": predicted,
            "cosine_similarity": output.get("cosine_similarity"),
            "threshold_used": output.get("threshold_used"),
        },
    )


def run_similarity_evals(
    *,
    dataset: str = DEFAULT_DATASET,
    threshold: float | None = None,
) -> ExperimentResult:
    """Run the semantic similarity experiment against a Langfuse dataset."""
    if not dataset.strip():
        raise ValueError("Dataset name must not be empty")

    threshold_used = (
        app_config().dedup.cosine_threshold if threshold is None else threshold
    )
    client = langfuse_client()
    langfuse_dataset = client.get_dataset(dataset)
    if not langfuse_dataset.items:
        raise ValueError(
            f"Langfuse dataset {dataset!r} has no items; add labeled news pairs "
            "before running the experiment"
        )

    task = partial(news_similarity_task, threshold=threshold_used)
    embedding_config = app_config().embeddings
    return langfuse_dataset.run_experiment(
        name=f"News similarity threshold={threshold_used:.2f}",
        description=(
            "Evaluate embedding cosine similarity against human-labeled news "
            "duplicate pairs"
        ),
        task=task,
        evaluators=[eval_semantic_similarity],
        metadata={
            "dataset": dataset,
            "threshold": threshold_used,
            "embedding_model": embedding_config.deployment,
            "embedding_dimensions": embedding_config.dimensions,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Langfuse news similarity threshold evaluation."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Langfuse dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold; defaults to config.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_similarity_evals(
            dataset=args.dataset,
            threshold=args.threshold,
        )
        print(result.format())
    finally:
        shutdown_langfuse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
