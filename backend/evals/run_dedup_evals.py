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

DEFAULT_DATASET = "news-duplicate-pairs"


async def detect_duplicate(
    *,
    item: Any,
    threshold: float | None = None,
    dimensions: int | None = None,
    **kwargs: Any,
) -> dict[str, bool]:
    """Embed one dataset pair and classify it using the given threshold."""
    del kwargs
    threshold_used = (
        app_config().dedup.cosine_threshold if threshold is None else threshold
    )
    news_1, news_2 = item.input["news_1"], item.input["news_2"]
    embeddings = await embed_texts(
        [news_1, news_2],
        observe=True,
        dimensions=dimensions,
    )

    if len(embeddings) != 2:
        raise ValueError(
            f"Expected two embeddings for a news pair, received {len(embeddings)}"
        )

    cos_sim = cosine_similarity(embeddings[0], embeddings[1])
    return {"duplicate": cos_sim >= threshold_used}


def _expected_duplicate(expected_output: Any) -> bool:
    if isinstance(expected_output, bool):
        return expected_output
    if isinstance(expected_output, Mapping) and isinstance(
        expected_output.get("duplicate"), bool
    ):
        return expected_output["duplicate"]
    raise TypeError(
        "Dataset expected_output must be a boolean or a mapping with a boolean "
        "'duplicate' field"
    )


def _predicted_duplicate(output: Any) -> bool:
    if not isinstance(output, Mapping) or not isinstance(output.get("duplicate"), bool):
        raise TypeError("Task output must be a mapping with a boolean 'duplicate' field")
    return output["duplicate"]


def eval_duplicate_match(
    *,
    input: Any,
    output: Any,
    expected_output: Any = None,
    **kwargs: Any,
) -> Evaluation:
    """Score whether the threshold-based duplicate call matches the human label."""
    del input, kwargs
    expected = _expected_duplicate(expected_output)
    predicted = _predicted_duplicate(output)
    correct = predicted == expected

    return Evaluation(
        name="duplicate-correct",
        value=1.0 if correct else 0.0,
        data_type="BOOLEAN",
        comment=f"Predicted duplicate={predicted}; expected duplicate={expected}",
        metadata={
            "expected_duplicate": expected,
            "predicted_duplicate": predicted,
        },
    )


def run_dedup_evals(
    *,
    dataset: str = DEFAULT_DATASET,
    threshold: float | None = None,
    dimensions: int | None = None,
) -> ExperimentResult:
    """Run the duplicate-detection experiment against a Langfuse dataset."""
    if not dataset.strip():
        raise ValueError("Dataset name must not be empty")

    config = app_config()
    embedding_config = config.embeddings
    threshold_used = (
        config.dedup.cosine_threshold if threshold is None else threshold
    )
    dimensions_used = (
        embedding_config.dimensions if dimensions is None else dimensions
    )
    if dimensions_used <= 0:
        raise ValueError("Embedding dimensions must be positive")

    client = langfuse_client()
    langfuse_dataset = client.get_dataset(dataset)
    if not langfuse_dataset.items:
        raise ValueError(
            f"Langfuse dataset {dataset!r} has no items; add labeled news pairs "
            "before running the experiment"
        )

    task = partial(
        detect_duplicate,
        threshold=threshold_used,
        dimensions=dimensions_used,
    )
    return langfuse_dataset.run_experiment(
        name=f"dedup-threshold-{threshold_used:.2f}-dimensions-{dimensions_used}",
        description=(
            "Evaluate embedding cosine similarity against human-labeled news "
            "duplicate pairs"
        ),
        task=task,
        evaluators=[eval_duplicate_match],
        metadata={
            "dataset": dataset,
            "threshold": threshold_used,
            "embedding_model": embedding_config.deployment,
            "embedding_dimensions": dimensions_used,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Langfuse news duplicate-detection threshold and "
            "embedding-dimension evaluation."
        )
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
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Embedding dimensions; defaults to config.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_dedup_evals(
            dataset=args.dataset,
            threshold=args.threshold,
            dimensions=args.dimensions,
        )
        print(result.format())
    finally:
        shutdown_langfuse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
