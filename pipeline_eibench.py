"""
EIBench Layer 1-3 Pipeline — Quickstart Script

This script demonstrates the first three layers of the EIBench pipeline:
  Layer 1: Corpus loading from the JSON fixture dataset
  Layer 2: Corpus poisoning via the attack engine
  Layer 3: Vector ingestion and retrieval via Qdrant

All configuration is read from environment variables (see .env.example).
Run with: python pipeline_eibench.py [--top-k 3] [--poison-rate 0.5]

For full experiment support (Layers 4-5: generation + evaluation), use the CLI:
  python -m eiger run experiments/baseline.yaml

What this script does NOT do:
  - Run LLM generation or compute evaluation metrics (that is Layer 4/5).
  - Persist results to disk; output is printed to stdout for quick inspection.
  - Support distributed/parallel execution; it is a single-process quickstart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eiger.attacks import NumericalShiftAttack, AttributionSwitchAttack
from eiger.config import get_settings
from eiger.core.models import AttackConfig, Claim, Document
from eiger.ingestion.corpus_builder import CorpusBuilder
from eiger.utils.logging import configure_logging, get_logger
from eiger.utils.seeding import seed_everything

# Module-level logger; structured log entries are emitted via structlog.
log = get_logger(__name__)


# ─── Layer 1: Dataset loading ─────────────────────────────────────────────────

def load_json_fixture(path: str | Path) -> list[Claim]:
    """
    Load claims from a local JSON fixture file and parse them into Claim objects.

    The fixture file is expected to be a JSON array of objects with the keys:
      - ``claim_id``      (str): unique identifier for the claim
      - ``original_fact`` (str): the factual statement to be tested
      - ``context_query`` (str): the query used to retrieve supporting documents

    This function is a lightweight alternative to the full dataset loader used
    in production experiments. It is intentionally simple: no pagination,
    no schema validation beyond what Claim's Pydantic model enforces.

    Args:
        path: Absolute or relative path to the JSON fixture file.

    Returns:
        List of ``Claim`` objects parsed from the fixture.

    Raises:
        FileNotFoundError: If the fixture file does not exist at ``path``.
        json.JSONDecodeError: If the file content is not valid JSON.
        KeyError: If a required field is missing from a claim entry.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    claims = []
    for item in raw:
        claims.append(Claim(
            claim_id=item["claim_id"],
            original_fact=item["original_fact"],
            context_query=item["context_query"],
            # Hard-code the source tag so downstream components can identify
            # that this data came from the local fixture rather than a live API.
            source_dataset="json_fixture",
        ))

    log.info("dataset.loaded", n_claims=len(claims), source=str(path))
    return claims


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(top_k: int = 3, poison_rate: float = 0.5, seed: int = 42) -> None:
    """
    Execute the full Layer 1-3 EIBench pipeline end-to-end.

    Steps:
      1. Load claims from the local JSON fixture.
      2. Build a poisoned corpus using NumericalShiftAttack and AttributionSwitchAttack.
      3. Connect to Qdrant, create a fresh collection, encode and upsert all documents.
      4. Run a retrieval query for each claim and print the top-k results.

    Args:
        top_k:       Number of documents to retrieve per query. Defaults to 3.
        poison_rate: Fraction of documents to poison (0.0 = clean, 1.0 = all
                     poisoned). Defaults to 0.5.
        seed:        Random seed for reproducible poisoning. Defaults to 42.

    Raises:
        SystemExit: If ``qdrant-client`` or ``sentence-transformers`` are not
                    installed (exits with code 1 and prints an install hint).
    """
    settings = get_settings()
    # Seed all random sources early so that every step downstream is reproducible.
    seed_everything(seed)

    # ── Layer 1: Load corpus ──────────────────────────────────────────────────
    # The fixture file is co-located with this script in the project root.
    fixture_path = Path(__file__).parent / "eibench_raw_claims.json"
    claims = load_json_fixture(fixture_path)

    # ── Layer 2: Build poisoned corpus ────────────────────────────────────────
    # Each attack is paired with an AttackConfig that specifies its poison_rate.
    # Both attacks run on the same corpus; the CorpusBuilder decides which
    # documents each attack is applied to based on the seed and poison_rate.
    attacks = [
        (NumericalShiftAttack(), AttackConfig(name="numerical_shift", poison_rate=poison_rate)),
        (AttributionSwitchAttack(), AttackConfig(name="attribution_switch", poison_rate=poison_rate)),
    ]
    builder = CorpusBuilder(attacks=attacks, seed=seed)
    corpus = builder.build(claims)

    log.info(
        "corpus.summary",
        ground_truth=len(corpus.ground_truth_docs),
        poisoned=len(corpus.poisoned_docs),
        total=len(corpus.all_documents),
    )

    # ── Layer 3: Ingest into Qdrant + retrieve ────────────────────────────────
    # These imports are deferred because qdrant-client and sentence-transformers
    # are optional dependencies not guaranteed to be present in all environments.
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        log.error(
            "missing_dependency",
            error=str(exc),
            hint="pip install qdrant-client sentence-transformers",
        )
        sys.exit(1)

    # Connect to Qdrant using host/port from environment configuration.
    log.info("qdrant.connecting", host=settings.qdrant_host, port=settings.qdrant_port)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Load the sentence encoder; model is downloaded on first use and cached.
    log.info("embedder.loading", model=settings.default_embedder)
    encoder = SentenceTransformer(settings.default_embedder)

    # Use a fixed, descriptive collection name to make Qdrant inspection easy.
    collection_name = "eibench_pipeline_demo"

    # ── Collection setup ─────────────────────────────────────────────────
    # Always start with a clean collection so that re-running the script
    # does not accumulate duplicate vectors from previous runs.
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            # Dimension is model-dependent; read it from the encoder at runtime
            # rather than hard-coding so the script adapts to model changes.
            size=encoder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )
    log.info("qdrant.collection_created", collection=collection_name)

    # ── Encode and upsert all documents ──────────────────────────────────
    all_docs = corpus.all_documents
    points = []
    for i, doc in enumerate(all_docs, start=1):
        # Qdrant requires integer IDs starting from 1; we use the enumeration
        # index as a stable surrogate key within this demo run.
        vector = encoder.encode(doc.text).tolist()
        payload = {
            "text": doc.text,
            "type": doc.doc_type,
            "claim_id": doc.claim_id,
        }
        # PoisonedDocument carries an attack_name attribute; include it in the
        # payload so retrieval results can show which attack produced each hit.
        if hasattr(doc, "attack_name"):
            payload["attack"] = doc.attack_name  # type: ignore[attr-defined]

        points.append(PointStruct(id=i, vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)
    log.info("qdrant.upserted", n_points=len(points))

    # ── Retrieve and print results ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS — EIBench Layer 3")
    print("=" * 60)

    for claim in claims:
        # Encode the claim's context_query to obtain the search vector.
        query_vector = encoder.encode(claim.context_query).tolist()
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
        )

        print(f"\nQuery: {claim.context_query}")
        print(f"Claim ID: {claim.claim_id}")
        print("-" * 40)

        for rank, hit in enumerate(results.points, start=1):
            p = hit.payload or {}
            score = getattr(hit, "score", 0.0)
            doc_type = p.get("type", "unknown").upper()
            attack = p.get("attack", "N/A")
            text = p.get("text", "")
            # Truncate long document text to 120 characters for readability.
            print(f"[{rank}] {doc_type} (attack={attack}) score={score:.4f}")
            print(f"    {text[:120]}{'...' if len(text) > 120 else ''}")

    print("\n" + "=" * 60)
    log.info("pipeline.complete")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse command-line arguments and launch the pipeline.

    Exposed as the ``__main__`` entry point so the script can be run directly
    (``python pipeline_eibench.py``) or via ``python -m pipeline_eibench``.
    """
    parser = argparse.ArgumentParser(description="EIBench Layer 1-3 pipeline quickstart")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved documents per query",
    )
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.5,
        help="Fraction of corpus to poison [0.0–1.0]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    # Configure structured logging before any log calls in run_pipeline.
    configure_logging(args.log_level)
    run_pipeline(top_k=args.top_k, poison_rate=args.poison_rate, seed=args.seed)


if __name__ == "__main__":
    main()
