"""
EIBench Layer 1-3 Pipeline — Quickstart Script

This script demonstrates the first three layers of the EIBench pipeline:
  Layer 1: Corpus loading from the JSON fixture dataset
  Layer 2: Corpus poisoning via the attack engine
  Layer 3: Vector ingestion and retrieval via Qdrant

All configuration is read from environment variables (see .env.example).
Run with: python pipeline_eibench.py [--top-k 3] [--poison-rate 0.5]

For full experiment support, use the CLI:
  python -m eiger run experiments/baseline.yaml
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

log = get_logger(__name__)


def load_json_fixture(path: str | Path) -> list[Claim]:
    """Load claims from a local JSON fixture file."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    claims = []
    for item in raw:
        claims.append(Claim(
            claim_id=item["claim_id"],
            original_fact=item["original_fact"],
            context_query=item["context_query"],
            source_dataset="json_fixture",
        ))
    log.info("dataset.loaded", n_claims=len(claims), source=str(path))
    return claims


def run_pipeline(top_k: int = 3, poison_rate: float = 0.5, seed: int = 42) -> None:
    """End-to-end Layer 1-3 pipeline."""
    settings = get_settings()
    seed_everything(seed)

    # ── Layer 1: Load corpus ──────────────────────────────────────────────────
    fixture_path = Path(__file__).parent / "eibench_raw_claims.json"
    claims = load_json_fixture(fixture_path)

    # ── Layer 2: Build poisoned corpus ────────────────────────────────────────
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
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        log.error("missing_dependency", error=str(exc), hint="pip install qdrant-client sentence-transformers")
        sys.exit(1)

    log.info("qdrant.connecting", host=settings.qdrant_host, port=settings.qdrant_port)
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    log.info("embedder.loading", model=settings.default_embedder)
    encoder = SentenceTransformer(settings.default_embedder)

    collection_name = "eibench_pipeline_demo"

    # Reset collection for a clean run
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    )
    log.info("qdrant.collection_created", collection=collection_name)

    # Encode and upsert all documents
    all_docs = corpus.all_documents
    points = []
    for i, doc in enumerate(all_docs, start=1):
        vector = encoder.encode(doc.text).tolist()
        payload = {
            "text": doc.text,
            "type": doc.doc_type,
            "claim_id": doc.claim_id,
        }
        if hasattr(doc, "attack_name"):
            payload["attack"] = doc.attack_name  # type: ignore[attr-defined]
        points.append(PointStruct(id=i, vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)
    log.info("qdrant.upserted", n_points=len(points))

    # Retrieve for each claim
    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS — EIBench Layer 3")
    print("=" * 60)

    for claim in claims:
        query_vector = encoder.encode(claim.context_query).tolist()
        results = client.query_points(collection_name=collection_name, query=query_vector, limit=top_k)

        print(f"\nQuery: {claim.context_query}")
        print(f"Claim ID: {claim.claim_id}")
        print("-" * 40)
        for rank, hit in enumerate(results.points, start=1):
            p = hit.payload or {}
            score = getattr(hit, "score", 0.0)
            doc_type = p.get("type", "unknown").upper()
            attack = p.get("attack", "N/A")
            text = p.get("text", "")
            print(f"[{rank}] {doc_type} (attack={attack}) score={score:.4f}")
            print(f"    {text[:120]}{'...' if len(text) > 120 else ''}")

    print("\n" + "=" * 60)
    log.info("pipeline.complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="EIBench Layer 1-3 pipeline quickstart")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved documents per query")
    parser.add_argument("--poison-rate", type=float, default=0.5, help="Fraction of corpus to poison [0.0–1.0]")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    configure_logging(args.log_level)
    run_pipeline(top_k=args.top_k, poison_rate=args.poison_rate, seed=args.seed)


if __name__ == "__main__":
    main()
