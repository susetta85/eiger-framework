"""
AVeriTecDataset — BaseDataset implementation over the AVeriTeC
fact-checking benchmark.

Where the data comes from
--------------------------
AVeriTeC (Automated Verification of Textual Claims over Evidence, NeurIPS
2023, CC BY 4.0) is documented in detail in docs/DATASETS.md, section 3.
Per that document, each claim record contains ``claim``/``label``/
``evidence``/``claim_date``/``speaker`` fields, one JSON object per line
(JSONL), split into ``train.jsonl``/``dev.jsonl``/``test.jsonl`` files
under a data directory (default: ``data/averitec/``, matching the
``data/snopes/`` convention already used by SnopesDataset).

Two things distinguish this loader from JSONFixtureDataset/SnopesDataset,
both driven directly by AVeriTeC's own schema rather than by additional
preprocessing:
  1. Only ``label == "Supported"`` records are kept — the same
     verified-true-only philosophy as SnopesDataset's
     ``normalised_rating == True`` filter (see eiger/datasets/snopes.py):
     ``Claim.original_fact`` must be a verified TRUE statement, since
     EIGER generates its own falsehoods via the attack registry rather
     than importing externally-sourced false claims as ground truth.
  2. Unlike Snopes, AVeriTeC claims do NOT need an LLM-generated
     ``context_query``: each record's ``evidence`` list already contains
     real question/answer/url triples produced by the benchmark's human
     annotators, so this loader uses the first evidence question
     directly as ``context_query`` (falling back to a simple templated
     question only for the rare record with no evidence at all). This
     keeps the loader fully offline and dependency-free — no Ollama call
     needed, unlike ``scripts/enrich_snopes_claims.py``.

AVeriTeC provides no stable per-record ``claim_id`` in the fields EIGER
consumes (see the field table in docs/DATASETS.md), so this loader
constructs one deterministically from each record's zero-based position
in the raw split file: ``f"AVERITEC_{index:05d}"``. Because ``index`` is
assigned by enumerating the raw file *before* the Supported-only filter
is applied, a claim's ID stays stable across code changes to the filter
logic (e.g. if a future revision also keeps "Conflicting" records) as
long as the underlying file and the intended semantics of `index` are
unchanged.

What this class does NOT do
----------------------------
- It does not implement automated download() yet: fetching AVeriTeC
  requires the optional HuggingFace ``datasets`` library and network
  access (see docs/DATASETS.md, section 3, for the exact manual `pip
  install datasets` + `load_dataset(...)` steps, or the alternative git-
  clone method). This mirrors JSONFixtureDataset's own stated rationale
  for why AVeriTeC wasn't the first loader implemented: real-corpus
  loaders that need new runtime dependencies and network access are
  deliberately sequenced after the dependency-free ones. download() here
  is a *guard*, not a fetcher: it no-ops if split files are already
  present, and raises a clear, actionable IngestionError (pointing at
  docs/DATASETS.md) if they are not — never a silent no-op that leaves
  a subsequent load() to fail with a confusing bare "file not found".
- It does not deduplicate records — unlike the raw Snopes export (which
  has genuine duplicate claim_ids across rows), AVeriTeC's own train/
  dev/test splits are not documented as containing duplicates.
- It does not independently re-verify AVeriTeC's own "Supported" label;
  like Snopes, claims loaded this way are tagged
  ``metadata["verified"] = False`` until the research team spot-checks
  a sample (see docs/DATASETS.md section 11, "Sprint Roadmap").
- It does not support arbitrary split names beyond what the data
  directory actually contains: ``load(split=...)`` simply looks for
  ``<data_dir>/<split>.jsonl`` and raises IngestionError if that exact
  file is missing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# Matches docs/DATASETS.md's established data/<name>/ convention for
# externally-sourced dataset files that are too large to commit to Git
# (see .gitignore's "Data & storage" section) and SnopesDataset's own
# _DEFAULT_ENRICHED_PATH resolution. Path(__file__) is
# eiger/datasets/averitec.py, so parents[0]=datasets, [1]=eiger,
# [2]=repository root.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "averitec"

# Only this label is treated as a verified-true ground-truth claim — see
# the module docstring's point 1.
_VERIFIED_TRUE_LABEL = "Supported"


class AVeriTecDataset(BaseDataset):
    """
    Loads Claim objects from AVeriTeC split files (Supported-label subset).

    Expected raw JSONL shape (one object per line, see docs/DATASETS.md
    section 3 for the authoritative field table):
        {
          "claim": str,
          "label": str,               # "Supported" rows are kept, others skipped
          "evidence": [{"question": str, "answer": str, "url": str}, ...],
          "claim_date": str,          # optional
          "speaker": str,             # optional
        }

    Field mapping to Claim:
        claim                          -> Claim.original_fact
        evidence[0]["question"]        -> Claim.context_query (templated
                                           fallback if no evidence)
        f"AVERITEC_{index:05d}"        -> Claim.claim_id (see module docstring)
        (this class)                   -> Claim.source_dataset = "averitec"
        label                          -> Claim.metadata["label"]
        claim_date (if present)        -> Claim.metadata["claim_date"]
        speaker (if present)           -> Claim.metadata["speaker"]
        evidence[*]["url"] (if any)    -> Claim.metadata["evidence_urls"]
        (always)                       -> Claim.metadata["verified"] = False
    """

    name: str = "averitec"
    description: str = (
        "AVeriTeC fact-checking claims (Supported-label subset only). "
        "Evidence questions are used as context_query directly — no LLM "
        "enrichment needed. Requires pre-downloaded *.jsonl split files "
        "under data/averitec/; automated download() is not yet "
        "implemented — see docs/DATASETS.md section 3 for manual steps."
    )

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """
        Args:
            data_dir: Optional override of the directory containing
                      AVeriTeC's ``<split>.jsonl`` files. Defaults to
                      ``data/averitec/`` at the repository root.
        """
        self.data_dir: Path = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        # Populated by load(); used by content_hash so the hash reflects
        # whatever was actually loaded rather than re-reading the file.
        self._loaded_claims: list[Claim] = []

    # ─── BaseDataset interface ─────────────────────────────────────────────

    def download(self, target_dir: str) -> None:
        """
        Guard, not a fetcher — see the class/module docstring.

        No-ops (logged at debug level) if at least one ``*.jsonl`` file
        already exists directly under ``target_dir``. Otherwise raises
        IngestionError with the manual download steps from
        docs/DATASETS.md, rather than silently doing nothing and letting
        a later load() fail with a confusing bare "file not found".

        Args:
            target_dir: Directory expected to contain AVeriTeC's
                        ``<split>.jsonl`` files.

        Raises:
            IngestionError: If no ``*.jsonl`` files are present.
        """
        target = Path(target_dir)
        if target.is_dir() and any(target.glob("*.jsonl")):
            log.debug("averitec.download_noop_already_present", target_dir=target_dir)
            return
        log.debug("averitec.download_missing", target_dir=target_dir)
        raise IngestionError(
            f"No AVeriTeC '*.jsonl' split files found under '{target_dir}'. "
            "Automated download is not implemented yet (it requires the "
            "optional HuggingFace 'datasets' library and network access) "
            "— see docs/DATASETS.md, section 3, for the exact manual "
            "download steps (pip install datasets; load_dataset(...), or "
            "git clone the AVeriTeC repository)."
        )

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """
        Parse ``<data_dir>/<split>.jsonl`` and return Claim objects for
        every ``label == "Supported"`` record, in file order.

        Args:
            split:      Selects ``<data_dir>/<split>.jsonl`` (e.g. "test",
                        "dev", "train" — whichever files are actually
                        present under data_dir).
            max_claims: If set, return at most this many claims, taking
                        the first N in file order after filtering (file
                        order is stable and deterministic — no re-sorting
                        is needed).

        Returns:
            List of Claim objects for the Supported-label subset.

        Raises:
            IngestionError: If the split file is missing, unreadable, has
                            invalid JSON on any line, or a Supported
                            record is missing the required "claim" field.
        """
        file_path = self.data_dir / f"{split}.jsonl"
        log.debug("averitec.load_start", path=str(file_path), split=split)
        raw_items = self._read_jsonl(file_path)

        try:
            claims = [
                self._to_claim(index, item)
                for index, item in enumerate(raw_items)
                if item.get("label") == _VERIFIED_TRUE_LABEL
            ]
        except KeyError as exc:
            raise IngestionError(
                f"AVeriTeC split file '{file_path}' has a '{_VERIFIED_TRUE_LABEL}' "
                f"record missing required field {exc}. Expected keys: claim, label."
            ) from exc

        if max_claims is not None:
            claims = claims[:max_claims]

        self._loaded_claims = claims
        log.debug("averitec.load_complete", n_claims=len(claims))
        return claims

    @property
    def content_hash(self) -> str:
        """
        SHA-256 (truncated to 16 hex chars) over the concatenated
        original_fact text of every claim loaded by the most recent
        load() call — same convention as JSONFixtureDataset/SnopesDataset.

        Returns "0" * 16 if load() has not been called yet.
        """
        if not self._loaded_claims:
            return "0" * 16
        combined = "".join(claim.original_fact for claim in self._loaded_claims)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # ─── Internal helpers ──────────────────────────────────────────────────

    def _read_jsonl(self, file_path: Path) -> list[dict[str, Any]]:
        """Read and JSON-decode a JSONL split file, wrapping failures."""
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(
                f"Could not read AVeriTeC split file at '{file_path}': {exc}. "
                "Run dataset.download(...) first, or see docs/DATASETS.md "
                "section 3 for manual download steps."
            ) from exc

        items: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestionError(
                    f"AVeriTeC split file '{file_path}' has invalid JSON "
                    f"on line {line_no}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise IngestionError(
                    f"AVeriTeC split file '{file_path}' line {line_no} must "
                    f"be a JSON object, got {type(record).__name__}."
                )
            items.append(record)
        return items

    def _to_claim(self, index: int, item: dict[str, Any]) -> Claim:
        """Map one raw Supported-label AVeriTeC record to a Claim."""
        evidence = item.get("evidence") or []
        context_query: str | None = None
        if isinstance(evidence, list) and evidence:
            first = evidence[0]
            if isinstance(first, dict) and first.get("question"):
                context_query = str(first["question"])
        if not context_query:
            context_query = f"Is it true that {item['claim']}?"

        metadata: dict[str, Any] = {
            "label": item.get("label", ""),
            "verified": False,
        }
        if item.get("claim_date"):
            metadata["claim_date"] = item["claim_date"]
        if item.get("speaker"):
            metadata["speaker"] = item["speaker"]
        evidence_urls = [
            e["url"] for e in evidence if isinstance(e, dict) and e.get("url")
        ]
        if evidence_urls:
            metadata["evidence_urls"] = evidence_urls

        return Claim(
            claim_id=f"AVERITEC_{index:05d}",
            original_fact=item["claim"],
            context_query=context_query,
            source_dataset=self.name,
            metadata=metadata,
        )
