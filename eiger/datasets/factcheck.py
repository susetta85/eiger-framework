"""
FactCheckDataset — BaseDataset implementation over the FactCheck.org
corpus (via the CLEF CheckThat! lab's pre-processed mirror).

Where the data comes from
--------------------------
FactCheck.org has no bulk download API; docs/DATASETS.md section 5
documents fetching a pre-processed mirror from the CLEF CheckThat! lab
instead. Per that section's field table, each record has ``claim_id``/
``claim``/``verdict``/``article_url``/``date`` fields.

A note on an assumption this loader makes: unlike PolitiFact (explicitly
documented as headerless TSV) or AVeriTeC (explicitly documented as
JSONL), docs/DATASETS.md section 5 does not specify a concrete raw file
format for FactCheck.org/CheckThat! — only the field table above. This
loader assumes **JSON Lines** (one JSON object per line, matching
AVeriTeC's format and this project's own ``data/<name>/<split>.jsonl``
convention), since the field table describes a flat record shape with no
documented column ordering. This is an unverified assumption, exactly
like ``PolitiFactDataset``'s caveat about its ``context`` column index:
if the real downloaded CheckThat! mirror turns out to use a different
format, only this loader's parsing method needs to change, not its
public contract.

Two things distinguish this loader from JSONFixtureDataset/SnopesDataset,
both driven by the documented field table:
  1. Only ``verdict == "true"`` records are kept — the same verified-
     true-only philosophy as Snopes/AVeriTeC/PolitiFact (see
     docs/DATASETS.md section 1's Overview and each of those loaders'
     own module docstrings for the full rationale: EIGER generates its
     own falsehoods via the attack registry, it does not import
     externally-sourced false claims as ground truth).
  2. Like PolitiFact (and unlike AVeriTeC), there is no evidence Q&A or
     natural-language question in the documented fields, so
     ``context_query`` is a simple templated fallback — no LLM enrichment
     step required.

Unlike AVeriTeC (which has no per-record id in the documented fields),
FactCheck.org's own ``claim_id`` is used directly (prefixed
``FACTCHECK_``), mirroring SnopesDataset's ``SNOPES_<id>`` and
PolitiFactDataset's ``POLITIFACT_<id>`` conventions.

What this class does NOT do
----------------------------
- It does not implement automated download() yet: fetching the
  CheckThat! mirror requires network access (see docs/DATASETS.md
  section 5). Like AVeriTecDataset/PolitiFactDataset, download() here is
  a *guard*, not a fetcher.
- It does not independently re-verify FactCheck.org's own "true" verdict;
  like the other loaders, claims are tagged
  ``metadata["verified"] = False`` until the research team spot-checks a
  sample.
- It does not scrape FactCheck.org's live search endpoint (docs/DATASETS.md
  section 5 mentions this as an alternative to the CheckThat! mirror) —
  only the pre-processed mirror's file format is supported.
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

# Matches docs/DATASETS.md's established data/<name>/ convention. Path(__file__)
# is eiger/datasets/factcheck.py, so parents[0]=datasets, [1]=eiger,
# [2]=repository root.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "factcheck"

# Only this verdict is treated as a verified-true ground-truth claim —
# see the module docstring's note on the project-wide filter philosophy.
_VERIFIED_TRUE_VERDICT = "true"


class FactCheckDataset(BaseDataset):
    """
    Loads Claim objects from FactCheck.org/CheckThat! split files
    (verified-true subset only).

    Expected raw JSONL shape (one object per line — see the module
    docstring's caveat that this format is not independently re-verified
    against the real downloaded mirror):
        {
          "claim_id": str,
          "claim": str,
          "verdict": str,       # "true" rows are kept, others skipped
          "article_url": str,   # optional
          "date": str,          # optional
        }

    Field mapping to Claim:
        claim                           -> Claim.original_fact
        (templated fallback)            -> Claim.context_query
        f"FACTCHECK_{claim_id}"         -> Claim.claim_id
        (this class)                    -> Claim.source_dataset = "factcheck_org"
        verdict                         -> Claim.metadata["verdict"]
        article_url, date (if present)  -> Claim.metadata["article_url"/"date"]
        (always)                        -> Claim.metadata["verified"] = False
    """

    # Registry name matches docs/DATASETS.md section 5's own
    # "Loading with EIGER" usage snippet (get_dataset("factcheck_org")),
    # even though the module/class are named "factcheck"/"FactCheckDataset"
    # (matching eiger/datasets/README.md's contents table).
    name: str = "factcheck_org"
    description: str = (
        "FactCheck.org fact-checks via the CheckThat! mirror (verified "
        "'true'-verdict subset only). No evidence Q&A available — "
        "context_query is a templated fallback, no LLM required. "
        "Requires pre-downloaded *.jsonl split files under data/factcheck/; "
        "automated download() is not yet implemented — see "
        "docs/DATASETS.md section 5 for manual steps."
    )

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """
        Args:
            data_dir: Optional override of the directory containing
                      ``<split>.jsonl`` files. Defaults to
                      ``data/factcheck/`` at the repository root.
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
            target_dir: Directory expected to contain
                        ``<split>.jsonl`` files.

        Raises:
            IngestionError: If no ``*.jsonl`` files are present.
        """
        target = Path(target_dir)
        if target.is_dir() and any(target.glob("*.jsonl")):
            log.debug("factcheck.download_noop_already_present", target_dir=target_dir)
            return
        log.debug("factcheck.download_missing", target_dir=target_dir)
        raise IngestionError(
            f"No FactCheck.org '*.jsonl' split files found under "
            f"'{target_dir}'. Automated download is not implemented yet "
            "— see docs/DATASETS.md, section 5, for the exact manual "
            "download steps (the CLEF CheckThat! lab mirror)."
        )

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """
        Parse ``<data_dir>/<split>.jsonl`` and return Claim objects for
        every ``verdict == "true"`` record, in file order.

        Args:
            split:      Selects ``<data_dir>/<split>.jsonl``.
            max_claims: If set, return at most this many claims, taking
                        the first N in file order after filtering (file
                        order is stable and deterministic — no re-sorting
                        is needed).

        Returns:
            List of Claim objects for the verified-true subset.

        Raises:
            IngestionError: If the split file is missing, unreadable, has
                            invalid JSON on any line, or a "true"-verdict
                            record is missing a required field.
        """
        file_path = self.data_dir / f"{split}.jsonl"
        log.debug("factcheck.load_start", path=str(file_path), split=split)
        raw_items = self._read_jsonl(file_path)

        try:
            claims = [
                self._to_claim(item)
                for item in raw_items
                if str(item.get("verdict", "")).strip().lower() == _VERIFIED_TRUE_VERDICT
            ]
        except KeyError as exc:
            raise IngestionError(
                f"FactCheck.org split file '{file_path}' has a "
                f"'{_VERIFIED_TRUE_VERDICT}'-verdict record missing "
                f"required field {exc}. Expected keys: claim_id, claim, verdict."
            ) from exc

        if max_claims is not None:
            claims = claims[:max_claims]

        self._loaded_claims = claims
        log.debug("factcheck.load_complete", n_claims=len(claims))
        return claims

    @property
    def content_hash(self) -> str:
        """
        SHA-256 (truncated to 16 hex chars) over the concatenated
        original_fact text of every claim loaded by the most recent
        load() call — same convention as the other dataset loaders.

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
                f"Could not read FactCheck.org split file at '{file_path}': "
                f"{exc}. Run dataset.download(...) first, or see "
                "docs/DATASETS.md section 5 for manual download steps."
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
                    f"FactCheck.org split file '{file_path}' has invalid "
                    f"JSON on line {line_no}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise IngestionError(
                    f"FactCheck.org split file '{file_path}' line {line_no} "
                    f"must be a JSON object, got {type(record).__name__}."
                )
            items.append(record)
        return items

    def _to_claim(self, item: dict[str, Any]) -> Claim:
        """Map one raw verified-true FactCheck.org record to a Claim."""
        claim_text = item["claim"]
        context_query = f"Is it true that {claim_text}?"

        metadata: dict[str, Any] = {
            "verdict": item.get("verdict", ""),
            "verified": False,
        }
        if item.get("article_url"):
            metadata["article_url"] = item["article_url"]
        if item.get("date"):
            metadata["date"] = item["date"]

        return Claim(
            claim_id=f"FACTCHECK_{item['claim_id']}",
            original_fact=claim_text,
            context_query=context_query,
            source_dataset=self.name,
            metadata=metadata,
        )
