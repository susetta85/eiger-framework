"""
PolitiFactDataset — BaseDataset implementation over the LIAR benchmark
(PolitiFact statements, Wang 2017).

Where the data comes from
--------------------------
LIAR (docs/DATASETS.md, section 4) ships as plain TSV files with NO header
row, one statement per line, split into ``train.tsv``/``test.tsv``/
``valid.tsv`` under a data directory (default: ``data/politifact/``,
matching the ``data/snopes/``/``data/averitec/`` convention). Per this
project's own documented field table, the columns relevant to EIGER are:

    0: id, 1: label, 2: statement, 3: subject, 4: speaker,
    5: job_title, 8: context

``label`` is one of six truth ratings: ``pants-fire``, ``false``,
``barely-true``, ``half-true``, ``mostly-true``, ``true``.

A note on a documentation inconsistency this loader deliberately does NOT
follow: docs/DATASETS.md section 4's own "Loading with EIGER" example
comment says to filter for ``label in {"false", "pants-fire"}`` "for use
as adversarial ground truth". That contradicts the project's actual,
repeatedly-documented and already-implemented philosophy (see this
module's own filter below, ``eiger/datasets/snopes.py``'s
``normalised_rating == True`` filter, ``eiger/datasets/averitec.py``'s
``label == "Supported"`` filter, and docs/DATASETS.md section 1's
Overview): ``Claim.original_fact`` must be a verified TRUE statement —
EIGER generates its own falsehoods via the attack registry, it does not
import externally-sourced false claims as ground truth. That stale
example comment predates Snopes/AVeriTeC's implementation and should be
corrected in a follow-up docs pass; this loader follows the philosophy
actually implemented everywhere else, not the stale comment.

Two things distinguish this loader from JSONFixtureDataset/SnopesDataset,
both driven by LIAR's own schema:
  1. Only ``label == "true"`` records are kept — the strictest of the six
     ratings, matching Snopes'/AVeriTeC's verified-true-only philosophy
     (see the note above).
  2. LIAR has no evidence Q&A pairs (unlike AVeriTeC) and no natural-
     language question at all (like Snopes). Unlike Snopes, this loader
     does not require a separate LLM-enrichment script: it falls back to
     a simple templated question, same as AVeriTecDataset's fallback for
     evidence-less records. This keeps the loader offline and dependency
     -free; a future ``scripts/enrich_politifact_claims.py`` (mirroring
     ``scripts/enrich_snopes_claims.py``) could improve context_query
     quality later without changing this class's contract.

What this class does NOT do
----------------------------
- It does not implement automated download() yet: fetching LIAR requires
  network access to fetch and unzip
  ``https://www.cs.ucsb.edu/~william/data/liar_dataset.zip`` (see
  docs/DATASETS.md section 4). Like AVeriTecDataset, download() here is a
  *guard*, not a fetcher: it no-ops if split files already exist, and
  raises a clear, actionable IngestionError pointing at the docs
  otherwise.
- It does not independently re-verify PolitiFact's own "true" rating;
  like Snopes/AVeriTeC, claims loaded this way are tagged
  ``metadata["verified"] = False`` until the research team spot-checks a
  sample.
- It does not assume the real downloaded LIAR TSV has 14 columns with
  ``context`` truly at index 8 — that index comes from this project's own
  documented field table (docs/DATASETS.md section 4), not independently
  re-verified against the actual upstream file. If ``context`` ends up
  misaligned once real data is downloaded, only the column index constant
  below needs correcting, not this class's structure. To stay robust to
  that uncertainty, only ``id``/``label``/``statement`` (columns 0-2) are
  strictly required per row; ``subject``/``speaker``/``job_title``/
  ``context`` are all read defensively and simply omitted from
  ``Claim.metadata`` if the row is too short to contain them.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# Matches docs/DATASETS.md's established data/<name>/ convention. Path(__file__)
# is eiger/datasets/politifact.py, so parents[0]=datasets, [1]=eiger,
# [2]=repository root.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "politifact"

# Only this label is treated as a verified-true ground-truth claim — see
# the module docstring's note on why this differs from docs/DATASETS.md
# section 4's own (stale) example comment.
_VERIFIED_TRUE_LABEL = "true"

# Column indices per docs/DATASETS.md section 4's field table. See the
# module docstring's caveat: `_CONTEXT_COLUMN` in particular is not
# independently re-verified against a real downloaded LIAR file.
_ID_COLUMN = 0
_LABEL_COLUMN = 1
_STATEMENT_COLUMN = 2
_SUBJECT_COLUMN = 3
_SPEAKER_COLUMN = 4
_JOB_TITLE_COLUMN = 5
_CONTEXT_COLUMN = 8
_MIN_REQUIRED_COLUMNS = _STATEMENT_COLUMN + 1  # id, label, statement


class PolitiFactDataset(BaseDataset):
    """
    Loads Claim objects from LIAR split files (verified-true subset only).

    Expected raw TSV shape (no header row, tab-separated, see the module
    docstring for the authoritative column table and its caveats):
        id \\t label \\t statement \\t subject \\t speaker \\t job_title \\t ... \\t context \\t ...

    Field mapping to Claim:
        statement (col 2)               -> Claim.original_fact
        (templated fallback)             -> Claim.context_query
        f"POLITIFACT_{id}" (col 0)       -> Claim.claim_id
        (this class)                     -> Claim.source_dataset = "politifact"
        label (col 1)                    -> Claim.metadata["label"]
        subject, speaker, job_title,
            context (cols 3/4/5/8, each if present and non-empty) -> Claim.metadata[...]
        (always)                         -> Claim.metadata["verified"] = False
    """

    name: str = "politifact"
    description: str = (
        "PolitiFact/LIAR statements (verified 'true'-label subset only). "
        "No evidence Q&A available — context_query is a templated "
        "fallback, no LLM required. Requires pre-downloaded *.tsv split "
        "files under data/politifact/; automated download() is not yet "
        "implemented — see docs/DATASETS.md section 4 for manual steps."
    )

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """
        Args:
            data_dir: Optional override of the directory containing LIAR's
                      ``<split>.tsv`` files. Defaults to
                      ``data/politifact/`` at the repository root.
        """
        self.data_dir: Path = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        # Populated by load(); used by content_hash so the hash reflects
        # whatever was actually loaded rather than re-reading the file.
        self._loaded_claims: list[Claim] = []

    # ─── BaseDataset interface ─────────────────────────────────────────────

    def download(self, target_dir: str) -> None:
        """
        Guard, not a fetcher — see the class/module docstring.

        No-ops (logged at debug level) if at least one ``*.tsv`` file
        already exists directly under ``target_dir``. Otherwise raises
        IngestionError with the manual download steps from
        docs/DATASETS.md, rather than silently doing nothing and letting
        a later load() fail with a confusing bare "file not found".

        Args:
            target_dir: Directory expected to contain LIAR's
                        ``<split>.tsv`` files.

        Raises:
            IngestionError: If no ``*.tsv`` files are present.
        """
        target = Path(target_dir)
        if target.is_dir() and any(target.glob("*.tsv")):
            log.debug("politifact.download_noop_already_present", target_dir=target_dir)
            return
        log.debug("politifact.download_missing", target_dir=target_dir)
        raise IngestionError(
            f"No LIAR '*.tsv' split files found under '{target_dir}'. "
            "Automated download is not implemented yet — see "
            "docs/DATASETS.md, section 4, for the exact manual download "
            "steps (wget the liar_dataset.zip archive and unzip it)."
        )

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """
        Parse ``<data_dir>/<split>.tsv`` and return Claim objects for
        every ``label == "true"`` record, in file order.

        Args:
            split:      Selects ``<data_dir>/<split>.tsv`` (e.g. "test",
                        "valid", "train" — LIAR's own standard split names).
            max_claims: If set, return at most this many claims, taking
                        the first N in file order after filtering (file
                        order is stable and deterministic — no re-sorting
                        is needed).

        Returns:
            List of Claim objects for the verified-true subset.

        Raises:
            IngestionError: If the split file is missing, unreadable, or
                            a row has fewer than the 3 required columns
                            (id, label, statement).
        """
        file_path = self.data_dir / f"{split}.tsv"
        log.debug("politifact.load_start", path=str(file_path), split=split)
        rows = self._read_tsv(file_path)

        claims = [
            self._to_claim(row)
            for row in rows
            if row[_LABEL_COLUMN].strip().lower() == _VERIFIED_TRUE_LABEL
        ]

        if max_claims is not None:
            claims = claims[:max_claims]

        self._loaded_claims = claims
        log.debug("politifact.load_complete", n_claims=len(claims))
        return claims

    @property
    def content_hash(self) -> str:
        """
        SHA-256 (truncated to 16 hex chars) over the concatenated
        original_fact text of every claim loaded by the most recent
        load() call — same convention as JSONFixtureDataset/SnopesDataset/
        AVeriTecDataset.

        Returns "0" * 16 if load() has not been called yet.
        """
        if not self._loaded_claims:
            return "0" * 16
        combined = "".join(claim.original_fact for claim in self._loaded_claims)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # ─── Internal helpers ──────────────────────────────────────────────────

    def _read_tsv(self, file_path: Path) -> list[list[str]]:
        """
        Read a LIAR-format TSV file (no header, no quote-escaping) and
        return one list of column strings per non-blank row.
        """
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(
                f"Could not read PolitiFact/LIAR split file at '{file_path}': "
                f"{exc}. Run dataset.download(...) first, or see "
                "docs/DATASETS.md section 4 for manual download steps."
            ) from exc

        rows: list[list[str]] = []
        # QUOTE_NONE: LIAR statements may contain literal quote characters
        # that would otherwise be misinterpreted as CSV quoting.
        reader = csv.reader(
            text.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE
        )
        for line_no, row in enumerate(reader, start=1):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue  # skip blank lines
            if len(row) < _MIN_REQUIRED_COLUMNS:
                raise IngestionError(
                    f"PolitiFact/LIAR split file '{file_path}' line {line_no} "
                    f"has only {len(row)} column(s), expected at least "
                    f"{_MIN_REQUIRED_COLUMNS} (id, label, statement)."
                )
            rows.append(row)
        return rows

    def _to_claim(self, row: list[str]) -> Claim:
        """Map one raw verified-true LIAR row to a Claim."""
        raw_id = row[_ID_COLUMN].strip()
        # LIAR's own ids are conventionally suffixed ".json" (e.g.
        # "11972.json"); stripped for a cleaner claim_id, mirroring
        # SnopesDataset's "SNOPES_<id>" prefix convention.
        if raw_id.endswith(".json"):
            raw_id = raw_id[: -len(".json")]

        statement = row[_STATEMENT_COLUMN].strip()
        context_query = f"Is it true that {statement}?"

        metadata: dict[str, Any] = {
            "label": row[_LABEL_COLUMN].strip(),
            "verified": False,
        }
        for column, key in (
            (_SUBJECT_COLUMN, "subject"),
            (_SPEAKER_COLUMN, "speaker"),
            (_JOB_TITLE_COLUMN, "job_title"),
            (_CONTEXT_COLUMN, "context"),
        ):
            if column < len(row) and row[column].strip():
                metadata[key] = row[column].strip()

        return Claim(
            claim_id=f"POLITIFACT_{raw_id}",
            original_fact=statement,
            context_query=context_query,
            source_dataset=self.name,
            metadata=metadata,
        )
