"""
CorpusClaimDataset — BaseDataset implementation over the project's
human-curated, Mistral/Ollama-generated claim corpus.

Where the data comes from
--------------------------
A research collaborator maintains ``Corpus_claim_RAG_Mistral_output_v3.xlsx``:
a 5,672-row, fully-processed claim corpus (of 21,804 raw candidates), built
by her own Mistral/Ollama pipeline, distinct from and parallel to the four
loaders elsewhere in this package (which all pull from public fact-checking
exports and feed EIGER's own mechanical attack registry). See
docs/CLAIM_AND_RESEARCH_QUESTIONS.md §7 for the full two-pipeline picture
and the "EIGER × Corpus Claim — Protocollo di allineamento" document shared
with her for the data contract this loader implements.

Unlike this project's other four loaders, this file needs no separate
filter/enrich preprocessing script: the workbook's ``01_Corpus_claim`` sheet
is already the final, fully-processed corpus (the ``00_Riepilogo`` sheet
title states it is the "final corpus, without blocked rows" — this loader
nonetheless still defensively re-checks ``preflight_status != "blocked"``
itself; see ``load()``'s docstring for why this doesn't fully hold in the
live file).

Why this loader is a direct schema match, not a reinterpretation
-------------------------------------------------------------------
The workbook's own ``normalized_label`` column already uses exactly
``"verified_true"``/``"verified_false"`` — the same two values as
``eiger.core.models.GroundTruthLabel`` — and ``risk_level``/
``sensitivity_class`` already use this project's own 1-5 / S0-S3 scales.
This loader maps them straight across with no reinterpretation, which is
itself a real (and reassuring) confirmation that the two-axis
ground_truth_label/manipulation_status schema proposed in the alignment
document matches what the collaborator's pipeline already produces.

The ethical gate this loader enforces
----------------------------------------
Every one of the 5,672 rows currently has ``requires_human_review = True``
— none of this corpus has been human-validated yet (see
docs/CLAIM_AND_RESEARCH_QUESTIONS.md §7, point 4). Mirroring this project's
``excluded_from_benchmark``/``allow_non_benchmark_attacks`` gating pattern
(``eiger/core/interfaces.py``, ``eiger/ingestion/corpus_builder.py``),
``load()`` defaults to returning **zero claims** rather than silently
treating unreviewed rows as usable: ``include_unreviewed=True`` is an
explicit, deliberate opt-in for non-paper engineering work only. This is
not a bug or a placeholder — it is this loader's way of encoding "this
corpus is not yet validated" directly in code, so a paper-facing experiment
config can never accidentally consume unreviewed rows.

What this class does NOT do
----------------------------
- It does not consume ``modified_claim`` (the collaborator's own
  Mistral-generated poisoned variant) as a poisoned Document — EIGER's
  ``CorpusBuilder``/attack registry generates poisoning mechanically from
  ``claim_original`` for every loader in this package, this one included,
  so that a single, consistent, seed-derived poisoning mechanism produces
  every experiment's manipulated documents. ``modified_claim`` and its
  full provenance (``manipulation_type_applied``, ``claim_change_description``,
  etc.) are carried into ``Claim.metadata`` for inspection/traceability,
  not fed into the corpus as ground truth or as a pre-made attack.
- It does not implement ``download()`` as a fetcher: this is a
  human-curated, non-public corpus with no automated source to fetch from.
- It does not perform the human review itself — see the ethical gate above.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# Matches docs/DATASETS.md's established data/<name>/ convention. Path(__file__)
# is eiger/datasets/corpus_claim.py, so parents[0]=datasets, [1]=eiger,
# [2]=repository root. Unlike the other four loaders' source files, this
# workbook is not itself downloadable — the caller must copy or symlink it
# here (or pass an explicit path=...); see the class docstring.
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "corpus_claim" / "Corpus_claim_RAG_Mistral_output_v3.xlsx"
)

_SHEET_NAME = "01_Corpus_claim"

# See GroundTruthLabel (eiger/core/models.py): the workbook's own
# normalized_label column already uses these exact two values.
_VALID_GROUND_TRUTH_LABELS = {"verified_true", "verified_false"}

# Rows with this preflight_status are excluded unconditionally (not gated
# behind include_unreviewed): the workbook's own 00_Riepilogo sheet states
# this corpus is meant to be "senza righe blocked" (without blocked rows),
# but a live spot-check found 5 preflight_status="blocked" rows still
# present in 01_Corpus_claim — this loader enforces that exclusion itself
# rather than trusting the workbook's title.
_BLOCKED_PREFLIGHT_STATUS = "blocked"

# docs/ETHICS_AND_THREAT_MODEL.md §5: S3 ("excluded") must never enter the
# corpus at all — unlike S0-S2, it is not a usable-with-caution
# classification, it is "keep this claim out entirely". No row in the file
# inspected while writing this loader was S3 (only S1/S2 appear today), but
# nothing about the schema prevents a future export from including one, and
# this loader is the first one to populate sensitivity_class from real,
# external data at all — so this exclusion is enforced here rather than
# assumed. Excluded unconditionally, the same way blocked preflight_status
# rows are, regardless of include_unreviewed.
_EXCLUDED_SENSITIVITY_CLASS = "S3"

# Columns carried into Claim.metadata verbatim when present and non-None —
# provenance from the collaborator's own pipeline, kept for inspection and
# traceability but not consumed as ground truth or as a pre-made attack
# (see the module docstring's "What this class does NOT do").
_PROVENANCE_METADATA_COLUMNS = (
    "source_platform",
    "factcheck_url",
    "original_rating",
    "topic",
    "subtopic",
    "review_status",
    "requires_human_review",
    "preflight_status",
    "generation_status",
    "manipulation_type_applied",
    "modified_claim",
    "modified_verdict_normalized",
    "changed_element",
    "claim_change_description",
    "pipeline_safety_flags",
    # Bug fix (found during the multilingual-scoping investigation,
    # docs/CLAIM_AND_RESEARCH_QUESTIONS.md §7): the raw workbook has an
    # original_language_iso column (36/5,672 rows tagged non-English, 17 of
    # them genuinely Spanish-language claims from the PolitiFact/"PFF_"
    # source), but it was never in this tuple, so it was silently dropped at
    # load time — nothing downstream (including the new engineering pilot
    # and calibration scripts) could tell a non-English claim apart from an
    # English one. Now carried through to Claim.metadata like every other
    # provenance column.
    "original_language_iso",
)


class CorpusClaimDataset(BaseDataset):
    """
    Loads Claim objects from the collaborator's Mistral-generated claim
    corpus (``Corpus_claim_RAG_Mistral_output_v3.xlsx``, sheet
    ``01_Corpus_claim``).

    Field mapping to Claim:
        claim_original                          -> Claim.original_fact
        (templated fallback, no natural question
         column in this workbook)                -> Claim.context_query
        item_id (e.g. "SNP_000002", used as-is)  -> Claim.claim_id
        (this class)                              -> Claim.source_dataset = "corpus_claim"
        normalized_label                          -> Claim.ground_truth_label
        risk_level                                 -> Claim.risk_level (already 1-5)
        sensitivity_class                          -> Claim.sensitivity_class (already S0-S3)
        source_platform, factcheck_url, original_rating, topic, subtopic,
        review_status, requires_human_review, preflight_status,
        generation_status, manipulation_type_applied, modified_claim,
        modified_verdict_normalized, changed_element,
        claim_change_description, pipeline_safety_flags
            (each, if present and non-None)        -> the same key under Claim.metadata

    See the module docstring for why ``requires_human_review`` gates
    ``load()``'s default output to empty, and for why ``modified_claim``
    is metadata only, not a consumed poisoned document.
    """

    name: str = "corpus_claim"
    description: str = (
        "Human-curated, Mistral-generated claim corpus (5,672 rows, "
        "verified_true/verified_false already labeled). requires_human_review "
        "is True for every row today — load() returns zero claims unless "
        "include_unreviewed=True is passed explicitly (engineering use "
        "only, not paper-facing). Requires the workbook to be copied or "
        "symlinked to data/corpus_claim/, or an explicit path=... override."
    )

    def __init__(self, path: str | Path | None = None) -> None:
        """
        Args:
            path: Optional override of the workbook's location. Defaults
                  to ``data/corpus_claim/Corpus_claim_RAG_Mistral_output_v3.xlsx``
                  at the repository root.
        """
        self.path: Path = Path(path) if path is not None else _DEFAULT_PATH
        # Populated by load(); used by content_hash so the hash reflects
        # whatever was actually loaded rather than re-reading the file.
        self._loaded_claims: list[Claim] = []

    # ─── BaseDataset interface ─────────────────────────────────────────────

    def download(self, target_dir: str) -> None:
        """
        Guard, not a fetcher — this is a human-curated corpus with no
        automated source (see the module docstring). No-ops (logged at
        debug level) if the workbook already exists directly under
        ``target_dir``; otherwise raises IngestionError with instructions.

        Args:
            target_dir: Directory expected to contain the workbook.

        Raises:
            IngestionError: If the workbook is not present under target_dir.
        """
        target = Path(target_dir) / "Corpus_claim_RAG_Mistral_output_v3.xlsx"
        if target.is_file():
            log.debug("corpus_claim.download_noop_already_present", target_dir=target_dir)
            return
        log.debug("corpus_claim.download_missing", target_dir=target_dir)
        raise IngestionError(
            f"'{target}' not found. This is a human-curated corpus with no "
            "automated download — copy or symlink "
            "Corpus_claim_RAG_Mistral_output_v3.xlsx into this directory, "
            "or pass an explicit path=... to CorpusClaimDataset()."
        )

    def load(
        self,
        split: str = "test",
        max_claims: int | None = None,
        include_unreviewed: bool = False,
    ) -> list[Claim]:
        """
        Parse the ``01_Corpus_claim`` sheet and return Claim objects.

        Args:
            split:      Accepted but ignored — the workbook has no train/
                        dev/test partitioning (this loader's rows are
                        already the collaborator's own "final" set).
            max_claims: If set, return at most this many claims, taking
                        the first N in sheet-row order after filtering.
            include_unreviewed: Default False. Every row in the workbook
                        currently has requires_human_review=True — with
                        the default, load() returns an empty list rather
                        than silently treating unreviewed rows as usable
                        (see the module docstring's "ethical gate"
                        section). Pass True only for non-paper engineering
                        work; every returned Claim still carries
                        Claim.metadata["requires_human_review"] so callers
                        can filter further themselves.

        Returns:
            List of Claim objects. Rows with preflight_status="blocked" or
            sensitivity_class="S3" are always excluded, regardless of
            include_unreviewed (see _EXCLUDED_SENSITIVITY_CLASS's own
            comment). Blank rows (no item_id or no claim_original) are
            skipped, matching PolitiFactDataset's own blank-row handling.

        Raises:
            IngestionError: If the workbook is missing, unreadable/corrupt,
                            the expected sheet is not present, or the
                            header row has duplicate column names. A
                            workbook missing the item_id/claim_original
                            *columns* entirely does not raise — every row
                            degrades to "blank" (see Returns above) and
                            load() returns an empty list.
        """
        rows = self._read_rows()

        claims: list[Claim] = []
        for row in rows:
            # This blank/missing-value check is also what makes
            # _to_claim's own row["item_id"]/row["claim_original"] direct
            # indexing safe: every row reaching _to_claim below is
            # guaranteed to have both, non-empty — a column missing from
            # the header entirely reads back as None via .get() here too,
            # so a mis-headed workbook degrades to an empty result rather
            # than a raised error.
            if not row.get("item_id") or not row.get("claim_original"):
                continue
            if row.get("preflight_status") == _BLOCKED_PREFLIGHT_STATUS:
                continue
            if row.get("sensitivity_class") == _EXCLUDED_SENSITIVITY_CLASS:
                continue
            if not include_unreviewed and row.get("requires_human_review") is not False:
                continue
            claims.append(self._to_claim(row))

        if max_claims is not None:
            claims = claims[:max_claims]

        self._loaded_claims = claims
        log.debug(
            "corpus_claim.load_complete",
            n_claims=len(claims),
            include_unreviewed=include_unreviewed,
        )
        return claims

    @property
    def content_hash(self) -> str:
        """
        SHA-256 (truncated to 16 hex chars) over the concatenated
        original_fact text of every claim loaded by the most recent
        load() call — same convention as every other loader in this
        package.

        Returns "0" * 16 if load() has not been called yet.
        """
        if not self._loaded_claims:
            return "0" * 16
        combined = "".join(claim.original_fact for claim in self._loaded_claims)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # ─── Internal helpers ──────────────────────────────────────────────────

    def _read_rows(self) -> list[dict[str, Any]]:
        """Read the 01_Corpus_claim sheet into a list of header-keyed dicts."""
        try:
            import openpyxl
            from openpyxl.utils.exceptions import InvalidFileException
            from zipfile import BadZipFile
        except ImportError as exc:
            raise IngestionError(
                "openpyxl is required to load CorpusClaimDataset. Install "
                "it with: pip install 'eiger[data-import]' (or: pip install "
                "openpyxl)."
            ) from exc

        try:
            workbook = openpyxl.load_workbook(self.path, data_only=True, read_only=True)
        except FileNotFoundError as exc:
            raise IngestionError(
                f"Could not read corpus claim workbook at '{self.path}': {exc}. "
                "Run dataset.download(...) first, or see the class "
                "docstring for manual setup steps."
            ) from exc
        except (OSError, BadZipFile, InvalidFileException) as exc:
            # Covers a corrupt/non-xlsx file (zipfile.BadZipFile) and
            # openpyxl's own InvalidFileException for a path that exists but
            # isn't a readable workbook — e.g. a directory, or a plain .csv
            # renamed to .xlsx. Confirmed by testing directly: neither
            # exception is an OSError subclass, so both are listed
            # explicitly rather than assumed to be covered by OSError alone.
            # FileNotFoundError (an OSError subclass) is a distinct, more
            # specific case already handled above.
            raise IngestionError(
                f"Could not read corpus claim workbook at '{self.path}': {exc}. "
                "The file may be corrupt or not a valid .xlsx workbook."
            ) from exc

        try:
            if _SHEET_NAME not in workbook.sheetnames:
                raise IngestionError(
                    f"Workbook '{self.path}' has no sheet named '{_SHEET_NAME}'. "
                    f"Found: {workbook.sheetnames}"
                )
            sheet = workbook[_SHEET_NAME]

            rows_iter = sheet.iter_rows(values_only=True)
            try:
                header = next(rows_iter)
            except StopIteration:
                return []

            header_names = [str(name) for name in header if name is not None]
            duplicates = {name for name in header_names if header_names.count(name) > 1}
            if duplicates:
                raise IngestionError(
                    f"Workbook '{self.path}' sheet '{_SHEET_NAME}' has "
                    f"duplicate column name(s) {sorted(duplicates)} in its "
                    "header row — cannot map unambiguously to Claim fields."
                )
            header_index = {str(name): idx for idx, name in enumerate(header) if name is not None}
            n_columns = len(header)

            rows: list[dict[str, Any]] = []
            for raw_row in rows_iter:
                if raw_row is None:
                    continue
                # Defensive: a sheet with no <dimension> element (some
                # non-Excel export tools omit it) can make read_only mode
                # yield rows shorter than the header row. Missing trailing
                # cells are treated as None, exactly like an explicitly
                # blank cell would be, rather than raising IndexError.
                padded_row = raw_row + (None,) * (n_columns - len(raw_row))
                rows.append({name: padded_row[idx] for name, idx in header_index.items()})
            return rows
        finally:
            workbook.close()

    def _to_claim(self, row: dict[str, Any]) -> Claim:
        """
        Map one raw corpus_claim row to a Claim.

        risk_level/sensitivity_class/normalized_label are read defensively
        (fall back to None rather than raising) even though every row in
        the file inspected while writing this loader had clean values for
        all three — this is an external, human-maintained spreadsheet that
        can and will be re-exported with new rows later, and one malformed
        cell (e.g. a typo'd "S4") should not raise a pydantic
        ValidationError that aborts loading every other, valid row in the
        same load() call.
        """
        claim_text = str(row["claim_original"])
        context_query = f"Is it true that {claim_text}?"

        raw_label = row.get("normalized_label")
        ground_truth_label = raw_label if raw_label in _VALID_GROUND_TRUTH_LABELS else None

        raw_risk_level = row.get("risk_level")
        # bool is a subclass of int in Python (isinstance(True, int) is True,
        # and True == 1 would otherwise slip through the range check below)
        # — explicitly excluded so a stray boolean cell can't be silently
        # miscast as risk_level=1.
        risk_level = (
            raw_risk_level
            if isinstance(raw_risk_level, int)
            and not isinstance(raw_risk_level, bool)
            and 1 <= raw_risk_level <= 5
            else None
        )

        raw_sensitivity_class = row.get("sensitivity_class")
        sensitivity_class = (
            raw_sensitivity_class if raw_sensitivity_class in ("S0", "S1", "S2", "S3") else None
        )

        metadata: dict[str, Any] = {}
        for column in _PROVENANCE_METADATA_COLUMNS:
            value = row.get(column)
            if value is not None:
                metadata[column] = value

        return Claim(
            claim_id=str(row["item_id"]),
            original_fact=claim_text,
            context_query=context_query,
            source_dataset=self.name,
            risk_level=risk_level,
            sensitivity_class=sensitivity_class,
            ground_truth_label=ground_truth_label,
            metadata=metadata,
        )
