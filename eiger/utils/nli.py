"""
Three-way NLI scoring (entailment / neutral / contradiction) for analysis scripts.

Used by the answer-level and defence analyses (``scripts/contradiction_audit.py``,
``scripts/analyze_answers.py``, ``scripts/calibration_kit.py``). It is the same
model and the same pipeline call as ``SourceIntegrityMetric``
(``cross-encoder/nli-MiniLM2-L6-H768``, ``text-classification`` with
``top_k=None``), so numbers from the two places are comparable.

Why a separate helper rather than reusing SourceIntegrityMetric
----------------------------------------------------------------
SourceIntegrityMetric returns only P(entailment) and is tied to an
EvaluationRecord. The analyses need all three probabilities for arbitrary text
pairs, in batches, with a cache: the same document pairs recur in every seed.

What this class does NOT do
---------------------------
- It does not decide thresholds. Callers report probabilities, and any
  threshold is chosen (and documented) by the analysis that uses it.
- It does not claim calibration. NLI probabilities from a small cross-encoder
  are uncalibrated; ``scripts/calibration_kit.py`` checks them against human
  labels.
"""

from __future__ import annotations

from typing import Any

DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"

LABELS = ("entailment", "neutral", "contradiction")
_PREFIXES = {"entail": "entailment", "neutral": "neutral", "contradict": "contradiction"}


def normalize_scores(raw: Any) -> dict[str, float]:
    """
    Map one pipeline output (a list of ``{label, score}`` dicts, possibly wrapped
    in a one-element list) to ``{"entailment", "neutral", "contradiction"}``.

    Raises:
        ValueError: if any of the three labels is missing (not a 3-way NLI model).
    """
    if raw and isinstance(raw, list) and isinstance(raw[0], list):
        raw = raw[0]
    out: dict[str, float] = {}
    for item in raw or []:
        label = str(item["label"]).lower()
        for prefix, name in _PREFIXES.items():
            if label.startswith(prefix):
                out[name] = float(item["score"])
    missing = [name for name in LABELS if name not in out]
    if missing:
        got = [item["label"] for item in raw or []]
        raise ValueError(
            f"NLI output lacks labels {missing} (got {got}); a 3-way NLI model is required."
        )
    return out


class NLIScorer:
    """
    Batched, cached three-way NLI probabilities for (premise, hypothesis) pairs.

    Args:
        model_name: HuggingFace model id. Defaults to the Source Integrity model.
        pipeline:   Optional pre-built callable with the transformers pipeline
                    signature (used by tests; loaded lazily otherwise).
        batch_size: Pairs per pipeline call.
    """

    def __init__(
        self, model_name: str = DEFAULT_NLI_MODEL, pipeline: Any = None, batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = pipeline
        self._cache: dict[tuple[str, str], dict[str, float]] = {}

    def _load(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline("text-classification", model=self.model_name, device=-1)
        return self._pipeline

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def scores(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Return one ``{entailment, neutral, contradiction}`` dict per pair, in order."""
        todo = list(dict.fromkeys(p for p in pairs if p not in self._cache))
        if todo:
            pipe = self._load()
            for start in range(0, len(todo), self.batch_size):
                chunk = todo[start : start + self.batch_size]
                inputs = [
                    {"text": premise, "text_pair": hypothesis} for premise, hypothesis in chunk
                ]
                results = pipe(inputs, top_k=None)
                if len(chunk) == 1 and results and isinstance(results[0], dict):
                    results = [results]  # some versions unwrap a single input
                if len(results) != len(chunk):
                    raise ValueError(
                        f"NLI pipeline returned {len(results)} outputs for {len(chunk)} inputs."
                    )
                for pair, raw in zip(chunk, results):
                    self._cache[pair] = normalize_scores(raw)
        return [self._cache[p] for p in pairs]

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Three-way probabilities for a single pair."""
        return self.scores([(premise, hypothesis)])[0]

    def symmetric_contradiction(self, text_a: str, text_b: str) -> float:
        """max(P(contradiction | a, b), P(contradiction | b, a)): NLI is not symmetric."""
        forward, backward = self.scores([(text_a, text_b), (text_b, text_a)])
        return max(forward["contradiction"], backward["contradiction"])
