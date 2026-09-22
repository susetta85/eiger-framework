"""
Integration tests for RAGASFaithfulnessScorer against the REAL, actually
installed `ragas` / `langchain-ollama` / `langchain-community` packages —
no `sys.modules` mocking anywhere in this file.

Unlike tests/unit/test_ragas_scorer.py (always runs, mocks every ragas
import), this module exercises the genuine import chain and the genuine
RAGAS object types, matching the project's existing split between
test_pipeline_fake_infra.py (mocked, always runs) and
test_pipeline_live_infra.py (real infra, skips gracefully) — see that
file's own docstring for the convention this one follows.

Two tiers, skipped independently so this file stays runnable (and useless
runs stay harmless) whatever is or isn't installed/reachable:

  1. Package-only tests (`_require_real_ragas` fixture): skipped unless the
     pinned `ragas` extra is actually importable
     (`pip install -e ".[ragas]"` — see eiger/metrics/ragas_scorer.py's
     module docstring for the exact pinned versions and why). These verify
     real construction/wiring — that RAGASFaithfulnessScorer.__init__
     actually builds genuine ragas.metrics.Faithfulness /
     AnswerCorrectness instances, and that __call__'s lazy
     `from ragas.dataset_schema import SingleTurnSample` import resolves to
     the real class with the exact fields this project relies on — without
     needing a live Ollama server (ChatOllama/LangchainLLMWrapper
     construction does not itself open a network connection).

  2. Live-scoring test (`_require_real_ragas_and_ollama` fixture): further
     skipped unless Ollama is reachable at EIGER_OLLAMA_HOST:PORT with
     the judge model already pulled (`make bootstrap` / `make
     ollama-pull`). This is the only test in the whole suite that performs
     an actual end-to-end RAGAS judge call — real HTTP calls to a real LLM,
     so its outcome is not deterministic and it is NOT asserted against
     exact score values, only against the documented output contract
     (dict, both keys present, both values in [0.0, 1.0]).

Neither tier is part of the project's 100% unit-coverage gate
(pyproject.toml's `--cov-fail-under=100` targets `tests/` broadly, but see
Makefile: `test-unit` runs `tests/unit/` only; this file lives under
`tests/integration/`, matching test_pipeline_live_infra.py).
"""

from __future__ import annotations

import socket

import pytest

from eiger.config import get_settings
from eiger.core.models import Claim, GenerationResult
from eiger.retrieval import SentenceTransformerEmbedder

# Must already be pulled on the target Ollama server for the live-scoring
# test to actually execute (see module docstring / _require_real_ragas_and_ollama).
_MODEL_NAME = "llama3.1:8b"


def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def _require_real_ragas() -> None:
    """Skip this module's package-only tests if the `ragas` extra isn't installed."""
    pytest.importorskip(
        "ragas",
        reason=(
            "The pinned 'ragas' extra is not installed — run "
            "`pip install -e \".[ragas]\"` to exercise real "
            "RAGASFaithfulnessScorer wiring (see "
            "eiger/metrics/ragas_scorer.py's docstring for the exact "
            "pinned versions this requires)."
        ),
    )
    pytest.importorskip("langchain_ollama", reason="langchain-ollama not installed.")


@pytest.fixture(scope="module")
def _require_real_ragas_and_ollama(_require_real_ragas: None) -> None:
    """Additionally skip the live-scoring test unless Ollama is reachable."""
    settings = get_settings()
    if not _port_open(settings.ollama_host, settings.ollama_port):
        pytest.skip(
            "Ollama not reachable at the configured EIGER_OLLAMA_HOST/PORT. "
            f"Run `make up` and `make ollama-pull` (pulls {_MODEL_NAME}) to "
            "exercise real end-to-end RAGAS scoring."
        )


def test_real_ragas_construction_builds_genuine_metric_instances(
    _require_real_ragas: None,
) -> None:
    """
    Constructing RAGASFaithfulnessScorer against the real, installed ragas
    package produces genuine ragas.metrics.Faithfulness/AnswerCorrectness
    instances (not the unit tests' MagicMocks), and __call__'s own lazy
    import of the real SingleTurnSample class round-trips the exact fields
    this project relies on — all without needing a live Ollama connection
    (constructing ChatOllama/LangchainLLMWrapper does not itself dial out).
    """
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerCorrectness, Faithfulness

    from eiger.metrics.ragas_scorer import RAGASFaithfulnessScorer

    scorer = RAGASFaithfulnessScorer(
        embedder=SentenceTransformerEmbedder(),
        model_name=_MODEL_NAME,
    )

    assert isinstance(scorer._faithfulness, Faithfulness)
    assert isinstance(scorer._answer_correctness, AnswerCorrectness)

    # Exercise the real (non-mocked) SingleTurnSample construction path
    # directly — this is exactly the object __call__ builds at line 266 of
    # ragas_scorer.py, using the real ragas class rather than a mock.
    sample = SingleTurnSample(
        user_input="What happened to inflation?",
        response="Inflation rose to 3.5%.",
        retrieved_contexts=["Inflation rose to 3.5% in 2023."],
        reference="Inflation rose to 3.5% in 2023.",
    )
    assert sample.user_input == "What happened to inflation?"
    assert sample.retrieved_contexts == ["Inflation rose to 3.5% in 2023."]


def test_real_end_to_end_scoring_against_live_ollama(
    _require_real_ragas_and_ollama: None,
) -> None:
    """
    Full, real RAGAS judge call: real ragas, real Ollama, real HTTP.

    Not asserted against exact values (a live LLM judge is not
    deterministic) — only against the documented output contract. Per
    ragas_scorer.py's own docstring, Ollama-as-judge reliability is
    unvalidated for EIBench's claims; this test is a wiring smoke test,
    not a quality benchmark.
    """
    from eiger.metrics.ragas_scorer import RAGASFaithfulnessScorer

    settings = get_settings()
    scorer = RAGASFaithfulnessScorer(
        embedder=SentenceTransformerEmbedder(),
        model_name=_MODEL_NAME,
        host=settings.ollama_host,
        port=settings.ollama_port,
    )

    claim = Claim(
        claim_id="C1",
        original_fact="Inflation rose to 3.5% in 2023.",
        context_query="What happened to inflation?",
        source_dataset="test_fixture",
    )
    generation = GenerationResult(
        claim_id="C1",
        query="What happened to inflation?",
        context_docs=["Inflation rose to 3.5% in 2023, according to the report."],
        answer="Inflation rose to 3.5%.",
        model_name=_MODEL_NAME,
    )

    result = scorer(claim, generation)

    assert set(result) == {"ragas_faithfulness", "ragas_answer_correctness"}
    for value in result.values():
        assert 0.0 <= value <= 1.0
