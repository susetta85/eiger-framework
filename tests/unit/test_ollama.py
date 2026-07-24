"""
Unit tests for OllamaLLM (eiger.llm.ollama).

Tests verify:
  - Default and custom constructor parameters are stored correctly
  - base_url is assembled correctly from host/port
  - _client starts as None (lazy-load contract)
  - _get_client() raises ImportError when httpx is missing
  - _get_client() is idempotent (reuses the same client)
  - _get_client() creates an httpx.Client with correct base_url/timeout
  - generate() posts to /api/generate with correct model/prompt/options
  - generate() allows per-call temperature/max_tokens overrides
  - generate() forwards extra kwargs as top-level request fields
  - generate() returns the "response" field from the JSON body
  - generate() raises GenerationError (with chained cause) on HTTP failure
  - generate() raises GenerationError (with chained cause) on non-JSON response
  - generate() raises GenerationError when "response" is missing from the JSON
  - build_rag_prompt() includes the query and numbered document blocks
  - build_rag_prompt() falls back to a placeholder when context_docs is empty
  - build_rag_prompt() preserves document order via [Document N] numbering

What these tests do NOT cover:
  - A real running Ollama server (covered in integration tests).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.exceptions import GenerationError
from eiger.llm.ollama import DEFAULT_HOST, DEFAULT_MODEL, RAG_SYSTEM_INSTRUCTION, OllamaLLM

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_llm_with_mock_client() -> tuple[OllamaLLM, MagicMock]:
    """
    Return an OllamaLLM whose _client is already set to a MagicMock.

    Bypasses _get_client() so tests focus on generate()'s request/response
    handling rather than connection setup.
    """
    llm = OllamaLLM()
    mock_client = MagicMock()
    llm._client = mock_client
    return llm, mock_client


def _make_response(json_data: dict, raise_for_status_error: Exception | None = None) -> MagicMock:
    """Build a mock httpx.Response with the given JSON body."""
    resp = MagicMock()
    resp.json.return_value = json_data
    if raise_for_status_error is not None:
        resp.raise_for_status.side_effect = raise_for_status_error
    else:
        resp.raise_for_status.return_value = None
    return resp


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestOllamaLLMInit:
    """Tests for __init__ defaults and attribute storage."""

    def test_default_model_name(self) -> None:
        assert OllamaLLM().model_name == DEFAULT_MODEL

    def test_default_host(self) -> None:
        assert OllamaLLM().host == DEFAULT_HOST

    def test_default_port(self) -> None:
        assert OllamaLLM().port == 11434

    def test_default_temperature(self) -> None:
        assert OllamaLLM().temperature == 0.0

    def test_default_max_tokens(self) -> None:
        assert OllamaLLM().max_tokens == 512

    def test_default_timeout(self) -> None:
        assert OllamaLLM().timeout == 60.0

    def test_custom_params_stored(self) -> None:
        llm = OllamaLLM(
            model_name="mistral:7b",
            host="ollama-server",
            port=9999,
            temperature=0.7,
            max_tokens=256,
            timeout=30.0,
        )
        assert llm.model_name == "mistral:7b"
        assert llm.host == "ollama-server"
        assert llm.port == 9999
        assert llm.temperature == 0.7
        assert llm.max_tokens == 256
        assert llm.timeout == 30.0

    def test_client_starts_as_none(self) -> None:
        assert OllamaLLM()._client is None

    def test_base_url_default(self) -> None:
        assert OllamaLLM().base_url == f"http://{DEFAULT_HOST}:11434"

    def test_base_url_custom(self) -> None:
        llm = OllamaLLM(host="myhost", port=1234)
        assert llm.base_url == "http://myhost:1234"


# ─── _get_client ──────────────────────────────────────────────────────────────

class TestGetClient:
    """Tests for the lazy httpx.Client initialiser."""

    def test_raises_when_httpx_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "httpx", None)
        llm = OllamaLLM()
        with pytest.raises(ImportError, match="httpx"):
            llm._get_client()

    def test_idempotent_returns_same_client(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        assert llm._get_client() is mock_client
        assert llm._get_client() is mock_client

    def test_creates_client_with_correct_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_httpx_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_httpx_module.Client.return_value = mock_client_instance
        monkeypatch.setitem(sys.modules, "httpx", mock_httpx_module)

        llm = OllamaLLM(host="myhost", port=4242, timeout=15.0)
        with patch("eiger.llm.ollama.log"):
            client = llm._get_client()

        mock_httpx_module.Client.assert_called_once_with(
            base_url="http://myhost:4242", timeout=15.0
        )
        assert client is mock_client_instance


# ─── generate() ────────────────────────────────────────────────────────────────

class TestGenerate:
    """Tests for generate()."""

    def test_posts_to_generate_endpoint(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"response": "answer"})
        with patch("eiger.llm.ollama.log"):
            llm.generate("hello")
        args, _ = mock_client.post.call_args
        assert args[0] == "/api/generate"

    def test_payload_contains_model_and_prompt(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"response": "answer"})
        with patch("eiger.llm.ollama.log"):
            llm.generate("what is x?")
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["model"] == DEFAULT_MODEL
        assert payload["prompt"] == "what is x?"
        assert payload["stream"] is False

    def test_payload_options_use_instance_defaults(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        llm.temperature = 0.3
        llm.max_tokens = 128
        mock_client.post.return_value = _make_response({"response": "answer"})
        with patch("eiger.llm.ollama.log"):
            llm.generate("hello")
        options = mock_client.post.call_args.kwargs["json"]["options"]
        assert options["temperature"] == 0.3
        assert options["num_predict"] == 128

    def test_temperature_override_via_kwargs(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"response": "answer"})
        with patch("eiger.llm.ollama.log"):
            llm.generate("hello", temperature=0.9)
        options = mock_client.post.call_args.kwargs["json"]["options"]
        assert options["temperature"] == 0.9

    def test_max_tokens_override_via_kwargs(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"response": "answer"})
        with patch("eiger.llm.ollama.log"):
            llm.generate("hello", max_tokens=999)
        options = mock_client.post.call_args.kwargs["json"]["options"]
        assert options["num_predict"] == 999

    def test_extra_kwargs_forwarded_as_top_level_fields(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"response": "answer"})
        with patch("eiger.llm.ollama.log"):
            llm.generate("hello", system="be terse")
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["system"] == "be terse"
        # Must not leak into the options sub-dict.
        assert "system" not in payload["options"]

    def test_returns_response_field(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"response": "42% inflation"})
        with patch("eiger.llm.ollama.log"):
            answer = llm.generate("hello")
        assert answer == "42% inflation"

    def test_raises_generation_error_on_http_failure(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.side_effect = RuntimeError("connection refused")
        with patch("eiger.llm.ollama.log"), pytest.raises(
            GenerationError, match="Ollama generation failed"
        ):
            llm.generate("hello")

    def test_http_failure_chains_original_exception(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        original = RuntimeError("connection refused")
        mock_client.post.side_effect = original
        with patch("eiger.llm.ollama.log"), pytest.raises(GenerationError) as exc_info:
            llm.generate("hello")
        assert exc_info.value.__cause__ is original

    def test_raises_generation_error_on_bad_status(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response(
            {"response": "unused"}, raise_for_status_error=RuntimeError("500 error")
        )
        with patch("eiger.llm.ollama.log"), pytest.raises(
            GenerationError, match="Ollama generation failed"
        ):
            llm.generate("hello")

    def test_raises_generation_error_on_non_json_response(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        bad_response = MagicMock()
        bad_response.raise_for_status.return_value = None
        bad_response.json.side_effect = ValueError("not json")
        mock_client.post.return_value = bad_response
        with patch("eiger.llm.ollama.log"), pytest.raises(
            GenerationError, match="non-JSON response"
        ):
            llm.generate("hello")

    def test_raises_generation_error_when_response_field_missing(self) -> None:
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"unexpected": "shape"})
        with patch("eiger.llm.ollama.log"), pytest.raises(
            GenerationError, match="missing the 'response' field"
        ):
            llm.generate("hello")

    def test_raises_generation_error_when_response_field_is_not_a_string(self) -> None:
        """
        A non-string "response" field (e.g. a number or nested object) must be
        rejected explicitly rather than silently returned to the caller.
        """
        llm, mock_client = _make_llm_with_mock_client()
        mock_client.post.return_value = _make_response({"response": 42})
        with patch("eiger.llm.ollama.log"), pytest.raises(
            GenerationError, match="non-string 'response' field"
        ):
            llm.generate("hello")


# ─── build_rag_prompt() ────────────────────────────────────────────────────────

class TestBuildRagPrompt:
    """Tests for build_rag_prompt()."""

    def test_includes_system_instruction(self) -> None:
        llm = OllamaLLM()
        prompt = llm.build_rag_prompt("query?", ["doc text"])
        assert RAG_SYSTEM_INSTRUCTION in prompt

    def test_includes_query(self) -> None:
        llm = OllamaLLM()
        prompt = llm.build_rag_prompt("What was the 2023 inflation rate?", ["doc text"])
        assert "What was the 2023 inflation rate?" in prompt

    def test_numbers_documents_in_order(self) -> None:
        llm = OllamaLLM()
        prompt = llm.build_rag_prompt("q", ["first doc", "second doc"])
        assert "[Document 1]\nfirst doc" in prompt
        assert "[Document 2]\nsecond doc" in prompt
        assert prompt.index("[Document 1]") < prompt.index("[Document 2]")

    def test_empty_context_uses_placeholder(self) -> None:
        llm = OllamaLLM()
        prompt = llm.build_rag_prompt("q", [])
        assert "(no documents retrieved)" in prompt
        assert "[Document" not in prompt

    def test_ends_with_answer_prompt(self) -> None:
        llm = OllamaLLM()
        prompt = llm.build_rag_prompt("q", ["doc"])
        assert prompt.rstrip().endswith("Answer:")
