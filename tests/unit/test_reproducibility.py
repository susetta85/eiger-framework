"""
Tests for the reproducibility metadata recorded in results.json.

Covers:
  - BaseLLM.backend_info() default (empty dict)
  - OllamaLLM.backend_info(): server version and model digest, matching by
    name, by "<name>:latest" and by the "model" key; "not found" and
    "unavailable" values instead of exceptions
  - ExperimentRunner._capture_environment(): package versions,
    "not installed", HF_HUB_OFFLINE
  - ExperimentRunner._capture_llm_backend(): dict, non-dict, exception
  - the backend info ends up in ExperimentResult.environment
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.interfaces import BaseLLM
from eiger.experiments.runner import _TRACKED_PACKAGES, ExperimentRunner
from eiger.llm.ollama import OllamaLLM
from tests.unit.test_ollama import _make_llm_with_mock_client, _make_response
from tests.unit.test_runner import _make_claim, _make_runner


class _MinimalLLM(BaseLLM):
    model_name = "minimal"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return "answer"

    def build_rag_prompt(self, query: str, context_docs: list[str]) -> str:
        return query


def _client_with(version: Any, tags: Any) -> tuple[OllamaLLM, MagicMock]:
    """OllamaLLM whose /api/version and /api/tags return the given responses."""
    llm, client = _make_llm_with_mock_client()

    def get(path: str) -> MagicMock:
        resp = version if path == "/api/version" else tags
        if isinstance(resp, Exception):
            raise resp
        return resp

    client.get.side_effect = get
    return llm, client


class TestBaseLLMBackendInfo:
    def test_default_is_empty(self) -> None:
        assert _MinimalLLM().backend_info() == {}


class TestOllamaBackendInfo:
    def test_version_and_digest_by_name(self) -> None:
        llm, _ = _client_with(
            _make_response({"version": "0.5.7"}),
            _make_response(
                {
                    "models": [
                        {"name": "other:1b", "digest": "x"},
                        {"name": "llama3.1:8b", "digest": "abc"},
                    ]
                }
            ),
        )
        info = llm.backend_info()
        assert info == {
            "llm_backend": "ollama",
            "llm_model": "llama3.1:8b",
            "ollama_version": "0.5.7",
            "llm_model_digest": "abc",
        }

    def test_untagged_name_matches_latest(self) -> None:
        llm, _ = _client_with(
            _make_response({"version": "0.5.7"}),
            _make_response({"models": [{"name": "llama3.1:latest", "digest": "lat"}]}),
        )
        llm.model_name = "llama3.1"
        assert llm.backend_info()["llm_model_digest"] == "lat"

    def test_match_on_model_key(self) -> None:
        llm, _ = _client_with(
            _make_response({"version": "0.5.7"}),
            _make_response({"models": [{"name": "alias", "model": "llama3.1:8b", "digest": "m"}]}),
        )
        assert llm.backend_info()["llm_model_digest"] == "m"

    def test_model_not_in_list(self) -> None:
        llm, _ = _client_with(
            _make_response({"version": "0.5.7"}),
            _make_response({"models": None}),
        )
        assert llm.backend_info()["llm_model_digest"] == "not found"

    def test_missing_version_field(self) -> None:
        llm, _ = _client_with(_make_response({}), _make_response({"models": []}))
        assert llm.backend_info()["ollama_version"] == "unavailable"

    def test_http_errors_are_values(self) -> None:
        llm, _ = _client_with(
            _make_response({}, raise_for_status_error=RuntimeError("500")),
            ConnectionError("refused"),
        )
        info = llm.backend_info()
        assert info["ollama_version"] == "unavailable"
        assert info["llm_model_digest"] == "unavailable"

    def test_no_httpx(self) -> None:
        llm = OllamaLLM()
        with patch.object(OllamaLLM, "_get_client", side_effect=ImportError("no httpx")):
            info = llm.backend_info()
        assert info["ollama_version"] == "unavailable"
        assert info["llm_model_digest"] == "unavailable"


class TestCaptureEnvironmentPackages:
    def test_tracked_packages_present(self) -> None:
        env = ExperimentRunner._capture_environment()
        for package in _TRACKED_PACKAGES:
            assert f"pkg_{package}" in env
            assert isinstance(env[f"pkg_{package}"], str)

    def test_not_installed(self) -> None:
        with patch(
            "eiger.experiments.runner.metadata.version",
            side_effect=metadata.PackageNotFoundError("x"),
        ):
            env = ExperimentRunner._capture_environment()
        assert all(env[f"pkg_{p}"] == "not installed" for p in _TRACKED_PACKAGES)

    def test_hf_hub_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        assert ExperimentRunner._capture_environment()["hf_hub_offline"] == "unset"
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert ExperimentRunner._capture_environment()["hf_hub_offline"] == "1"


class TestCaptureLLMBackend:
    def test_dict_is_stringified(self, tmp_path: Path) -> None:
        runner, _, _, llm = _make_runner(tmp_path)
        llm.backend_info.return_value = {"ollama_version": "0.5.7", "n": 3}
        assert runner._capture_llm_backend() == {"ollama_version": "0.5.7", "n": "3"}

    def test_non_dict_ignored(self, tmp_path: Path) -> None:
        runner, _, _, llm = _make_runner(tmp_path)
        llm.backend_info.return_value = ["not", "a", "dict"]
        assert runner._capture_llm_backend() == {}

    def test_exception_reported(self, tmp_path: Path) -> None:
        runner, _, _, llm = _make_runner(tmp_path)
        llm.backend_info.side_effect = RuntimeError("boom")
        assert runner._capture_llm_backend() == {"llm_backend": "unavailable"}

    def test_recorded_in_result(self, tmp_path: Path) -> None:
        runner, _, _, llm = _make_runner(tmp_path)
        llm.backend_info.return_value = {"llm_model_digest": "abc"}
        result = runner.run([_make_claim()], save=False)
        assert result.environment["llm_model_digest"] == "abc"
        assert "pkg_numpy" in result.environment
        assert "python_version" in result.environment
