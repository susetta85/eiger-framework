# eiger.llm

**Status: `OllamaLLM` implemented (Sprint 2, Step 5). `OpenAILLM` remains future work.**

This module provides LLM generation backends for the RAG answer-generation
step of the EIGER pipeline. All concrete implementations extend `BaseLLM`
from `eiger.core.interfaces`.

---

## Supported Backends

| Backend       | Class          | Status  | Notes                                          |
|---------------|----------------|---------|------------------------------------------------|
| Ollama        | `OllamaLLM`    | ✅ Implemented | HTTP client against a local Ollama server's `/api/generate` |
| OpenAI-compatible | `OpenAILLM` | 🔲 Planned | Would work with OpenAI API and compatible endpoints |

Ollama is the primary target for local and reproducible experiments (pinned
`ollama/ollama:0.2.8` in `docker-compose.yml`).

---

## `OllamaLLM`

```python
from eiger.llm import OllamaLLM

llm = OllamaLLM(
    model_name="llama3.1:8b",
    host="localhost",
    port=11434,
    temperature=0.0,   # deterministic — required for reproducible experiments
    max_tokens=512,
    timeout=60.0,
)

prompt = llm.build_rag_prompt(
    query="What was the 2023 inflation rate?",
    context_docs=["The WHO reported inflation rose to 3.5% in 2023."],
)
answer = llm.generate(prompt)
```

- **Lazy HTTP client**: the underlying `httpx.Client` is created on the first
  call to `generate()`, not at construction time.
- **Non-streaming**: every request sends `"stream": false`, so Ollama returns
  the full answer in one JSON response rather than incremental chunks —
  simpler to reason about for offline experiment reproducibility.
- **Per-call overrides**: `generate(prompt, temperature=..., max_tokens=...)`
  overrides the instance defaults for a single call. `ExperimentRunner`
  always passes `config.llm.temperature`/`max_tokens` explicitly on every
  call, so the *actual* generation parameters used always match the
  experiment config regardless of how the injected `OllamaLLM` was
  constructed.
- **Response validation**: `generate()` raises `GenerationError` if the HTTP
  request fails, the response isn't valid JSON, the `"response"` field is
  missing, or that field isn't a string — never silently returns a malformed
  value.
- `httpx` is an explicit `pyproject.toml` dependency (it was previously only
  a transitive dependency of `qdrant-client`).

---

## Interface Contract

```python
from eiger.core.interfaces import BaseLLM

class BaseLLM(ABC):
    model_name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response given a fully-assembled prompt string."""

    @abstractmethod
    def build_rag_prompt(self, query: str, context_docs: list[str]) -> str:
        """Construct a RAG prompt from a query and retrieved document texts."""
```

---

## RAG Prompt Template

`OllamaLLM.build_rag_prompt()` uses a fixed template (not currently
Jinja2-configurable): a system instruction telling the model to answer using
only the provided context, followed by numbered `[Document N]` blocks (in
retrieval-rank order — useful for auditing which document the model actually
relied on), the question, and an `Answer:` cue. When `context_docs` is
empty, an explicit `(no documents retrieved)` placeholder is used instead of
a blank context section.

A configurable (e.g. Jinja2-based) template is not implemented; the current
template is adequate for the built-in metrics but is a reasonable place to
extend if a future experiment needs a different prompting style.

---

## Configuration Reference

LLM backends are configured via `LLMConfig` from `eiger.core.models`:

```python
class LLMConfig(BaseModel):
    backend: str = "ollama"    # "ollama" | "openai" (only "ollama" implemented)
    model: str = "llama3.1:8b"
    temperature: float = 0.0   # [0.0, 2.0]; 0.0 required for reproducibility
    max_tokens: int = 512
```

**Note:** `ExperimentRunner` does not construct an `OllamaLLM` from this
config internally — there is no LLM factory yet. The caller constructs the
actual `OllamaLLM(model_name=config.llm.model, ...)` instance and injects it;
`ExperimentRunner` still reads `config.llm.temperature`/`max_tokens` on every
`generate()` call for provenance-accurate reproducibility (see
`eiger/experiments/README.md`).

---

## Model Setup (Ollama)

By default, Ollama runs **natively** on the host (not in Docker) — see
`setup.sh` / `make bootstrap`, which installs it if missing and
automatically pulls the default model. Only Qdrant runs via Docker
(`make up`). Pull models directly:

```bash
make ollama-pull            # pulls the default model (llama3.1:8b)
ollama pull mistral:7b
ollama list
```

If you'd rather use the pinned `ollama/ollama:0.2.8` container instead
of a native install, it's available behind docker-compose.yml's
opt-in `docker-ollama` profile (`make up-docker-ollama`); in that case
pull models with `docker exec eiger-ollama ollama pull llama3.1:8b`.

---

## Test coverage

`tests/unit/test_ollama.py` (38 tests) covers construction, the lazy client,
`generate()` (including every failure mode), and `build_rag_prompt()` with
100% line coverage, using a mocked `httpx` client — no real Ollama server
required.

## Remaining work

- [ ] `OpenAILLM` — client against OpenAI-compatible `/v1/chat/completions`
- [ ] Configurable RAG prompt template
- [ ] Integration tests: round-trip against a live Ollama instance
