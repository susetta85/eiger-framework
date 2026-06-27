# eiger.llm

**Status: Not yet implemented — planned for Sprint 3.**

This module will provide LLM generation backends for the RAG answer-generation
step of the EIGER pipeline. All concrete implementations extend `BaseLLM`
from `eiger.core.interfaces`.

---

## Supported Backends

| Backend       | Class          | Status  | Notes                                          |
|---------------|----------------|---------|------------------------------------------------|
| Ollama        | `OllamaLLM`    | Planned | Supports any model served by a local Ollama instance |
| OpenAI-compatible | `OpenAILLM` | Planned | Works with OpenAI API and compatible endpoints |

Ollama is the primary target for local and reproducible experiments.
OpenAI-compatible support covers hosted models and local inference servers
(e.g., LM Studio, vLLM) that expose the `/v1/chat/completions` interface.

---

## Interface Contract

```python
from eiger.core.interfaces import BaseLLM

class BaseLLM(ABC):
    model_name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response given a prompt string.

        Returns the raw text response from the model.
        """

    @abstractmethod
    def build_rag_prompt(self, query: str, context_docs: list[str]) -> str:
        """
        Construct a RAG prompt from a query and retrieved document texts.

        The default template follows a standard context-question format:
        retrieved documents are listed as numbered passages, followed by
        the query and an instruction to answer using only the provided context.
        The template is configurable via LLMConfig.
        """
```

---

## Configuration Reference

LLM backends are configured via `LLMConfig` from `eiger.core.models`:

```python
class LLMConfig(BaseModel):
    backend: str       # "ollama" | "openai"
    model: str         # Model name as recognized by the backend
    temperature: float # Sampling temperature in [0.0, 2.0]
    max_tokens: int    # Maximum tokens in the generated response
```

Example (from `experiments/baseline_v1.yaml`):

```yaml
llm:
  backend: ollama
  model: llama3.1:8b
  temperature: 0.0
  max_tokens: 512
```

Setting `temperature: 0.0` is required for reproducible experiments.

---

## Model Setup (Ollama)

Ollama runs as a Docker service alongside Qdrant. Pull a model into the
running container with:

```bash
# Pull Llama 3.1 8B (used in baseline and ablation experiments)
docker exec ollama ollama pull llama3.1:8b

# Pull Mistral 7B as an alternative
docker exec ollama ollama pull mistral:7b

# List available models
docker exec ollama ollama list
```

The Ollama service is started with `make up` (`docker compose up -d`).

---

## RAG Prompt Template

The default prompt template wraps retrieved context and the query in a
structured format that instructs the model to answer using only the provided
passages. The template is configurable; custom templates can be supplied as a
Jinja2 string in `LLMConfig`. When no template is configured, the default
context-question format is used.

---

## Sprint 3 Milestone

- [ ] `BaseLLM` ABC (already defined in `eiger.core.interfaces`)
- [ ] `OllamaLLM` — HTTP client against the Ollama REST API
- [ ] `OpenAILLM` — client against OpenAI-compatible `/v1/chat/completions`
- [ ] Configurable RAG prompt template
- [ ] Unit tests: prompt construction, response parsing
- [ ] Integration tests: round-trip against a live Ollama instance
