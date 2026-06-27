# eiger.config

This package manages all runtime configuration for the EIGER framework. It provides a
single `EigerSettings` object built on Pydantic BaseSettings, which reads values from
environment variables, a `.env` file, and built-in defaults — in that order of priority.

---

## Settings reference

| Environment variable           | Field               | Default                                        | Description                                     |
|-------------------------------|---------------------|------------------------------------------------|-------------------------------------------------|
| `EIGER_QDRANT_HOST`           | `qdrant_host`       | `localhost`                                    | Hostname of the Qdrant vector-store server      |
| `EIGER_QDRANT_PORT`           | `qdrant_port`       | `6333`                                         | Port of the Qdrant vector-store server          |
| `EIGER_OLLAMA_HOST`           | `ollama_host`       | `localhost`                                    | Hostname of the Ollama inference server         |
| `EIGER_OLLAMA_PORT`           | `ollama_port`       | `11434`                                        | Port of the Ollama inference server             |
| `EIGER_DEFAULT_EMBEDDER`      | `default_embedder`  | `sentence-transformers/all-MiniLM-L6-v2`       | Model identifier for the default embedder       |
| `EIGER_RESULTS_DIR`           | `results_dir`       | `results/`                                     | Directory where experiment results are written  |
| `EIGER_LOG_LEVEL`             | `log_level`         | `INFO`                                         | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `EIGER_DEFAULT_SEED`          | `default_seed`      | `42`                                           | Global random seed for reproducibility          |

### Computed properties

- `qdrant_url` — returns `http://{qdrant_host}:{qdrant_port}`
- `ollama_url` — returns `http://{ollama_host}:{ollama_port}`

---

## Resolution order

Values are resolved in the following order (highest priority first):

1. Environment variable with the `EIGER_` prefix
2. Variable defined in a `.env` file in the working directory
3. Built-in default defined in `EigerSettings`

---

## Usage

### Accessing settings

```python
from eiger.config.settings import get_settings

settings = get_settings()
print(settings.qdrant_url)    # http://localhost:6333
print(settings.ollama_url)    # http://localhost:11434
print(settings.default_seed)  # 42
```

`get_settings()` is decorated with `@lru_cache`, so it returns the same instance on
every call within a process. This means configuration is read once at startup.

### Overriding via environment variable

```bash
EIGER_QDRANT_HOST=qdrant-server EIGER_LOG_LEVEL=DEBUG python run_experiment.py
```

Or export before running:

```bash
export EIGER_DEFAULT_SEED=123
export EIGER_RESULTS_DIR=/data/runs/
python run_experiment.py
```

### Using a .env file

Place a `.env` file in the project root (or the working directory where EIGER is
invoked). See `.env.example` at the repository root for a template covering all fields.

```
EIGER_QDRANT_HOST=localhost
EIGER_QDRANT_PORT=6333
EIGER_OLLAMA_HOST=localhost
EIGER_OLLAMA_PORT=11434
EIGER_DEFAULT_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
EIGER_RESULTS_DIR=results/
EIGER_LOG_LEVEL=INFO
EIGER_DEFAULT_SEED=42
```

---

## Testing note

Because `get_settings()` uses `@lru_cache`, tests that need fresh settings must clear
the cache between test cases:

```python
from eiger.config.settings import get_settings
import os

def test_custom_seed(monkeypatch):
    monkeypatch.setenv("EIGER_DEFAULT_SEED", "99")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.default_seed == 99
    get_settings.cache_clear()  # restore for subsequent tests
```

Always call `get_settings.cache_clear()` both before and after any test that patches
environment variables to avoid state leaking between test cases.
