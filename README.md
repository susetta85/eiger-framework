# EIGER · EIBench

**Epistemic Integrity Benchmark for Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen.svg)](tests/)
[![Sprint](https://img.shields.io/badge/sprint-1%20complete-blue.svg)](docs/ARCHITECTURE.md)

---

## What is EIBench?

EIBench is an open-source research framework for measuring **Epistemic Robustness** in Retrieval-Augmented Generation (RAG) systems.

Standard RAG evaluation asks: *"Is the answer faithful to the retrieved context?"*  
EIBench asks a harder question: *"What happens when the retrieved context is adversarially poisoned?"*

It introduces and operationalises the **Faithful Falsehood** — an answer that is:
- ✅ **Faithful** to the retrieved context (the LLM did its job correctly), and
- ❌ **Factually wrong** relative to independently verified ground truth (the context itself was poisoned)

This failure mode is invisible to standard faithfulness metrics (RAGAS, TruLens) because faithfulness is measured against context, not against truth.

---

## Research Hypothesis

> *"Faithfulness and Source Integrity are independent evaluation dimensions. As corpus poisoning increases, faithfulness remains high while Source Integrity decreases — leading to a measurable increase in Faithful Falsehoods."*

---

## Core Metrics

### Faithful Falsehood Rate (FFR)

$$\text{FFR} = \frac{|\{a : \text{faithful}(a) > \tau_f \;\wedge\; \text{correct}(a) < \tau_c\}|}{|A|}$$

where $A$ is the set of all generated answers, $\tau_f = 0.8$ and $\tau_c = 0.2$ are configurable thresholds.

**Interpretation:** FFR = 0.0 is ideal. FFR = 1.0 means every answer is faithful to poisoned context but wrong relative to ground truth.

### Source Integrity (SI)

$$\text{SI} = \frac{1}{|R|} \sum_{d \in R} P(\text{entailment} \mid d, g)$$

where $R$ is the set of retrieved documents and $g$ is the ground-truth claim. Computed via an NLI cross-encoder.

**Interpretation:** SI = 1.0 means all retrieved context is factually consistent with ground truth. SI = 0.0 means all retrieved context contradicts ground truth.

### Epistemic Risk Score (ERS)

$$\text{ERS} = \frac{0.3 \cdot \text{plausibility} + 0.4 \cdot \text{verification\_difficulty} + 0.3 \cdot \text{editorial\_risk}}{5}$$

Scores in [1, 5] are provided by human annotators or a calibrated LLM judge.

---

## Architecture

EIBench is a six-layer pipeline. Each layer is independently extensible via a plugin architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                        EIGER Pipeline                        │
├──────────┬──────────┬──────────┬──────────┬────────┬────────┤
│  Layer 1 │  Layer 2 │  Layer 3 │  Layer 4 │ Layer 5│ Layer 6│
│  Corpus  │ Poisoning│ Retrieval│Generation│  Eval  │Analytics│
│  Builder │  Engine  │  (Qdrant)│  (Ollama)│ Engine │Plotting│
├──────────┴──────────┼──────────┴──────────┼────────┴────────┤
│    eiger/ingestion  │   eiger/retrieval   │  eiger/metrics  │
│    eiger/attacks    │   eiger/llm         │  eiger/visual.  │
└─────────────────────┴─────────────────────┴─────────────────┘
         ↑                      ↑                    ↑
   eiger/datasets         eiger/vector_stores   eiger/experiments
         ↑                      ↑                    ↑
                    eiger/core  (models + interfaces)
                    eiger/config (Pydantic Settings)
                    eiger/utils  (logging, seeding)
```

### Implementation status

| Layer | Component | Status |
|-------|-----------|--------|
| 1 | Corpus Builder | ✅ Sprint 1 |
| 2 | Poisoning Engine (4 attack types) | ✅ Sprint 1 |
| 3 | Qdrant retrieval | ✅ Sprint 1 (wiring Sprint 3) |
| 4 | Llama 3.1 / Mistral via Ollama | 🔄 Sprint 3 |
| 5 | SI (NLI), FFR, ERS, RAGAS | 🔄 Sprint 4 |
| 6 | Degradation curves, HTML report | 🔄 Sprint 5 |

---

## Attack Taxonomy

| ID | Name | Description | EIBench Type |
|----|------|-------------|--------------|
| `numerical_shift` | Numerical Shift | Swaps adjacent digits: `3.5%` → `35.%` | Type 1 |
| `date_manipulation` | Date Manipulation | Shifts year references: `2024` → `2019` | Type 2 |
| `attribution_switch` | Attribution Switch | Replaces sources: `WHO` → `a blog` | Type 3 |
| `causal_manipulation` | Causal Manipulation | Injects fabricated causal clauses | Type 4 |

All attacks: deterministic (seed-controlled), isolated (no global state mutation), extensible (plugin registry).

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker (for Qdrant and Ollama)

### 2. Setup

```bash
git clone <repo-url>
cd eiger-framework

# Create virtual environment and install
make setup
source venv/bin/activate

# Copy environment template
make env
# Edit .env if needed (defaults work for local Docker Compose)
```

### 3. Start infrastructure

```bash
make up
# Starts Qdrant (port 6333) and Ollama (port 11434)
```

### 4. Run the quickstart pipeline (Layers 1–3)

```bash
python pipeline_eibench.py --poison-rate 0.3 --top-k 5
```

### 5. Run the full test suite

```bash
make test
```

### 6. Run a full experiment

```bash
# Pull the LLM first (one-time, ~5GB)
docker exec eiger-ollama ollama pull llama3.1:8b

# Run baseline experiment
python -m eiger run experiments/baseline_v1.yaml

# Run ablation study (all 4 attacks)
python -m eiger run experiments/ablation_attacks.yaml
```

---

## Project Structure

```
eiger-framework/
│
├── eiger/                    # Main Python package
│   ├── core/                 # Domain models, ABCs, exceptions
│   ├── attacks/              # Adversarial poisoning strategies
│   ├── datasets/             # Dataset loaders (AVeriTeC, PolitiFact, …)
│   ├── ingestion/            # Corpus builder
│   ├── retrieval/            # Retrieval strategies (dense, sparse, hybrid)
│   ├── vector_stores/        # Vector store adapters (Qdrant, FAISS, Chroma)
│   ├── llm/                  # LLM backends (Ollama, OpenAI-compatible)
│   ├── metrics/              # Evaluation metrics (FFR, SI, ERS, RAGAS)
│   ├── experiments/          # Experiment runner and orchestrator
│   ├── config/               # Pydantic Settings
│   └── utils/                # Logging, seeding, hashing
│
├── tests/
│   ├── unit/                 # Fast, no external services (63 tests)
│   └── integration/          # Requires docker compose up
│
├── experiments/              # YAML experiment definitions
│   ├── baseline_v1.yaml
│   └── ablation_attacks.yaml
│
├── docs/                     # Extended documentation
│   ├── ARCHITECTURE.md
│   ├── REPRODUCING.md
│   ├── CONTRIBUTING.md
│   └── DATASETS.md
│
├── infra/
│   └── containerlab/         # Optional distributed topology
│
├── docker-compose.yml        # Development infrastructure
├── pyproject.toml            # Package metadata and dependencies
├── Makefile                  # Common workflows
└── .env.example              # Environment variable template
```

---

## Configuration

All runtime parameters are read from environment variables (prefix `EIGER_`) or a `.env` file. No credentials or host addresses appear in source code.

```bash
# .env (never commit this file)
EIGER_QDRANT_HOST=localhost
EIGER_QDRANT_PORT=6333
EIGER_OLLAMA_HOST=localhost
EIGER_OLLAMA_PORT=11434
EIGER_DEFAULT_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
EIGER_DEFAULT_SEED=42
```

Experiments are fully specified in YAML — no code changes needed to run a new configuration:

```yaml
# experiments/my_experiment.yaml
experiment_id: my_exp_v1
seed: 42
dataset:
  name: averitec
attacks:
  - name: numerical_shift
    poison_rate: 0.3
retriever:
  type: dense
  top_k: 5
llm:
  model: llama3.1:8b
metrics: [ffr, source_integrity, ers]
```

---

## Reproducibility

Every experiment run produces a provenance block alongside its results:

```json
{
  "experiment_id": "ablation_attacks_v1",
  "config_hash": "a3f2c1d8",
  "timestamp": "2026-06-27T14:30:00Z",
  "git_commit": "5258c34",
  "dataset_hash": "sha256:def456...",
  "environment": { "python": "3.11.4", "platform": "linux/amd64" },
  "metrics": { "ffr": 0.31, "source_integrity": 0.62, "ers": 0.74 }
}
```

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the full step-by-step guide.

---

## Extending the Framework

Add a new attack in three steps:

```python
# 1. Subclass BaseAttack
from eiger.core.interfaces import BaseAttack
from eiger.core.models import Document, PoisonedDocument

class MyAttack(BaseAttack):
    name = "my_attack"
    description = "Does something adversarial."

    def apply(self, document: Document, seed: int, **kwargs) -> PoisonedDocument:
        ...

    def describe(self) -> dict:
        return {"attack": self.name}

# 2. Register it
from eiger.attacks.registry import register_attack
register_attack(MyAttack)

# 3. Use it in any YAML config
# attacks:
#   - name: my_attack
#     poison_rate: 0.3
```

The same pattern applies to datasets, metrics, retrievers, and LLM backends. See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for the full guide.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed architecture, dependency graph, design decisions |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | Step-by-step guide to reproduce all paper results |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | How to add attacks, metrics, datasets, and LLM backends |
| [docs/DATASETS.md](docs/DATASETS.md) | Supported datasets, download instructions, schemas |
| [eiger/core/README.md](eiger/core/README.md) | Domain models and abstract interfaces |
| [eiger/attacks/README.md](eiger/attacks/README.md) | Attack taxonomy and implementation details |
| [eiger/metrics/README.md](eiger/metrics/README.md) | Metric definitions, formulas, and implementation notes |
| [experiments/README.md](experiments/README.md) | How to define and run experiments |
| [tests/README.md](tests/README.md) | Testing strategy and how to run each suite |

---

## Citation

If you use EIBench in your research, please cite:

```bibtex
@inproceedings{eiger2026,
  title     = {EIBench: Measuring Epistemic Integrity in Retrieval-Augmented Generation},
  author    = {[Authors]},
  booktitle = {[Venue 2026]},
  year      = {2026},
  url       = {https://doi.org/[DOI]}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
