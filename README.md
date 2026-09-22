# EIGER · EIBench

**Epistemic Integrity Benchmark for Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing%20(100%25%20coverage)-brightgreen.svg)](tests/)
[![Sprint](https://img.shields.io/badge/sprint-5-blue.svg)](docs/CLAIM_AND_RESEARCH_QUESTIONS.md)

---

## What is EIBench?

EIBench is an open-source research framework for measuring **Epistemic Robustness** in Retrieval-Augmented Generation (RAG) systems.

Standard RAG evaluation asks: *"Is the answer faithful to the retrieved context?"*  
EIBench asks a harder question: *"What happens when the retrieved context is adversarially poisoned?"*

It introduces and operationalises the **Faithful Falsehood** — an answer that is:
- ✅ **Faithful** to the retrieved context (the LLM did its job correctly), and
- ❌ **Factually wrong** relative to independently verified ground truth (the context itself was poisoned)

This failure mode is invisible to standard faithfulness metrics (RAGAS, TruLens, and hallucination benchmarks like [RAGTruth](https://aclanthology.org/2025.emnlp-industry.54/)/[RAGChecker](https://arxiv.org/abs/2408.08067)/FaithJudge) because they measure faithfulness against context, not against independently-verified truth.

**How this differs from existing RAG poisoning/robustness work:** [PoisonedRAG](https://arxiv.org/abs/2402.07867) established knowledge-corruption attacks via adversarially-optimized injected text; [RAGuard](https://arxiv.org/abs/2502.16101) is the closest existing benchmark in spirit, sampling naturally-occurring misinformation from fact-checking sources. EIBench instead applies a controlled taxonomy of factual-edit attacks (numerical, attributional, causal, temporal) to independently-verified true documents, enabling per-error-type ablation rather than a single aggregate attack-success number — see [`docs/ARCHITECTURE.md` §2](docs/ARCHITECTURE.md#2-related-work-and-positioning) for the full literature positioning, differentiation, and an honest list of current limitations (most notably: FFR/ERS today rely on a heuristic, not-yet-validated faithfulness proxy — see that section before treating any FFR/ERS numbers as publication-ready).

---

## Project Claim & Research Questions

**Claim:** RAG relocates hallucination risk from the model to the corpus. A small share of poisoned documents, if stylistically credible, can be retrieved and cited in an answer that is *faithful to its context* and *false relative to the world* — and because a cited source reads as an epistemic warrant to a human reviewer, this is as much an editorial vulnerability as a technical one.

> *Working hypothesis (H2): Faithfulness and Source Integrity are independent evaluation dimensions. As corpus poisoning increases, faithfulness remains high while Source Integrity decreases — leading to a measurable increase in Faithful Falsehoods.*

The full claim, five research questions (RQ1–RQ5) with companion hypotheses, the threat model, the manipulation taxonomy, and an honest gap analysis against the multidisciplinary research proposal live in [`docs/CLAIM_AND_RESEARCH_QUESTIONS.md`](docs/CLAIM_AND_RESEARCH_QUESTIONS.md) — start there if you're new to the project or picking up a task from the roadmap.

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
| 1 | Corpus Builder + Ingestion Pipeline (embed + upsert) | ✅ Sprint 1 + 2 |
| 2 | Poisoning Engine (6 attack types — M01–M06 taxonomy complete) | ✅ Sprint 1 → 5 |
| 3 | Dense retrieval (Qdrant + sentence-transformers); Sparse (BM25); Hybrid (RRF fusion) | ✅ Sprint 2 + 4 + 5 |
| 4 | Llama 3.1 / Mistral via Ollama | ✅ Sprint 2 |
| 5 | FFR, ERS implemented; SI (NLI) falls back to 0.0 without `transformers`/`torch`; FFR's faithfulness signal is pluggable — heuristic embedding proxy (default) or real RAGAS via an Ollama judge (opt-in, Sprint 5, `pip install -e ".[ragas]"`) | ✅ Sprint 1 + 2 → 5 |
| 6 | `ExperimentRunner` orchestration + `results.json` provenance | ✅ Sprint 2 |
| 7 | Dataset registry + `JSONFixtureDataset` loader | ✅ Sprint 3 |
| 8 | `eiger` CLI (`run`/`list-datasets`/`list-attacks`/`list-metrics`) | ✅ Sprint 3 |
| 9 | `SnopesDataset` (4,832 verified-true claims) + `scripts/enrich_snopes_claims.py` (filter/dedupe/LLM context_query) | ✅ Sprint 3 — not yet independently reviewed, see `docs/DATASETS.md` §8 |
| 10 | `AVeriTecDataset` (Supported-label subset, evidence-question context_query — no LLM needed) | ✅ Sprint 3 — loader only, `download()` is a guard not a fetcher; not yet independently reviewed, see `docs/DATASETS.md` §3 |
| 11 | `PolitiFactDataset` (LIAR "true"-label subset, templated context_query — no LLM needed) | ✅ Sprint 3 — loader only, `download()` is a guard not a fetcher; not yet independently reviewed, see `docs/DATASETS.md` §4 |
| 12 | `FactCheckDataset` (CheckThat! mirror, "true"-verdict subset, templated context_query — no LLM needed) | ✅ Sprint 3 — loader only, `download()` is a guard not a fetcher, raw format assumed JSONL (not independently verified); not yet independently reviewed, see `docs/DATASETS.md` §5 |
| — | OpenAI LLM backend, degradation curves / HTML report | 🔄 Future sprints |
| 16 | `RAGASFaithfulnessScorer` (real RAGAS via an Ollama LLM judge), opt-in via `faithfulness_scorer: "ragas"` | ✅ Sprint 5 — see `eiger/metrics/README.md` |
| 13 | Project claim, RQ1–RQ5/H1–H5, threat model, M01–M06 taxonomy, and Sprint 4/5 gap analysis documented | ✅ Sprint 3→4 — see `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` |
| 14 | `SparseRetriever` (BM25 via `rank-bm25`), selectable via `retriever.type: sparse` | ✅ Sprint 4 — see `eiger/retrieval/README.md` |
| 15 | `HybridRetriever` (RRF fusion of dense + sparse), selectable via `retriever.type: hybrid` | ✅ Sprint 5 — see `eiger/retrieval/README.md` |

---

## Attack Taxonomy

| ID | Name | Description | EIBench Type | Project Code |
|----|------|-------------|--------------|--------------|
| `numerical_shift` | Numerical Shift | Swaps adjacent digits: `3.5%` → `35.%` | Type 1 | M01 |
| `date_manipulation` | Date Manipulation | Shifts year references: `2024` → `2019` | Type 2 | M02 |
| `attribution_switch` | Attribution Switch | Replaces sources: `WHO` → `a blog` | Type 3 | M03 |
| `causal_manipulation` | Causal Manipulation | Injects fabricated causal clauses | Type 4 | M04 |
| `cherry_picking` | Cherry-Picking | Deletes a comparative baseline clause: `"4.1%, compared to 6.8% in 2020"` → `"4.1%"` | Type 5 | M05 |
| `missing_context` | Missing Context | Deletes an entire qualifying/caveat sentence | Type 6 | M06 |

All six implemented attacks: deterministic (seed-controlled), isolated (no global state mutation), extensible (plugin registry). `cherry_picking`/`missing_context` complete the project's M01–M06 manipulation taxonomy 1:1 (Sprint 4/5) — see [`docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §5](docs/CLAIM_AND_RESEARCH_QUESTIONS.md#5-manipulation-taxonomy-m01m06).

---

## Quick Start

### 0. Starting from a fresh machine (nothing installed yet)?

```bash
git clone <repo-url>
cd eiger-framework
make bootstrap   # macOS (Homebrew) or Debian/Ubuntu Linux (apt) — installs
                 # Python + Docker if missing, then runs `make setup`.
                 # See setup.sh for exactly what it does on each platform.
```

Then skip to step 3. If you already have Python 3.10+, go straight to step 1.

**Physical machine, VM, doesn't matter** — `make bootstrap`/`setup.sh`
work the same on a physical Mac (via Homebrew + OrbStack/Docker Desktop)
or a Debian/Ubuntu Linux machine or VM (via apt + Docker Engine). There
is no separate "environment" to log into: OrbStack/Docker Desktop/Docker
Engine are just a Docker *engine* running in the background on whichever
machine you're on; `docker`/`docker compose` commands from that machine's
normal terminal work the same either way. **ContainerLab is not
needed** — it's an entirely optional, separate stack
(`infra/containerlab/`) only for distributed network-topology research,
not for running or developing EIGER experiments (see
`docs/ARCHITECTURE.md`, "Distributed Research Stack").

### 1. Prerequisites

- Python 3.10+
- Docker (via OrbStack or Docker Desktop — for Qdrant and Ollama)

### 2. Setup

```bash
git clone <repo-url>
cd eiger-framework

# Create virtual environment and install
make setup
eval "$(make -s activate)"   # activates the venv; run 'make activate' any time to see the plain command

# Copy environment template
make env
# Edit .env if needed (defaults work for local Docker Compose)
```

Note: `make setup` creates the venv as `venv-<hostname>/`, not a plain `venv/` —
this keeps things working even if the repo folder is shared/synced between
multiple machines (a venv embeds OS-specific absolute paths, so a single
shared `venv/` would break on whichever machine set up last).

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
# The default model is already pulled by 'make bootstrap'. To pull it again
# (or after a manual Ollama install), one-time, ~5GB:
make ollama-pull
```

**Via the CLI** (`eiger` console script, or `python -m eiger`):

```bash
eiger list-datasets     # averitec, factcheck_org, json_fixture, politifact, snopes
eiger list-attacks      # numerical_shift, attribution_switch, date_manipulation, causal_manipulation, cherry_picking, missing_context
eiger list-metrics      # ers, ffr, prd, prr, source_integrity

eiger run experiments/config.yaml
```

Example `experiments/config.yaml`:

```yaml
dataset:
  name: json_fixture
retriever:
  collection_name: eiger_baseline_v1
llm:
  model: llama3.1:8b
metrics: [ffr, ers, source_integrity]
output_dir: results/baseline_v1
```

Only `retriever.type: dense`, `sparse` (Sprint 4 — BM25), or `hybrid`
(Sprint 5 — RRF fusion, see `eiger/retrieval/README.md`),
`retriever.vector_store: qdrant`, `llm.backend: ollama`, and
`faithfulness_scorer: embedding` (default), `ragas` (Sprint 5, see
`eiger/metrics/README.md`), or `none` are implemented — the CLI raises a
clear `ConfigurationError` for any other value (e.g. an `openai` backend,
which the config schema already accepts for forward-compatibility but
nothing implements yet).

**Via the Python API** (equivalent to the above, useful in notebooks or when
you need direct control over component construction):

```python
from eiger.experiments import ExperimentRunner
from eiger.retrieval import SentenceTransformerEmbedder
from eiger.vector_stores import QdrantVectorStore
from eiger.llm import OllamaLLM
from eiger.metrics import EmbeddingFaithfulnessScorer
from eiger.core.models import ExperimentConfig, DatasetConfig, RetrieverConfig, LLMConfig

config = ExperimentConfig(
    dataset=DatasetConfig(name="json_fixture"),
    retriever=RetrieverConfig(collection_name="eiger_baseline_v1"),
    llm=LLMConfig(model="llama3.1:8b"),
    metrics=["ffr", "ers", "source_integrity"],
    output_dir="results/baseline_v1",
)
embedder = SentenceTransformerEmbedder()

runner = ExperimentRunner(
    config=config,
    embedder=embedder,
    vector_store=QdrantVectorStore(),
    llm=OllamaLLM(model_name=config.llm.model),
    faithfulness_scorer=EmbeddingFaithfulnessScorer(embedder),  # see FFR note below
)
result = runner.run(my_claims)  # writes results/baseline_v1/results.json
```

**FFR note:** `EmbeddingFaithfulnessScorer` is a cosine-similarity *proxy* for
the faithfulness/answer-correctness signal FFR needs — not RAGAS. Report
FFR computed this way as "FFR (embedding-similarity proxy)". A real RAGAS
integration (`RAGASFaithfulnessScorer`, an Ollama LLM judge) is also
available as of Sprint 5 — opt in via `faithfulness_scorer: "ragas"` in
your experiment config (requires `pip install -e ".[ragas]"`; see
`eiger/metrics/README.md` for the exact pinned versions and known
judge-reliability caveats before trusting its numbers).

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
│   ├── retrieval/            # Retrieval strategies (dense, sparse/BM25, hybrid/RRF — all implemented)
│   ├── vector_stores/        # Vector store adapters (Qdrant implemented; FAISS/Chroma planned)
│   ├── llm/                  # LLM backends (Ollama implemented; OpenAI-compatible planned)
│   ├── metrics/              # Evaluation metrics (FFR, ERS, Source Integrity, PRR, PRD)
│   ├── experiments/          # Experiment runner and orchestrator
│   ├── config/               # Pydantic Settings
│   └── utils/                # Logging, seeding, hashing
│
├── tests/
│   ├── unit/                 # Fast, no external services (485 tests, 100% coverage)
│   └── integration/          # Requires docker compose up
│
├── experiments/              # YAML experiment definitions
│   ├── baseline_v1.yaml
│   ├── ablation_attacks.yaml
│   └── snopes_pilot.yaml
│
├── scripts/                  # Claim intake/enrichment (not part of the eiger package)
│   ├── import_claims_xlsx.py     # xlsx -> unverified candidate JSON
│   └── enrich_snopes_claims.py   # Snopes export -> LLM-enriched claims
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
| [docs/CLAIM_AND_RESEARCH_QUESTIONS.md](docs/CLAIM_AND_RESEARCH_QUESTIONS.md) | Project claim, RQ1–RQ5/H1–H5, manipulation taxonomy, gap analysis |
| [docs/ETHICS_AND_THREAT_MODEL.md](docs/ETHICS_AND_THREAT_MODEL.md) | Adversary model, sensitivity classes, risk levels, go/no-go phase checklist |
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
