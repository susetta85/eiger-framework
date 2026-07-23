# eiger.ingestion

This package is responsible for constructing the document corpus that EIGER uses during
retrieval-augmented fact-checking experiments. It takes a list of `Claim` objects and a
set of configured attacks, and produces a mixed corpus of ground-truth documents and
optionally poisoned documents ready for ingestion into the vector store.

---

## Corpus structure

For every claim in the input list, `CorpusBuilder` always adds exactly one ground-truth
document — a faithful representation of the claim's supporting evidence. It then
considers each registered attack in turn. An attack is applied to the claim with
probability `attack_cfg.poison_rate`, producing a `PoisonedDocument` that is added
alongside the ground-truth document.

The final corpus therefore contains:

- **Ground-truth documents** — one per claim, always present.
- **Poisoned documents** — zero or more per claim depending on which attacks fire.
- **`all_documents`** — the concatenation of both lists, in the order they were added.
  This is the list passed to the vector store.

The `poison_ratio` property of `CorpusBuilderResult` reports the fraction of all
documents that are poisoned, which is a useful sanity check before ingestion.

---

## Per-document seeding strategy

Reproducibility is critical in EIGER experiments. Whether an attack fires for a given
claim must be the same across runs, regardless of the order in which other claims or
attacks are processed. To achieve this, each (claim, attack) pair is assigned a
deterministic child seed derived from the experiment root seed:

```
child_seed = derive_seed(root_seed, claim.id, attack.__class__.__name__)
```

This seed is used to create an isolated `random.Random` instance via `make_rng`, which
is then consulted for the Bernoulli draw against `poison_rate`. Because the seed is
derived only from the root seed and the identities of the claim and attack — not from
any processing order — the outcome is stable even if the claim list is reordered or an
unrelated attack is added.

See `eiger.utils.seeding` for the implementation of `derive_seed` and `make_rng`.

---

## API

### `CorpusBuilder`

```python
class CorpusBuilder:
    def __init__(
        self,
        attacks: list[tuple[BaseAttack, AttackConfig]],
        seed: int,
    ) -> None: ...

    def build(self, claims: list[Claim]) -> CorpusBuilderResult: ...
```

- `attacks` — a list of `(attack_instance, attack_config)` pairs. Each `AttackConfig`
  carries at minimum a `poison_rate: float` in `[0, 1]`.
- `seed` — the root seed for all per-(claim, attack) random decisions.

### `CorpusBuilderResult`

```python
@dataclass
class CorpusBuilderResult:
    ground_truth_docs: list[Document]
    poisoned_docs: list[PoisonedDocument]
    all_documents: list[Document]
    poison_ratio: float
```

---

## Usage example

```python
from eiger.ingestion.corpus_builder import CorpusBuilder
from eiger.attacks.numerical_shift import NumericalShiftAttack, NumericalShiftConfig
from eiger.config.settings import get_settings

settings = get_settings()

attacks = [
    (NumericalShiftAttack(), NumericalShiftConfig(poison_rate=0.3)),
]

builder = CorpusBuilder(attacks=attacks, seed=settings.default_seed)
result = builder.build(claims=my_claims)

print(f"Total documents : {len(result.all_documents)}")
print(f"Ground truth    : {len(result.ground_truth_docs)}")
print(f"Poisoned        : {len(result.poisoned_docs)}")
print(f"Poison ratio    : {result.poison_ratio:.2%}")
```

---

## Relationship to the vector store: `IngestionPipeline`

`CorpusBuilder.build()` produces `result.all_documents` but does not write anything to
storage. This separation allows the same `CorpusBuilderResult` to be inspected,
serialised, or used in offline analysis before any network calls are made.

**`IngestionPipeline`** (Sprint 2, Step 4 — implemented) is the component responsible
for the next step: embedding every document in a `CorpusBuilderResult` and upserting it
into a vector store.

```python
from eiger.ingestion import CorpusBuilder, IngestionPipeline
from eiger.retrieval import SentenceTransformerEmbedder
from eiger.vector_stores import QdrantVectorStore

embedder = SentenceTransformerEmbedder()
vector_store = QdrantVectorStore()

corpus = CorpusBuilder(attacks=my_attacks, seed=42).build(claims)

pipeline = IngestionPipeline(
    embedder=embedder, vector_store=vector_store, collection="eiger_corpus",
)
result = pipeline.ingest(corpus)  # reset=True by default: clean corpus per run

print(result.n_documents)     # ground-truth + poisoned
print(result.embedding_dim)   # from embedder.embedding_dim
```

- **`reset=True` by default**: calls `vector_store.reset_collection()` first, so every
  experiment run starts from a clean corpus. Pass `reset=False` to append to an
  existing collection instead.
- **Single `encode()` call**: all document texts are embedded in one batched call
  (`BaseEmbedder.encode()` already batches internally), not one call per document.
- **Empty corpus is a no-op, not an error**: `encode()`/`upsert()` are skipped entirely
  for an empty document list (the collection is still reset if requested).
- Any failure resetting the collection, encoding, or upserting is wrapped in
  `IngestionError` with the original exception chained as `__cause__`.

`DenseRetriever` (see `eiger/retrieval/README.md`) must be constructed with the *same*
embedder/vector_store/collection as `IngestionPipeline` for retrieval scores to be
meaningful — `ExperimentRunner` enforces this by sharing one embedder/vector_store
instance between both.

---

## Dataset loading (still pending)

The current implementation accepts `claims: list[Claim]` directly — there is no
`BaseDataset` implementation yet (`eiger/datasets/` is still empty). Real dataset
loaders for AVeriTeC, PolitiFact, and FactCheck.org remain future work. The
`CorpusBuilder` API is not expected to change; only the source of the `claims` list
will differ once a loader exists.
