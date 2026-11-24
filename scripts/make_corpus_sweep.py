"""
Generate the poison-rate sweep configs for the official corpus in git.

Dataset: ``corpus_claim_official_unreviewed`` (verified-true, English rows of
corpus_claim_official/, human review NOT required). Results are therefore
PRELIMINARY: the corpus is not yet human-validated (alignment protocol,
point 6). When enough rows are reviewed, rerun with ``--reviewed-only``,
which switches to the gated ``corpus_claim_official`` dataset.

Poison: the ``corpus_precomputed`` attack, i.e. the collaborator's own
manipulated claim (M01/M02/M04/M06). Each claim has exactly one variant, so
the nominal poison rate equals the expected share of poisoned claims.

The claim subsample is fixed (ordered by SHA-256 of item_id); the seed only
changes which claims receive their poisoned variant.

Answer-level sensitivity (PCS): ``--pcs`` adds the "pcs" metric, so every
record with a poisoned document in its top-k gets a second, counterfactual
generation without the poison (eiger/metrics/pcs.py). Output folders get a
``_pcs`` suffix; retrieval is identical to the run without PCS (same seed,
same corpus), which doubles as a check.

Multi-element variants: ``--corpus multi2`` / ``multi3`` with ``--k`` uses the
rule-based variants that change k numbers or years (eiger/datasets/
multi_element.py; generate data/multi_element/variants.csv first). The same
claims and seeds are used for every k, so k is compared within claim.
Graded variants (E10) on the multi2 claims: ``multi2_grad`` (and
``multi2_grad_noel``, election years left unchanged) with ``--k`` 1..4 = the
level of intensity of a single changed element.

Other families on the multi2 claims: ``multi2_numrand`` (random magnitude),
``multi2_mix`` (numbers, places, months, scale and direction words, fixed
rule), ``multi2_mixrand`` (same elements, random replacement).

Baselines for the multi-element corpora (same claims, same seeds):
- ``--corpus-variants``: poison with the official corpus's own variant of each
  claim instead of the rule-based one (reference for k=1);
- ``--rates 0 --seeds 42``: no attack. At 0% the seed changes nothing (no claim
  is poisoned), so one seed is enough; no ``--k`` needed. This is the
  no-attack answer for ``analyze_answers.py --noattack``.
- ``--retriever sparse|hybrid``: any of the above with BM25 or RRF.

Usage:
    python scripts/make_corpus_sweep.py --seeds 42 43 44 45 46 --max-claims 300
    python scripts/make_corpus_sweep.py --dry-run
    python scripts/make_corpus_sweep.py --queries genq --pcs --rates 10
    python scripts/make_corpus_sweep.py --corpus multi2 --queries genq --k 1 --rates 10 50 --pcs
    python scripts/make_corpus_sweep.py --corpus multi2 --queries genq --corpus-variants --rates 10 50 --pcs
    python scripts/make_corpus_sweep.py --corpus multi2 --queries genq --rates 0 --seeds 42

Output: experiments/corpus_sweep_preliminary/<rate>pct_seed<seed>.yaml
        (or experiments/corpus_sweep_reviewed/ with --reviewed-only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
RATES = [0, 1, 3, 5, 10]


DATASETS = {
    # (corpus, queries, reviewed_only) -> dataset name
    ("official", "template", False): "corpus_claim_official_unreviewed",
    ("official", "template", True): "corpus_claim_official",
    ("official", "genq", False): "corpus_claim_official_unreviewed_genq",
    ("official", "genq", True): "corpus_claim_official_genq",
    ("engineering", "template", False): "corpus_claim_engineering",
    ("engineering", "genq", False): "corpus_claim_engineering_genq",
    # Sensitivity analysis: keep queries that repeat a changed number or year.
    ("official", "genq_all", False): "corpus_claim_official_unreviewed_genq_all",
    ("engineering", "genq_all", False): "corpus_claim_engineering_genq_all",
    # Independent replication: claims that are only in the larger corpus.
    ("engineering_extra", "genq", False): "corpus_claim_engineering_extra_genq",
    # Multi-element variants (k changed numbers/years), same claims for every k.
    ("multi2", "genq", False): "corpus_claim_official_multi2_genq",
    ("multi3", "genq", False): "corpus_claim_official_multi3_genq",
    ("multi2_numrand", "genq", False): "corpus_claim_official_multi2_numrand_genq",
    ("multi2_mix", "genq", False): "corpus_claim_official_multi2_mix_genq",
    ("multi2_mixrand", "genq", False): "corpus_claim_official_multi2_mixrand_genq",
    # Graded variants (E10): same claims as multi2, --k is the level 1..4.
    ("multi2_grad", "genq", False): "corpus_claim_official_multi2_genq",
    ("multi2_grad_noel", "genq", False): "corpus_claim_official_multi2_genq",
}

# Which corpus file the corpus_precomputed attack reads its manipulations from.
VARIANT_SOURCE = {
    "official": "official",
    "engineering": "engineering",
    "engineering_extra": "engineering",
    "multi2": "multi_element",
    "multi3": "multi_element",
    "multi2_numrand": "multi_element",
    "multi2_mix": "multi_element",
    "multi2_mixrand": "multi_element",
    "multi2_grad": "multi_element",
    "multi2_grad_noel": "multi_element",
}
MAX_K = {"multi2": 2, "multi3": 3, "multi2_numrand": 2, "multi2_mix": 2, "multi2_mixrand": 2,
         "multi2_grad": 4, "multi2_grad_noel": 4}
# Variants file family of each multi-element corpus (eiger/datasets/multi_element.py).
FAMILY = {"multi2": "num", "multi3": "num", "multi2_numrand": "numrand",
          "multi2_mix": "mix", "multi2_mixrand": "mixrand",
          "multi2_grad": "grad", "multi2_grad_noel": "grad_noel"}


RETRIEVERS = ("dense", "sparse", "hybrid")
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TOP_K = 5


def model_tag(model: str) -> str:
    """Folder-safe model name: "qwen2.5:7b" -> "qwen2-5-7b"."""
    return "".join(c if c.isalnum() else "-" for c in model.lower()).strip("-")


def sweep_name(
    corpus: str, queries: str, reviewed_only: bool, retriever: str = "dense",
    pcs: bool = False, k: int | None = None, corpus_variants: bool = False,
    no_attack: bool = False, llm_model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K,
) -> str:
    kind = "reviewed" if reviewed_only else "preliminary"
    # Dense sweeps keep their original folder names, so existing results stay valid.
    if corpus_variants:
        corpus_tag = f"{corpus}_corpusvar"
    elif no_attack:
        corpus_tag = f"{corpus}_noattack"
    else:
        corpus_tag = f"{corpus}_k{k}" if k is not None else corpus
    if retriever == "dense":
        if corpus == "official" and queries == "template" and not pcs:
            return f"corpus_sweep_{kind}"
        base = f"corpus_sweep_{corpus_tag}_{queries}"
    else:
        base = f"corpus_sweep_{corpus_tag}_{queries}_{retriever}"
    # Non-default generator or context size get their own folders; the
    # defaults keep the existing names, so existing results stay valid.
    if llm_model != DEFAULT_MODEL:
        base += f"_{model_tag(llm_model)}"
    if top_k != DEFAULT_TOP_K:
        base += f"_top{top_k}"
    return f"{base}_pcs_{kind}" if pcs else f"{base}_{kind}"


def build_config(
    rate_pct: int, seed: int, max_claims: int, reviewed_only: bool,
    corpus: str = "official", queries: str = "template", retriever: str = "dense",
    pcs: bool = False, k: int | None = None, corpus_variants: bool = False,
    no_attack: bool = False, llm_model: str = DEFAULT_MODEL, top_k: int = DEFAULT_TOP_K,
) -> dict:
    kind = "reviewed" if reviewed_only else "preliminary"
    name = sweep_name(
        corpus, queries, reviewed_only, retriever, pcs, k, corpus_variants, no_attack, llm_model, top_k
    )
    tag = f"{rate_pct}pct_seed{seed}"
    params: dict = {"variant_source": "official" if corpus_variants else VARIANT_SOURCE[corpus]}
    if k is not None and not corpus_variants:
        params["k"] = k
        if FAMILY.get(corpus, "num") != "num":
            params["family"] = FAMILY[corpus]
        if FAMILY.get(corpus, "num").startswith("grad"):
            params["missing"] = "skip"
    attacks = (
        [{"name": "corpus_precomputed", "poison_rate": rate_pct / 100, "params": params}]
        if rate_pct > 0 else []
    )
    if no_attack:
        manipulations = "no attack (baseline)"
    elif k is not None and not corpus_variants and FAMILY[corpus] == "num":
        manipulations = f"rule-based multi-element variants, k={k} changed numbers/years"
    elif k is not None and not corpus_variants and FAMILY[corpus].startswith("grad"):
        manipulations = (
            f"rule-based graded variants ({FAMILY[corpus]}), level L{k} of 4: "
            "one element, increasing intensity"
        )
    elif k is not None and not corpus_variants:
        manipulations = f"rule-based multi-element variants ({FAMILY[corpus]}), k={k} changed elements"
    else:
        manipulations = "collaborator's precomputed manipulations (M01/M02/M04/M06)"
    return {
        "experiment_id": f"{name}_{tag}",
        "description": (
            f"{corpus} corpus sweep ({kind}, {queries} queries, {retriever} retrieval"
            + (f", generator {llm_model}" if llm_model != DEFAULT_MODEL else "")
            + (f", top-{top_k}" if top_k != DEFAULT_TOP_K else "")
            + f"), rate={rate_pct}%, seed={seed}, "
            f"{manipulations}." + (" PCS counterfactual generation on." if pcs else "")
            + ("" if reviewed_only else " PRELIMINARY: corpus not yet human-validated.")
        ),
        "seed": seed,
        "dataset": {
            "name": DATASETS[(corpus, queries, reviewed_only)],
            "path": None,
            "split": "test",
            "max_claims": max_claims,
        },
        "allow_non_benchmark_attacks": False,
        "attacks": attacks,
        "retriever": {
            "type": retriever,
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_store": "qdrant",
            "top_k": top_k,
            "collection_name": f"eibench_{name}_{tag}",
        },
        "llm": {"backend": "ollama", "model": llm_model, "temperature": 0.0, "max_tokens": 512},
        "faithfulness_scorer": "embedding",
        "metrics": ["ffr", "source_integrity", "ers", "prr", "prd"] + (["pcs"] if pcs else []),
        "output_dir": f"results/{name}/{tag}/",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-claims", type=int, default=300)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="dense",
                        help="retrieval method; sparse = BM25, hybrid = dense + BM25 fused with RRF")
    parser.add_argument("--corpus", choices=["official", "engineering", "engineering_extra", *MAX_K],
                        default="official",
                        help="engineering_extra = claims only in the larger corpus (disjoint from official); "
                             "multi2/multi3 = official claims with >= 2/3 numeric elements, rule-based "
                             "variants changing --k of them")
    parser.add_argument(
        "--k",
        type=int,
        help="multi-element corpora only: number of changed elements "
        "(multi2_grad*: level of intensity 1..4)",
    )
    parser.add_argument("--corpus-variants", action="store_true",
                        help="multi-element corpora: poison with the official corpus variant (baseline)")
    parser.add_argument("--pcs", action="store_true",
                        help="add the PCS metric (counterfactual generation without the poison)")
    parser.add_argument("--rates", type=int, nargs="+", default=RATES, help=f"poison rates in %% (default {RATES})")
    parser.add_argument("--queries", choices=["template", "genq", "genq_all"], default="template",
                        help="template 'Is it true that ...?'; genq = neutral generated queries (v2, "
                             "numeric-leak rows excluded); genq_all = same, leak rows kept (sensitivity)")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL,
                        help=f"Ollama model (default {DEFAULT_MODEL}); others get their own folder")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"documents in the context (default {DEFAULT_TOP_K}); others get their own folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if args.corpus != "official" and args.reviewed_only:
        parser.error("only the official corpus has human-reviewed rows")
    if (args.corpus, args.queries, args.reviewed_only) not in DATASETS:
        parser.error(f"combination not available: corpus={args.corpus}, queries={args.queries}")
    no_attack = set(args.rates) == {0}
    if args.corpus in MAX_K:
        if args.corpus_variants or no_attack:
            if args.k is not None:
                parser.error("--k is not used with --corpus-variants or with --rates 0")
            if args.corpus_variants and no_attack:
                parser.error("--corpus-variants needs a rate above 0")
        elif args.k is None or not 1 <= args.k <= MAX_K[args.corpus]:
            parser.error(f"--corpus {args.corpus} needs --k between 1 and {MAX_K[args.corpus]}")
        if 0 in args.rates and not no_attack:
            parser.error("for multi-element corpora run the 0% baseline on its own (--rates 0)")
    elif args.k is not None or args.corpus_variants:
        parser.error("--k and --corpus-variants are only for the multi-element corpora")
    no_attack = no_attack and args.corpus in MAX_K
    out_dir = REPO / "experiments" / sweep_name(
        args.corpus, args.queries, args.reviewed_only, args.retriever, args.pcs, args.k,
        args.corpus_variants, no_attack, args.llm_model, args.top_k,
    )
    planned = [
        (out_dir / f"{rate}pct_seed{seed}.yaml",
         build_config(rate, seed, args.max_claims, args.reviewed_only, args.corpus, args.queries,
                      args.retriever, args.pcs, args.k, args.corpus_variants, no_attack,
                      args.llm_model, args.top_k))
        for rate in args.rates
        for seed in args.seeds
    ]
    for path, cfg in planned:
        print(f"{path.relative_to(REPO)}  dataset={cfg['dataset']['name']}  "
              f"max_claims={args.max_claims}  attacks={[a['name'] for a in cfg['attacks']]}")
    if args.dry_run:
        print(f"\n{len(planned)} configs planned (dry run, nothing written).")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "# GENERATED by scripts/make_corpus_sweep.py — do not edit by hand.\n"
        f"# Corpus: {args.corpus}; queries: {args.queries}"
        + (f"; k={args.k}" if args.k is not None else "")
        + ("; official corpus variants (baseline)" if args.corpus_variants else "")
        + ("; no attack (baseline)" if no_attack else "")
        + ("; PCS on" if args.pcs else "")
        + (f"; generator {args.llm_model}" if args.llm_model != DEFAULT_MODEL else "")
        + (f"; top-{args.top_k}" if args.top_k != DEFAULT_TOP_K else "") + ".\n"
        + ("" if args.reviewed_only else "# PRELIMINARY: corpus not yet human-validated.\n")
    )
    for path, cfg in planned:
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nWrote {len(planned)} configs to {out_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
