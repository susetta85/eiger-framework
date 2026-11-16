"""Tests for the November 2026 campaign tools: NLI helper, record helpers,
multi-element variants, PCS/multi sweep configs, contradiction audit,
answer-level analysis and calibration kit."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import pytest

from eiger.core.exceptions import IngestionError
from eiger.core.models import Document
from eiger.datasets import multi_element as me
from eiger.utils import nli as nli_mod
from eiger.utils import records as rec

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ─── fake NLI pipeline ───────────────────────────────────────────────────────


def _digits(text: str) -> list[str]:
    return [t for t in text.replace(",", " ").replace(".", " ").split() if t.isdigit()]


def fake_pipeline(inputs, top_k=None):
    """Contradiction when the two texts carry different numbers; entailment when hypothesis is in premise."""
    out = []
    for item in inputs:
        p, h = item["text"], item["text_pair"]
        if h and h in p:
            scores = (0.9, 0.05, 0.05)
        elif _digits(p) and _digits(h) and set(_digits(p)) != set(_digits(h)):
            scores = (0.05, 0.05, 0.9)
        else:
            scores = (0.1, 0.8, 0.1)
        out.append(
            [
                {"label": "ENTAILMENT", "score": scores[0]},
                {"label": "neutral", "score": scores[1]},
                {"label": "contradiction", "score": scores[2]},
            ]
        )
    return out


def make_nli() -> nli_mod.NLIScorer:
    return nli_mod.NLIScorer(pipeline=fake_pipeline, batch_size=2)


# ─── eiger.utils.nli ─────────────────────────────────────────────────────────


def test_normalize_scores_flat_and_wrapped() -> None:
    raw = [
        {"label": "entailment", "score": 0.7},
        {"label": "NEUTRAL", "score": 0.2},
        {"label": "Contradiction", "score": 0.1},
    ]
    assert nli_mod.normalize_scores(raw) == {
        "entailment": 0.7,
        "neutral": 0.2,
        "contradiction": 0.1,
    }
    assert nli_mod.normalize_scores([raw])["entailment"] == 0.7


def test_normalize_scores_rejects_two_way_model() -> None:
    with pytest.raises(ValueError, match="3-way"):
        nli_mod.normalize_scores([{"label": "LABEL_0", "score": 1.0}])
    with pytest.raises(ValueError):
        nli_mod.normalize_scores([])


def test_scorer_batches_and_caches() -> None:
    calls = []

    def counting(inputs, top_k=None):
        calls.append(len(inputs))
        return fake_pipeline(inputs, top_k)

    scorer = nli_mod.NLIScorer(pipeline=counting, batch_size=2)
    pairs = [("a 1", "b 2"), ("x", "x"), ("a 1", "b 2"), ("p", "q")]
    out = scorer.scores(pairs)
    assert calls == [2, 1]  # 3 unique pairs, batch size 2
    assert out[0] == out[2] and out[0]["contradiction"] == 0.9
    assert scorer.cache_size == 3
    scorer.scores(pairs)
    assert calls == [2, 1]  # all cached
    assert scorer.score("x", "x")["entailment"] == 0.9
    assert scorer.symmetric_contradiction("in 2020", "in 2021") == 0.9


def test_scorer_accepts_unwrapped_single_output() -> None:
    def flat(inputs, top_k=None):
        return fake_pipeline(inputs, top_k)[0]

    scorer = nli_mod.NLIScorer(pipeline=flat, batch_size=1)
    assert scorer.score("x", "x")["entailment"] == 0.9


def test_scorer_rejects_wrong_output_length() -> None:
    scorer = nli_mod.NLIScorer(pipeline=lambda inputs, top_k=None: [], batch_size=4)
    with pytest.raises(ValueError, match="outputs"):
        scorer.scores([("a", "b")])


def test_scorer_lazy_loads_transformers(monkeypatch) -> None:
    created = {}

    def pipeline(task, model, device):
        created.update(task=task, model=model, device=device)
        return fake_pipeline

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=pipeline))
    scorer = nli_mod.NLIScorer()
    assert scorer.score("a", "a")["entailment"] == 0.9
    assert created == {
        "task": "text-classification",
        "model": nli_mod.DEFAULT_NLI_MODEL,
        "device": -1,
    }


# ─── eiger.utils.records ─────────────────────────────────────────────────────


def hit(rank: int, doc_type: str, claim: str, text: str = "") -> dict:
    return {
        "rank": rank,
        "document": {
            "doc_type": doc_type,
            "claim_id": claim,
            "text": text or f"{claim}-{doc_type}",
        },
    }


def record(
    claim: str, hits: list[dict], answer: str = "ans", meta: dict | None = None, **extra
) -> dict:
    return {
        "claim_id": claim,
        "retrieval": {"hits": hits},
        "generation": {
            "answer": answer,
            "query": f"q {claim}",
            "context_docs": [h["document"].get("text", "") for h in hits],
            "metadata": meta or {},
        },
        **extra,
    }


def test_classify_categories_and_ranks() -> None:
    own = rec.classify(
        record("A", [hit(2, "ground_truth", "A", "truth"), hit(1, "poisoned", "A", "variant")])
    )
    assert own["category"] == "own" and own["own_and_truth"] and own["variant_outranks_truth"]
    assert own["poison_rank1"] and own["own_rank"] == 1 and own["truth_rank"] == 2
    assert own["own_variant_text"] == "variant" and own["truth_text"] == "truth"
    below = rec.classify(record("A", [hit(1, "ground_truth", "A"), hit(2, "poisoned", "A")]))
    assert not below["variant_outranks_truth"] and not below["poison_rank1"]
    no_truth = rec.classify(record("A", [hit(1, "poisoned", "A")]))
    assert no_truth["variant_outranks_truth"] and not no_truth["own_and_truth"]
    other = rec.classify(record("A", [hit(1, "ground_truth", "A"), hit(2, "poisoned", "B")]))
    assert other["category"] == "other_only"
    flagged = rec.classify(
        record("A", [{"rank": 1, "document": {"is_poisoned": True, "claim_id": "B"}}])
    )
    assert flagged["category"] == "other_only"
    clean = rec.classify({"claim_id": "A"})
    assert clean["category"] == "clean" and not clean["poison_rank1"]
    # Unranked hits sort last; a hit without a document is tolerated.
    assert rec.ordered_hits({"retrieval": {"hits": [{"rank": 2}, {}]}})[0] == {"rank": 2}
    assert rec.hit_doc({}) == {}


def test_ground_truth_text_sources() -> None:
    assert rec.ground_truth_text(record("A", [], meta={"ground_truth": "from meta"})) == "from meta"
    assert (
        rec.ground_truth_text(record("A", [hit(1, "ground_truth", "A", "retrieved")]))
        == "retrieved"
    )
    assert rec.ground_truth_text({"claim_id": "A"}) is None


def test_find_runs_and_load_records(tmp_path: Path) -> None:
    for name, payload in (
        ("10pct_seed42", {"records": [{"claim_id": "A"}]}),
        ("0pct_seed43", {}),
        ("notarun", {}),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.json").write_text(json.dumps(payload))
    runs = rec.find_runs(tmp_path)
    assert [(r, s) for r, s, _ in runs] == [(0, 43), (10, 42)]
    assert rec.load_records(runs[1][2]) == [{"claim_id": "A"}]
    assert rec.load_records(runs[0][2]) == []


def test_wilson_interval() -> None:
    assert rec.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = rec.wilson_interval(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert rec.wilson_interval(10, 10)[1] == pytest.approx(1.0)


# ─── eiger.datasets.multi_element ────────────────────────────────────────────


def test_extract_elements_boundaries() -> None:
    text = (
        "In 2020, 1,800-square-foot homes cost $2.00 more, 50% of 12 states; COVID-19, F-35, the 1990s, 9/11, "
        "1st place, and 2020 again."
    )
    elements = me.extract_elements(text)
    assert [e.value for e in elements] == ["2020", "1,800", "2.00", "50", "12"]
    assert [e.kind for e in elements] == ["year", "integer", "decimal", "integer", "integer"]
    assert me.element_kind("1750") == "integer" and me.element_kind("2,024") == "integer"


def test_perturb_rules() -> None:
    rng = random.Random(0)
    assert me.perturb(me.Element("2020", "year"), rng) in {"2019", "2021"}
    assert me.perturb(me.Element("1", "integer"), rng) == "2"
    assert me.perturb(me.Element("0", "integer"), rng) == "1"
    assert me.perturb(me.Element("5", "integer"), rng) in {"4", "6"}
    assert me.perturb(me.Element("100", "integer"), rng) in {"90", "110"}
    assert me.perturb(me.Element("1,800", "integer"), rng) in {"1,620", "1,980"}
    assert me.perturb(me.Element("2.00", "decimal"), rng) in {"1.80", "2.20"}
    low = {me.perturb(me.Element("0.1", "decimal"), random.Random(s)) for s in range(20)}
    assert low == {"0.2"}  # 0.1 - 0.1 = 0 is not allowed: goes up instead


def test_make_variants_nested_and_deterministic() -> None:
    text = "In 2020, 12 states spent 1.5 billion on 300 schools."
    rows = me.make_variants("X", text)
    assert [r["k"] for r in rows] == [1, 2, 3]
    changes = [dict(c.split(" -> ") for c in r["changed_element"].split("; ")) for r in rows]
    assert changes[0].items() <= changes[1].items() <= changes[2].items()  # nested
    assert all(r["claim_original_reference"] == text and r["claim_text"] != text for r in rows)
    assert all(r["n_elements"] == 4 and r["manipulation_category"] == "MULTI_NUM" for r in rows)
    assert me.make_variants("X", text) == rows
    assert me.make_variants("Y", "No numbers here.") == []
    assert len(me.make_variants("Z", "Only 7 here.", max_k=3)) == 1


def test_make_variants_avoids_collisions(monkeypatch) -> None:
    proposals = iter(["11", "10", "13"])  # 11 collides with an element, 10 is the original
    monkeypatch.setattr(me, "perturb", lambda element, rng: next(proposals))
    rows = me.make_variants("X", "From 10 to 11.", max_k=1)
    changed = rows[0]["changed_element"]
    assert changed.endswith("-> 13")


def test_write_and_load_variants(tmp_path: Path) -> None:
    rows = me.make_variants("B", "In 2020 and 2021.") + me.make_variants("A", "Year 1999.")
    path = tmp_path / "v" / "variants.csv"
    digest = me.write_variants(rows, path)
    assert len(digest) == 64 and digest == me.write_variants(list(reversed(rows)), path)
    index = me.load_multi_element_index(path)
    assert set(index) == {("A", 1), ("B", 1), ("B", 2)}
    with pytest.raises(IngestionError, match="make_multi_element_variants"):
        me.load_multi_element_index(tmp_path / "missing.csv")


@pytest.fixture()
def variants_file(tmp_path: Path) -> Path:
    from eiger.datasets.corpus_claim_official import CorpusClaimOfficialUnreviewedDataset

    rows = []
    for claim in CorpusClaimOfficialUnreviewedDataset().load():
        rows += me.make_variants(claim.claim_id, claim.original_fact)
    path = tmp_path / "variants.csv"
    me.write_variants(rows, path)
    return path


def test_multi_datasets_load_same_claims_for_every_k(variants_file: Path, monkeypatch) -> None:
    two = me.CorpusClaimOfficialMulti2Dataset(variants_file=variants_file).load()
    three = me.CorpusClaimOfficialMulti3Dataset(variants_file=variants_file).load()
    assert len(two) > len(three) > 0
    assert {c.claim_id for c in three} <= {c.claim_id for c in two}
    assert all(c.metadata["n_numeric_elements"] >= 3 for c in three)
    assert (
        len(me.CorpusClaimOfficialMulti2Dataset(variants_file=variants_file).load(max_claims=5))
        == 5
    )
    # Leak check runs against OUR variant: a leaking query drops the claim.
    monkeypatch.setattr(me, "numeric_leak_tokens", lambda q, o, m: ["2020"])
    assert me.CorpusClaimOfficialMulti2Dataset(variants_file=variants_file).load() == []


def test_multi_dataset_skips_mismatched_rows(tmp_path: Path, variants_file: Path) -> None:
    index = me.load_multi_element_index(variants_file)
    rows = [dict(r) for r in index.values()]
    for r in rows:
        r["claim_original_reference"] = "something else"
    path = tmp_path / "bad.csv"
    me.write_variants(rows, path)
    assert me.CorpusClaimOfficialMulti2Dataset(variants_file=path).load() == []
    with pytest.raises(IngestionError):
        me.CorpusClaimOfficialMulti2Dataset(variants_file=tmp_path / "none.csv").load()
    assert me.CorpusClaimOfficialMulti2Dataset().variants_file == me.DEFAULT_MULTI_ELEMENT_FILE


def test_precomputed_attack_multi_element_source(variants_file: Path) -> None:
    from eiger.attacks.registry import get_attack

    attack = get_attack("corpus_precomputed")
    claim = me.CorpusClaimOfficialMulti3Dataset(variants_file=variants_file).load(max_claims=1)[0]
    doc = Document(claim_id=claim.claim_id, text=claim.original_fact)
    texts = []
    for k in (1, 2, 3):
        p = attack.apply(
            doc, seed=1, variant_source="multi_element", k=k, multi_element_file=str(variants_file)
        )
        assert p.attack_params["k"] == k and p.attack_params["manipulation_category"] == "MULTI_NUM"
        texts.append(p.text)
    assert len(set(texts)) == 3
    with pytest.raises(IngestionError, match="'k'"):
        attack.apply(doc, seed=1, variant_source="multi_element")
    with pytest.raises(IngestionError, match="multi_element"):
        attack.apply(doc, seed=1, variant_source="nope")
    other = Document(claim_id="missing", text="x")
    with pytest.raises(IngestionError, match="No manipulated variant"):
        attack.apply(
            other,
            seed=1,
            variant_source="multi_element",
            k=1,
            multi_element_file=str(variants_file),
        )


# ─── scripts/make_corpus_sweep.py ────────────────────────────────────────────


def test_sweep_names_and_configs() -> None:
    sweep = _load("make_corpus_sweep")
    assert sweep.sweep_name("official", "template", False) == "corpus_sweep_preliminary"
    assert sweep.sweep_name("official", "genq", False) == "corpus_sweep_official_genq_preliminary"
    assert (
        sweep.sweep_name("official", "genq", False, pcs=True)
        == "corpus_sweep_official_genq_pcs_preliminary"
    )
    assert (
        sweep.sweep_name("official", "template", False, pcs=True)
        == "corpus_sweep_official_template_pcs_preliminary"
    )
    assert (
        sweep.sweep_name("multi2", "genq", False, "sparse", True, 2)
        == "corpus_sweep_multi2_k2_genq_sparse_pcs_preliminary"
    )
    cfg = sweep.build_config(50, 42, 300, False, "multi3", "genq", "dense", True, 3)
    assert cfg["metrics"][-1] == "pcs"
    assert cfg["attacks"][0]["params"] == {"variant_source": "multi_element", "k": 3}
    assert "k=3" in cfg["description"] and "PCS" in cfg["description"]
    plain = sweep.build_config(10, 42, 300, False, "official", "genq")
    assert "pcs" not in plain["metrics"] and plain["attacks"][0]["params"] == {
        "variant_source": "official"
    }
    assert sweep.build_config(0, 42, 300, False)["attacks"] == []


def test_sweep_cli_validation(monkeypatch, capsys) -> None:
    sweep = _load("make_corpus_sweep")
    for argv in (
        ["--corpus", "multi2", "--queries", "genq"],
        ["--corpus", "multi2", "--queries", "genq", "--k", "3"],
        ["--queries", "genq", "--k", "1"],
    ):
        monkeypatch.setattr(sys, "argv", ["x", *argv, "--dry-run"])
        with pytest.raises(SystemExit):
            sweep.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--corpus",
            "multi3",
            "--queries",
            "genq",
            "--k",
            "3",
            "--pcs",
            "--rates",
            "10",
            "50",
            "--dry-run",
        ],
    )
    assert sweep.main() == 0
    assert "10 configs planned" in capsys.readouterr().out


# ─── scripts/contradiction_audit.py ──────────────────────────────────────────


def _write_run(base: Path, name: str, records: list[dict]) -> None:
    (base / name).mkdir(parents=True)
    (base / name / "results.json").write_text(json.dumps({"records": records}))


def test_contradiction_audit_end_to_end(tmp_path: Path, capsys) -> None:
    audit = _load("contradiction_audit")
    clean = [
        record(
            f"C{i}",
            [
                hit(1, "ground_truth", f"C{i}", f"fact {i} alpha"),
                hit(2, "ground_truth", "Z", "other fact beta"),
            ],
        )
        for i in range(10)
    ]
    poisoned = [
        record(
            "A",
            [
                hit(1, "poisoned", "A", "Rate was 21 in 2020"),
                hit(2, "ground_truth", "A", "Rate was 20 in 2020"),
            ],
        ),
        record(
            "B",
            [hit(1, "poisoned", "B", "Sold 5 cars"), hit(2, "ground_truth", "Q", "Unrelated text")],
        ),
        record(
            "C",
            [hit(1, "ground_truth", "C", "Plain fact"), hit(2, "poisoned", "D", "Other variant")],
        ),
    ]
    _write_run(tmp_path, "0pct_seed42", clean)
    _write_run(tmp_path, "10pct_seed42", poisoned)
    assert audit.main(["--results-dir", str(tmp_path)], nli=make_nli()) == 0
    out = capsys.readouterr().out
    assert "calibrated on 0% runs" in out and "fixed" in out
    rows = (tmp_path / "contradiction_audit_records.csv").read_text().splitlines()
    assert len(rows) == 14
    assert (tmp_path / "contradiction_audit.md").is_file()


def test_contradiction_audit_helpers(tmp_path: Path, capsys) -> None:
    audit = _load("contradiction_audit")
    assert audit.quantile([0.0, 1.0], 0.5) == 0.5 and audit.quantile([3.0], 0.9) == 3.0
    with pytest.raises(ValueError):
        audit.quantile([], 0.5)
    assert audit.calibrate_threshold([{"rate": 10, "max_contradiction": 1.0}], 0.05) is None
    assert audit._fmt_prop(0, 0).startswith("n/a")
    row = audit.score_record(record("A", [hit(1, "poisoned", "A", "x 1")]), make_nli())
    assert (
        row["group"] == "own_without_truth"
        and row["max_contradiction"] == 0.0
        and row["own_pair_contradiction"] is None
    )
    # No 0% run: only the fixed threshold.
    _write_run(
        tmp_path / "s",
        "10pct_seed1",
        [record("A", [hit(1, "poisoned", "A", "a 2"), hit(2, "ground_truth", "A", "a 3")])],
    )
    assert (
        audit.main(["--results-dir", str(tmp_path / "s"), "--max-records", "1"], nli=make_nli())
        == 0
    )
    assert "calibrated threshold not available" in capsys.readouterr().out
    assert audit.main(["--results-dir", str(tmp_path / "empty")], nli=make_nli()) == 1


# ─── scripts/analyze_answers.py ──────────────────────────────────────────────


def _answer_records(variant_answer: bool, prefix: str = "A") -> list[dict]:
    out = []
    for n in range(6):
        i = f"{prefix}{n}"
        truth, variant = f"Population was {100 + n} in 2020", f"Population was {200 + n} in 2020"
        answer = variant if variant_answer else truth
        out.append(
            record(
                i,
                [hit(1, "poisoned", i, variant), hit(2, "ground_truth", i, truth)],
                answer=answer,
                meta={"ground_truth": truth, "counterfactual_answer": truth},
            )
        )
    out.append(
        record(
            f"{prefix}O",
            [hit(1, "poisoned", "P", "x"), hit(2, "ground_truth", f"{prefix}O", "o")],
            answer="o",
            meta={"counterfactual_answer": "o"},
        )
    )
    out.append(record(f"{prefix}Cl", [hit(1, "ground_truth", f"{prefix}Cl", "c")]))
    return out


def test_analyze_answers_paired(tmp_path: Path, capsys) -> None:
    analysis = _load("analyze_answers")
    for label, adopts in (("k1", False), ("k2", True)):
        _write_run(tmp_path / label, "50pct_seed42", _answer_records(adopts))
        _write_run(tmp_path / label, "10pct_seed42", _answer_records(adopts))
    similarity = lambda a, b: 1.0 if a == b else 0.4  # noqa: E731
    out_md = tmp_path / "report.md"
    argv = [
        "--sweep",
        f"k1={tmp_path / 'k1'}",
        "--sweep",
        f"k2={tmp_path / 'k2'}",
        "--rates",
        "50",
        "--nli",
        "--out",
        str(out_md),
    ]
    assert analysis.main(argv, similarity=similarity, nli=make_nli()) == 0
    out = capsys.readouterr().out
    assert "Rate 50%" in out and "Rate 10%" not in out and "Paired differences" in out
    assert "| Answer adopts variant (NLI) | 0.000" in out
    assert "| Answer adopts variant (NLI) | 1.000" in out.split("Paired differences")[1]
    assert out_md.is_file()


def test_analyze_answers_edges(tmp_path: Path, capsys) -> None:
    analysis = _load("analyze_answers")
    assert analysis.cluster_bootstrap([], "pcs") == (None, None, None, 0)
    assert analysis._fmt((None, None, None, 0)) == "n/a"
    row = analysis.record_row(
        record("A", [hit(1, "poisoned", "A", "v 1")], meta={"ground_truth": "t 2"}),
        None,
        make_nli(),
    )
    assert row["pcs"] is None and "real_adopts" in row and "cf_adopts" not in row
    with pytest.raises(SystemExit):
        analysis.main(["--sweep", "nolabel", "--no-pcs"])
    assert analysis.main(["--sweep", f"x={tmp_path}", "--no-pcs"]) == 1
    _write_run(tmp_path / "one", "10pct_seed1", _answer_records(False))
    assert (
        analysis.main(
            ["--sweep", f"only={tmp_path / 'one'}", "--no-pcs"], similarity=lambda a, b: 1.0
        )
        == 0
    )
    assert "PCS, own exposure" not in capsys.readouterr().out


# ─── scripts/calibration_kit.py ──────────────────────────────────────────────


def _fill(path: Path, answers: dict[str, list]) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    ws = wb["Annotazione"]
    header = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        item = row[0].value
        for field, value in zip(
            ("dice_il_vero", "dice_la_variante", "fedelta_1_5", "correttezza_1_5"),
            answers.get(item, [None] * 4),
        ):
            row[header.index(field)].value = value
    wb.save(path)


def test_calibration_kit_full_cycle(tmp_path: Path, capsys) -> None:
    kit = _load("calibration_kit")
    sweep = tmp_path / "sweep"
    records = _answer_records(True, "A") + _answer_records(False, "B")
    for i, r in enumerate(records):
        r["metrics"] = {"ragas_faithfulness": 0.9, "ragas_answer_correctness": 0.3 + 0.01 * i}
    _write_run(sweep, "10pct_seed42", records)
    _write_run(sweep, "0pct_seed42", [record("Z", [hit(1, "ground_truth", "Z")])])
    prefix = tmp_path / "cal" / "batch"
    assert (
        kit.main(
            [
                "sample",
                "--results-dir",
                str(sweep),
                "--rate",
                "10",
                "--n-own",
                "8",
                "--n-other",
                "1",
                "--n-clean",
                "5",
                "--out-prefix",
                str(prefix),
            ]
        )
        == 0
    )
    key_path = Path(f"{prefix}_key.json")
    key = json.loads(key_path.read_text())
    assert len(key["items"]) == 8 + 1 + 2  # clean stratum has only 2 records
    assert len({i["claim_id"] for i in key["items"]}) == len(key["items"])

    class FakeRagas:
        def __call__(self, claim, generation):
            return {"ragas_faithfulness": 0.8, "ragas_answer_correctness": 0.5}

    args = kit.argparse.Namespace(key=str(key_path), ragas=True, ragas_model="m")
    assert kit.cmd_score(args, nli=make_nli(), ragas=FakeRagas()) == 0
    auto = json.loads(Path(f"{prefix}_automatic.json").read_text())
    assert set(auto) == {i["item_id"] for i in key["items"]}

    a_path, b_path = Path(f"{prefix}_annotatore_A.xlsx"), Path(f"{prefix}_annotatore_B.xlsx")
    ids = [i["item_id"] for i in key["items"]]
    a_answers = {
        i: ["sì", "no" if n % 2 else "sì", 5, 4 if n % 3 else 2] for n, i in enumerate(ids)
    }
    b_answers = {
        i: ["sì", "no" if n % 2 else "Sì", 4, 4 if n % 3 else 1] for n, i in enumerate(ids)
    }
    b_answers[ids[0]] = [None, None, None, None]
    _fill(a_path, a_answers)
    _fill(b_path, b_answers)
    assert (
        kit.main(["compare", "--key", str(key_path), "--a", str(a_path), "--b", str(b_path)]) == 0
    )
    report = capsys.readouterr().out
    assert "Agreement between annotators" in report and "NLI entails variant" in report
    assert "Spearman" in report
    assert f"(at least one answer each): {len(ids) - 1} of {len(ids)}." in report
    assert "WARNING" not in report
    assert Path(f"{prefix}_report.md").is_file()
    # Missing automatic file: comparison still runs.
    assert (
        kit.main(
            [
                "compare",
                "--key",
                str(key_path),
                "--a",
                str(a_path),
                "--b",
                str(b_path),
                "--automatic",
                str(tmp_path / "none.json"),
            ]
        )
        == 0
    )


def test_calibration_statistics() -> None:
    kit = _load("calibration_kit")
    assert kit.cohen_kappa([], []) is None
    assert kit.cohen_kappa(["a"], ["a"]) is None  # one category
    assert kit.cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert kit.cohen_kappa([1, 1], [1, 1], categories=[1, 2]) is None  # expected disagreement 0
    assert kit.cohen_kappa([1, 5, 3], [1, 5, 3], "quadratic", [1, 2, 3, 4, 5]) == 1.0
    assert kit.spearman([1, 2], [1, 2]) is None
    assert kit.spearman([1, 2, 3], [1, 1, 1]) is None
    assert kit.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert (
        kit._clean("") is None
        and kit._clean(4.0) == 4
        and kit._clean(" 3 ") == 3
        and kit._clean("Sì") == "sì"
    )
    assert kit.main(["sample", "--results-dir", "/nonexistent", "--out-prefix", "/tmp/x"]) == 1


def test_calibration_compare_empty_workbooks() -> None:
    kit = _load("calibration_kit")
    key = {"items": [{"item_id": "x", "claim_id": "c"}, {"item_id": "y", "claim_id": "d"}]}
    empty = {i: {f: None for f in kit.CATEGORICAL + kit.LIKERT} for i in ("x", "y")}
    report = "\n".join(kit.compare(key, empty, empty, {}))
    assert "(at least one answer each): 0 of 2." in report
    assert "WARNING: the workbooks have no answers yet" in report


def test_contradiction_audit_near_duplicate_filter(tmp_path: Path, capsys) -> None:
    audit = _load("contradiction_audit")
    assert audit.token_overlap("", "") == 0.0
    assert audit.token_overlap("Rate was 21 in 2020", "Rate was 20 in 2020") == pytest.approx(4 / 6)
    clean = [
        record(
            f"C{i}",
            [
                hit(1, "ground_truth", f"C{i}", f"fact {i} alpha"),
                hit(2, "ground_truth", "Z", "other fact beta 7"),
            ],
        )
        for i in range(10)
    ]
    own = record(
        "A",
        [
            hit(1, "poisoned", "A", "Rate was 21 in 2020"),
            hit(2, "ground_truth", "A", "Rate was 20 in 2020"),
        ],
    )
    _write_run(tmp_path, "0pct_seed42", clean)
    _write_run(tmp_path, "10pct_seed42", [own])
    nli = make_nli()
    unfiltered = audit.score_record(clean[0], nli)
    filtered = audit.score_record(clean[0], nli, min_overlap=0.4)
    assert (
        unfiltered["n_pairs"] == 1
        and filtered["n_pairs"] == 0
        and filtered["max_contradiction"] == 0.0
    )
    assert audit.score_record(own, nli, min_overlap=0.4)["max_contradiction"] == 0.9
    assert audit.main(["--results-dir", str(tmp_path), "--min-overlap", "0.4"], nli=nli) == 0
    out = capsys.readouterr().out
    assert "near-duplicate pairs only (word Jaccard >= 0.4)" in out
    assert (tmp_path / "contradiction_audit_overlap0.4.md").is_file()
    assert (tmp_path / "contradiction_audit_overlap0.4_records.csv").is_file()
    assert not (tmp_path / "contradiction_audit.md").exists()
