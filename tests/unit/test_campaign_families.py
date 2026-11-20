"""
Tests for the second campaign: variant families (random magnitude; places,
months, scale and direction words), baseline sweeps, within-sweep and
no-attack answer analysis, single-annotator calibration.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from eiger.core.exceptions import IngestionError
from eiger.core.models import Document
from eiger.datasets import multi_element as me

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeRng:
    """Deterministic stand-in for random.Random: fixed sign, randint, uniform, first choice."""

    def __init__(self, sign: int = 1, integer: int = 1, fraction: float = 0.1) -> None:
        self.sign, self.integer, self.fraction = sign, integer, fraction

    def choice(self, seq):
        seq = list(seq)
        return self.sign if seq == [-1, 1] else seq[0]

    def randint(self, low, high):
        return self.integer

    def uniform(self, low, high):
        return self.fraction


# ─── generator: families, extraction, rules ──────────────────────────────────


def test_family_files() -> None:
    assert me.family_file("num") == me.DEFAULT_MULTI_ELEMENT_FILE
    assert me.family_file("mixrand").name == "variants_mixrand.csv"
    with pytest.raises(IngestionError, match="Unknown multi-element family"):
        me.family_file("names")


def test_extract_entities_and_kinds() -> None:
    text = (
        "Texas and New Mexico added 2 million jobs in January 2020, more than Ohio; "
        "Costs increased. More than 3 billions. In May they met."
    )
    everything = me.FAMILIES["mix"].kinds
    found = {e.value: e.kind for e in me.extract_elements(text, everything)}
    assert found == {
        "Texas": "us_state",
        "New Mexico": "us_state",
        "2": "integer",
        "million": "scale",
        "January": "month",
        "2020": "year",
        "more than": "direction",
        "Ohio": "us_state",
        "increased": "direction",
        "More than": "direction",
        "3": "integer",
        "billions": "scale",
    }  # "Mexico" inside "New Mexico" and "May" are not elements
    numbers = me.extract_elements(text)
    assert [e.value for e in numbers] == ["2", "2020", "3"]


def test_perturb_random_rules() -> None:
    year = me.Element("2020", "year")
    assert me.perturb_random(year, FakeRng(sign=-1, integer=3)) == "2017"
    assert me.perturb_random(me.Element("1.5", "decimal"), FakeRng(sign=1, fraction=0.2)) == "1.8"
    # A decimal that would reach zero goes up instead.
    assert me.perturb_random(me.Element("0.1", "decimal"), FakeRng(sign=-1, fraction=0.5)) == "0.2"
    assert me.perturb_random(me.Element("3", "integer"), FakeRng(sign=-1, integer=5)) == "8"
    assert (
        me.perturb_random(me.Element("1,000", "integer"), FakeRng(sign=1, fraction=0.2)) == "1,200"
    )


def test_perturb_entity_rules() -> None:
    rng = FakeRng()
    assert me.perturb_entity(me.Element("Increased", "direction"), rng, "fixed") == "Decreased"
    assert me.perturb_entity(me.Element("less than", "direction"), rng, "random") == "more than"
    assert me.perturb_entity(me.Element("million", "scale"), rng, "fixed") == "billion"
    assert me.perturb_entity(me.Element("trillion", "scale"), rng, "fixed") == "billion"
    assert me.perturb_entity(me.Element("Millions", "scale"), rng, "fixed") == "Billions"
    assert me.perturb_entity(me.Element("million", "scale"), rng, "fixed", step=2) == "trillion"
    assert me.perturb_entity(me.Element("million", "scale"), rng, "random") == "thousand"
    assert me.perturb_entity(me.Element("Wyoming", "us_state"), rng, "fixed") == "Alabama"
    assert me.perturb_entity(me.Element("Ohio", "us_state"), rng, "fixed", step=2) == "Oregon"
    assert me.perturb_entity(me.Element("December", "month"), rng, "fixed") == "January"
    assert me.perturb_entity(me.Element("April", "month"), rng, "fixed") == "May"
    assert me.perturb_entity(me.Element("Afghanistan", "country"), rng, "random") == "Argentina"


def test_make_variants_mix_tags_and_direction_filter() -> None:
    text = "Sales increased in Texas in 2020 while costs decreased by 5 million."
    rows = me.make_variants("c1", text, max_k=3, family="mix")
    olds = [p.split(" -> ")[0] for p in rows[-1]["changed_element"].split("; ")]
    # increased/decreased are both present: neither is an element.
    assert set(olds) <= {"Texas", "2020", "5", "million"}
    assert all("[" in row["changed_element"] for row in rows)
    assert rows[0]["manipulation_category"] == "MULTI_MIX" and rows[0]["generator_version"] == "2"
    assert all(row["claim_text"] != text for row in rows)


def test_random_family_changes_same_elements_as_fixed() -> None:
    text = "In 2019, 12 states and Texas spent 3.5 billion; Ohio cut 40 jobs in March."
    for fixed, rand in (("num", "numrand"), ("mix", "mixrand")):
        a = me.make_variants("c9", text, family=fixed)
        b = me.make_variants("c9", text, family=rand)
        olds = [[p.split(" -> ")[0] for p in r["changed_element"].split("; ")] for r in a]
        assert olds == [[p.split(" -> ")[0] for p in r["changed_element"].split("; ")] for r in b]
    assert me.make_variants("c9", text, family="mixrand") == me.make_variants(
        "c9", text, family="mixrand"
    )


def test_entity_collisions_and_guard(monkeypatch) -> None:
    rows = me.make_variants("c2", "Alabama and Alaska.", max_k=2, family="mix")
    final = rows[-1]["claim_text"]
    assert "Alabama" not in final and "Alaska" not in final
    news = [p.split(" -> ")[1].split(" [")[0] for p in rows[-1]["changed_element"].split("; ")]
    assert len(set(news)) == 2
    monkeypatch.setattr(me, "perturb_entity", lambda element, rng, mode, step=1: "Alaska")
    with pytest.raises(IngestionError, match="no free replacement"):
        me.make_variants("c2", "Alabama and Alaska.", max_k=1, family="mix")


def test_parse_changes_and_query_leaks() -> None:
    assert me.parse_changes("a -> b [us_state]; 2 -> 3; junk") == [
        ("a", "b", "us_state"),
        ("2", "3", None),
    ]
    row = {"claim_text": "Utah did it.", "changed_element": "Texas -> Utah [us_state]"}
    assert me._query_leaks("mix", "What did Texas do?", "Texas did it.", row)
    assert not me._query_leaks("mix", "What did the state do?", "Texas did it.", row)


@pytest.fixture()
def mix_file(tmp_path: Path) -> Path:
    from eiger.datasets.corpus_claim_official import CorpusClaimOfficialUnreviewedDataset

    rows = []
    for claim in CorpusClaimOfficialUnreviewedDataset().load():
        rows += me.make_variants(claim.claim_id, claim.original_fact, family="mix")
    path = tmp_path / "variants_mix.csv"
    me.write_variants(rows, path)
    return path


def test_family_datasets_and_attack(mix_file: Path, monkeypatch, tmp_path: Path) -> None:
    from eiger.attacks.registry import get_attack

    claims = me.CorpusClaimOfficialMulti2MixDataset(variants_file=mix_file).load()
    assert len(claims) > 100
    assert me.CorpusClaimOfficialMulti2MixRandDataset().variants_file.name == "variants_mixrand.csv"
    assert me.CorpusClaimOfficialMulti2NumRandDataset().family == "numrand"
    attack = get_attack("corpus_precomputed")
    doc = Document(claim_id=claims[0].claim_id, text=claims[0].original_fact)
    p = attack.apply(
        doc,
        seed=1,
        variant_source="multi_element",
        k=2,
        family="mix",
        multi_element_file=str(mix_file),
    )
    assert p.attack_params["family"] == "mix" and p.attack_params["k"] == 2
    assert p.attack_params["manipulation_category"] == "MULTI_MIX"
    # Without an explicit file the family picks data/multi_element/<family file>.
    monkeypatch.setattr(me, "DEFAULT_MULTI_ELEMENT_FILE", tmp_path / "none" / "variants.csv")
    with pytest.raises(IngestionError, match="variants_mixrand.csv"):
        attack.apply(doc, seed=1, variant_source="multi_element", k=1, family="mixrand")


def test_generator_script_families(tmp_path: Path, monkeypatch, capsys) -> None:
    script = _load("make_multi_element_variants")
    out = tmp_path / "one.csv"
    assert script.main(["--family", "mix", "--out", str(out), "--show", "1"]) == 0
    assert out.is_file() and "SHA-256 mix:" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        script.main(["--all", "--out", str(out)])
    monkeypatch.setattr(me, "DEFAULT_MULTI_ELEMENT_FILE", tmp_path / "all" / "variants.csv")
    assert script.main(["--all"]) == 0
    assert sorted(p.name for p in (tmp_path / "all").iterdir()) == sorted(
        f.filename for f in me.FAMILIES.values()
    )
    with (tmp_path / "all" / "variants.csv").open() as fh:
        assert all(row["generator_version"] == "1" for row in csv.DictReader(fh))


# ─── scripts/make_corpus_sweep.py: families and baselines ────────────────────


def test_sweep_baselines_and_families() -> None:
    sweep = _load("make_corpus_sweep")
    assert (
        sweep.sweep_name("multi2", "genq", False, "sparse", True, None, True)
        == "corpus_sweep_multi2_corpusvar_genq_sparse_pcs_preliminary"
    )
    assert (
        sweep.sweep_name("multi2_mix", "genq", False, no_attack=True)
        == "corpus_sweep_multi2_mix_noattack_genq_preliminary"
    )
    var = sweep.build_config(10, 42, 300, False, "multi2", "genq", "dense", True, None, True)
    assert var["attacks"][0]["params"] == {"variant_source": "official"}
    mix = sweep.build_config(50, 43, 300, False, "multi2_mixrand", "genq", "hybrid", True, 2)
    assert mix["attacks"][0]["params"] == {
        "variant_source": "multi_element",
        "k": 2,
        "family": "mixrand",
    }
    assert "(mixrand)" in mix["description"]
    none = sweep.build_config(0, 42, 300, False, "multi2", "genq", no_attack=True)
    assert none["attacks"] == [] and "no attack" in none["description"]


def test_sweep_baseline_cli(monkeypatch, capsys) -> None:
    sweep = _load("make_corpus_sweep")
    bad = (
        ["--corpus", "multi2", "--corpus-variants", "--k", "1"],
        ["--corpus", "multi2", "--corpus-variants", "--rates", "0"],
        ["--corpus", "multi2", "--k", "1", "--rates", "0", "10"],
        ["--corpus", "multi2", "--rates", "0", "--k", "1"],
        ["--corpus-variants"],
    )
    for argv in bad:
        monkeypatch.setattr(sys, "argv", ["x", "--queries", "genq", *argv, "--dry-run"])
        with pytest.raises(SystemExit):
            sweep.main()
    for argv, expected in (
        (["--corpus", "multi2", "--rates", "0", "--seeds", "42"], "1 configs planned"),
        (["--corpus", "multi3", "--corpus-variants", "--rates", "10", "50", "--pcs"], "10 configs"),
        (["--corpus", "multi2_numrand", "--k", "2", "--rates", "10"], "5 configs planned"),
    ):
        monkeypatch.setattr(sys, "argv", ["x", "--queries", "genq", *argv, "--dry-run"])
        assert sweep.main() == 0
        assert expected in capsys.readouterr().out


def test_sweep_generator_and_top_k(monkeypatch, capsys) -> None:
    sweep = _load("make_corpus_sweep")
    assert sweep.model_tag("qwen2.5:7b") == "qwen2-5-7b"
    # Defaults keep the existing folder names.
    assert sweep.sweep_name("multi2", "genq", False, "dense", True, 1) == (
        "corpus_sweep_multi2_k1_genq_pcs_preliminary"
    )
    assert sweep.sweep_name(
        "multi2", "genq", False, "dense", True, 1, llm_model="mistral:7b", top_k=10
    ) == "corpus_sweep_multi2_k1_genq_mistral-7b_top10_pcs_preliminary"
    cfg = sweep.build_config(
        50, 42, 300, False, "multi2", "genq", "dense", True, 1, llm_model="gemma2:9b", top_k=3
    )
    assert cfg["llm"]["model"] == "gemma2:9b"
    assert cfg["retriever"]["top_k"] == 3
    assert "generator gemma2:9b" in cfg["description"] and "top-3" in cfg["description"]
    assert "_gemma2-9b_top3_" in cfg["output_dir"]
    default = sweep.build_config(50, 42, 300, False, "multi2", "genq", "dense", True, 1)
    assert default["llm"]["model"] == "llama3.1:8b" and default["retriever"]["top_k"] == 5
    assert "generator" not in default["description"]
    monkeypatch.setattr(sys, "argv", ["x", "--corpus", "multi2", "--k", "1", "--top-k", "0", "--dry-run"])
    with pytest.raises(SystemExit):
        sweep.main()


def test_sweep_generator_header(monkeypatch, tmp_path) -> None:
    sweep = _load("make_corpus_sweep")
    monkeypatch.setattr(sweep, "REPO", tmp_path)
    argv = ["x", "--corpus", "multi2", "--queries", "genq", "--k", "1", "--rates", "50",
            "--seeds", "42", "--llm-model", "qwen2.5:7b", "--top-k", "10"]
    monkeypatch.setattr(sys, "argv", argv)
    assert sweep.main() == 0
    [path] = list(tmp_path.glob("experiments/*/50pct_seed42.yaml"))
    text = path.read_text()
    assert "generator qwen2.5:7b; top-10." in text
    assert "corpus_sweep_multi2_k1_genq_qwen2-5-7b_top10_preliminary" in str(path)


# ─── scripts/analyze_answers.py: within-sweep, no attack, by kind ────────────


def _hit(rank, doc_type, claim, text, params=None):
    doc = {"doc_type": doc_type, "claim_id": claim, "text": text}
    if params is not None:
        doc["attack_params"] = params
    return {"rank": rank, "document": doc}


def _rec(claim, hits, answer, cf=None, truth=None):
    meta = {}
    if cf is not None:
        meta["counterfactual_answer"] = cf
    if truth is not None:
        meta["ground_truth"] = truth
    return {
        "claim_id": claim,
        "retrieval": {"hits": hits},
        "generation": {"answer": answer, "metadata": meta},
    }


def _write(base: Path, name: str, records: list[dict]) -> None:
    import json

    (base / name).mkdir(parents=True, exist_ok=True)
    (base / name / "results.json").write_text(json.dumps({"records": records}))


def _poisoned_run(kinds: dict[str, str]) -> list[dict]:
    out = []
    for claim, kind in kinds.items():
        truth, variant = f"{claim} was 10 in Texas", f"{claim} was 11 in Utah"
        params = {
            "changed_element": f"Texas -> Utah [{kind}]",
            "variant_source": "multi_element",
        }
        out.append(
            _rec(
                claim,
                [
                    _hit(1, "poisoned", claim, variant, params),
                    _hit(2, "ground_truth", claim, truth),
                ],
                answer=variant,
                cf=truth,
                truth=truth,
            )
        )
    out.append(_rec("CL", [_hit(1, "ground_truth", "CL", "clean truth")], answer="clean answer"))
    out.append(
        _rec(
            "OT",
            [_hit(1, "poisoned", "ZZ", "other", {}), _hit(2, "ground_truth", "OT", "t")],
            answer="other answer",
            cf="other cf",
        )
    )
    return out


def test_change_kind_variants() -> None:
    analysis = _load("analyze_answers")
    tagged = _rec(
        "A",
        [_hit(1, "poisoned", "A", "v", {"changed_element": "x -> y [month]; 1 -> 2 [year]"})],
        "a",
    )
    assert analysis.change_kind(tagged) == "month+year"
    untagged = _rec(
        "A",
        [
            _hit(
                1,
                "poisoned",
                "A",
                "v",
                {"changed_element": "2020 -> 2021; 9 -> 10", "variant_source": "multi_element"},
            )
        ],
        "a",
    )
    assert analysis.change_kind(untagged) == "integer+year"
    weird = _rec(
        "A",
        [
            _hit(
                1,
                "poisoned",
                "A",
                "v",
                {"changed_element": "x -> y", "variant_source": "multi_element"},
            )
        ],
        "a",
    )
    assert analysis.change_kind(weird) == "unknown"
    official = _rec("A", [_hit(1, "poisoned", "A", "v", {"manipulation_category": "M01"})], "a")
    assert analysis.change_kind(official) == "M01"
    assert analysis.change_kind(_rec("A", [_hit(1, "poisoned", "A", "v", {})], "a")) == "unknown"
    assert analysis.change_kind(_rec("A", [_hit(1, "ground_truth", "A", "t")], "a")) is None


def test_analyze_noattack_within_and_by_kind(tmp_path: Path, capsys) -> None:
    analysis = _load("analyze_answers")
    from tests.unit.test_campaign_tools import make_nli

    kinds = {f"C{i}": ("us_state" if i % 2 else "month") for i in range(6)}
    _write(tmp_path / "mix", "10pct_seed42", _poisoned_run(kinds))
    baseline = [
        _rec(c, [_hit(1, "ground_truth", c, f"{c} was 10 in Texas")], answer=f"{c} was 10 in Texas")
        for c in kinds
    ] + [_rec("CL", [_hit(1, "ground_truth", "CL", "clean truth")], answer="clean answer")]
    _write(tmp_path / "na", "0pct_seed42", baseline)
    similarity = lambda a, b: 1.0 if a == b else 0.3  # noqa: E731
    argv = [
        "--sweep",
        f"mix={tmp_path / 'mix'}",
        "--noattack",
        str(tmp_path / "na"),
        "--nli",
        "--by-kind",
    ]
    assert analysis.main(argv, similarity=similarity, nli=make_nli()) == 0
    out = capsys.readouterr().out
    assert "no-attack baseline" in out
    assert "Within each sweep, record by record" in out
    assert "P(entails truth): poisoned - no attack" in out
    assert "clean (check: ~0) | 0.000" in out
    assert "by kind of change" in out and "| month | 3 |" in out and "| us_state | 3 |" in out
    # A no-attack folder without a 0% run is an error.
    _write(tmp_path / "bad", "10pct_seed42", baseline)
    assert analysis.main([*argv[:2], "--noattack", str(tmp_path / "bad"), "--no-pcs"]) == 1
    assert analysis.render_by_kind("x", []) == []


# ─── scripts/calibration_kit.py: single annotator and retest ─────────────────


def test_calibration_single_annotator_and_retest(tmp_path: Path, capsys) -> None:
    kit = _load("calibration_kit")
    from tests.unit.test_campaign_tools import _answer_records, _fill, _write_run, make_nli

    sweep = tmp_path / "sweep"
    _write_run(sweep, "10pct_seed42", _answer_records(True, "A") + _answer_records(False, "B"))
    prefix = tmp_path / "cal" / "b"
    argv = ["sample", "--results-dir", str(sweep), "--n-own", "8", "--n-other", "1"]
    assert kit.main([*argv, "--n-clean", "2", "--out-prefix", str(prefix)]) == 0
    key_path = Path(f"{prefix}_key.json")
    args = kit.argparse.Namespace(key=str(key_path), ragas=False, ragas_model="m")
    assert kit.cmd_score(args, nli=make_nli()) == 0

    a_path = Path(f"{prefix}_annotatore_A.xlsx")
    import json

    ids = [i["item_id"] for i in json.loads(key_path.read_text())["items"]]
    _fill(a_path, {i: ["sì", "no" if n % 2 else "sì", 4, 3] for n, i in enumerate(ids)})
    capsys.readouterr()
    assert kit.main(["compare", "--key", str(key_path), "--a", str(a_path)]) == 0
    out = capsys.readouterr().out
    assert "Items annotated (single annotator): 11 of 11." in out
    assert "## Single annotator" in out and "against the annotator's labels" in out
    assert "NLI entails variant" in out

    assert kit.main(["retest", "--key", str(key_path), "--n", "5"]) == 0
    retest = Path(f"{prefix}_retest.xlsx")
    assert retest.is_file()
    retest_ids = list(kit.read_workbook(retest))
    assert len(retest_ids) == 5
    _fill(retest, {i: ["sì", "sì", 4, 2] for i in retest_ids})
    capsys.readouterr()
    argv = ["compare", "--key", str(key_path), "--a", str(a_path), "--b", str(retest), "--retest"]
    assert kit.main(argv) == 0
    out = capsys.readouterr().out
    assert "in both passes (at least one answer each): 5 of 5." in out
    assert "test-retest" in out and "against the annotator's first pass" in out
    assert kit.main(["compare", "--key", str(key_path), "--a", str(a_path), "--retest"]) == 1
