"""
Tests for the graded variant families (E10): one element, four increasing
intensities (eiger/datasets/multi_element.py, make_graded_variants).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from eiger.core.exceptions import IngestionError
from eiger.core.models import Document
from eiger.datasets import multi_element as me

REPO = Path(__file__).resolve().parents[2]


def _kind(text: str, value: str, include_elections: bool = True) -> str | None:
    element = next(e for e in me.extract_elements(text) if e.value == value)
    return me.graded_kind(text, element, include_elections)


class TestGradedKind:
    @pytest.mark.parametrize(
        ("text", "value", "expected"),
        [
            ("Route 66 is old.", "66", None),
            ("Section 8 housing grew.", "8", None),
            ("Up to No. 4 in the list.", "4", None),
            ("Between 65 and 70 people came.", "65", None),
            ("Between 65 and 70 people came.", "70", None),
            ("It took 3 to 5 days.", "3", None),
            ("It took 3 to 5 days.", "5", None),
            ("In 1999 prices fell.", "1999", "year"),
            ("About 40 percent of voters agree.", "40", "percent_points"),
            ("About 40% of people agree.", "40", "percent_points"),
            ("Sales rose 40 percent last year.", "40", "percent_change"),
            ("Costs grew by 250% overall.", "250", "percent_change"),
            ("It costs 12 dollars.", "12", "integer"),
            ("It costs 1.5 dollars.", "1.5", "decimal"),
        ],
    )
    def test_kinds(self, text: str, value: str, expected: str | None) -> None:
        assert _kind(text, value) == expected

    def test_election_years(self) -> None:
        pres = "Obama won the 2012 presidential election easily."
        assert _kind(pres, "2012") == "election_year_presidential"
        assert _kind(pres, "2012", include_elections=False) is None
        assert _kind("Obama won the 2013 presidential election.", "2013") is None
        house = "Republicans won the House in the 2010 midterm elections."
        assert _kind(house, "2010") == "election_year_house"
        assert _kind("Democrats won the House in the 2011 midterm elections.", "2011") is None
        assert _kind("Cruz won the 2018 Senate race in Texas.", "2018") is None
        assert _kind("He won the 2018 primary in Ohio.", "2018") is None
        assert _kind("The 2019 election was close.", "2019") is None
        # Other countries' elections are never changed.
        assert _kind("Putin won Russia's presidential election in 2012.", "2012") is None
        assert _kind("The 2016 presidential election in South Korea.", "2016") is None
        # The whole claim is the context, however far the election word is.
        far = "In May 2020, a lawyer from a small town said he would run in the " + (
            "far future for a seat, as he announced on television, in the next election."
        )
        assert _kind(far, "2020") is None


class TestGradedNumbers:
    def test_quantities_grow_at_every_level(self) -> None:
        assert me._graded_numbers("2", "integer") == ["3", "4", "5", "7"]
        assert me._graded_numbers("100", "integer") == ["115", "140", "200", "350"]
        assert me._graded_numbers("2,448", "integer") == ["2,815", "3,427", "4,896", "8,568"]
        assert me._graded_numbers("2.6", "decimal") == ["3.0", "3.6", "5.2", "9.1"]
        assert me._graded_numbers("0.1", "decimal") == ["0.2", "0.3", "0.4", "0.5"]

    def test_bounded_percentage_points(self) -> None:
        assert me._graded_numbers("30", "percent_points") == ["35", "40", "50", "70"]
        assert me._graded_numbers("97", "percent_points") == ["92", "87", "77", "57"]
        assert me._graded_numbers("60.5", "percent_points") == ["55.5", "50.5", "40.5", "20.5"]


class TestGradedYears:
    def test_generic_and_cycles(self) -> None:
        assert me._graded_years(1990, "year", 1) == ["1991", "1993", "2000", "2015"]
        # Forward would pass LATEST_YEAR: the other direction is used.
        assert me._graded_years(2012, "year", 1) == ["2011", "2009", "2002", "1987"]
        assert me._graded_years(2012, "election_year_presidential", 1) == [
            "2008",
            "2004",
            "2000",
            "1996",
        ]
        assert me._graded_years(2000, "election_year_presidential", 1) == [
            "2004",
            "2008",
            "2012",
            "2016",
        ]
        assert me._graded_years(2010, "election_year_house", -1) == [
            "2008",
            "2006",
            "2004",
            "2002",
        ]

    def test_order_with_other_years_and_bounds(self) -> None:
        # 2021 cannot move past 2022, so only backwards.
        assert me._graded_years(2021, "year", 1, frozenset({2022})) == [
            "2020",
            "2018",
            "2011",
            "1996",
        ]
        assert me._graded_years(2000, "year", 1, frozenset({1999, 2001})) is None
        assert me._graded_years(1801, "year", -1, frozenset({1810})) is None


class TestMakeGradedVariants:
    def test_four_levels_same_element(self) -> None:
        rows = me.make_variants("C1", "In 1990 it cost 100 dollars.", family="grad")
        assert [r["k"] for r in rows] == [1, 2, 3, 4]
        olds = {me.parse_changes(r["changed_element"])[0][0] for r in rows}
        assert len(olds) == 1
        assert all(r["manipulation_category"] == "GRADED_NUM" for r in rows)
        assert all(r["generator_version"] == "3" for r in rows)
        assert len({r["claim_text"] for r in rows}) == 4

    def test_no_usable_element(self) -> None:
        assert me.make_graded_variants("C1", "No numbers here.") == []
        assert me.make_graded_variants("C2", "Route 66 and Section 8.") == []

    def test_election_years_by_family(self) -> None:
        text = "Obama won the 2012 presidential election with 332 electoral votes."
        kinds = {
            seed: me.parse_changes(
                me.make_variants("C", text, generator_seed=seed, family="grad")[0][
                    "changed_element"
                ]
            )[0][2]
            for seed in range(8)
        }
        assert "election_year_presidential" in kinds.values()
        for seed in range(8):
            rows = me.make_variants("C", text, generator_seed=seed, family="grad_noel")
            assert rows[0]["manipulation_category"] == "GRADED_NUM_NOEL"
            assert me.parse_changes(rows[0]["changed_element"])[0][0] == "332"

    def test_never_collides_with_another_element(self) -> None:
        text = "12 apples, 14 pears and 16 plums."
        for seed in range(10):
            rows = me.make_graded_variants("C", text, generator_seed=seed)
            old = me.parse_changes(rows[0]["changed_element"])[0][0]
            news = {me.parse_changes(r["changed_element"])[0][1] for r in rows}
            assert not news & ({"12", "14", "16"} - {old})


def test_precomputed_attack_missing_skip(tmp_path: Path) -> None:
    from eiger.attacks.registry import get_attack

    rows = me.make_graded_variants("C1", "In 1990 it cost 100 dollars.")
    path = tmp_path / "grad.csv"
    me.write_variants(rows, path)
    attack = get_attack("corpus_precomputed")
    doc = Document(claim_id="C1", text="In 1990 it cost 100 dollars.")
    p = attack.apply(doc, seed=1, variant_source="multi_element", k=4, multi_element_file=str(path))
    assert p.text != doc.text
    other = Document(claim_id="missing", text="x")
    same = attack.apply(
        other,
        seed=1,
        variant_source="multi_element",
        k=1,
        multi_element_file=str(path),
        missing="skip",
    )
    assert same is other
    with pytest.raises(IngestionError, match="No manipulated variant"):
        attack.apply(
            other, seed=1, variant_source="multi_element", k=1, multi_element_file=str(path)
        )


def test_sweep_configs_for_graded_corpora() -> None:
    spec = importlib.util.spec_from_file_location(
        "make_corpus_sweep", REPO / "scripts" / "make_corpus_sweep.py"
    )
    sweep = importlib.util.module_from_spec(spec)
    sys.modules["make_corpus_sweep"] = sweep
    spec.loader.exec_module(sweep)
    cfg = sweep.build_config(50, 42, 300, False, "multi2_grad", "genq", "dense", True, 3)
    params = cfg["attacks"][0]["params"]
    assert params == {
        "variant_source": "multi_element",
        "k": 3,
        "family": "grad",
        "missing": "skip",
    }
    assert cfg["dataset"]["name"] == "corpus_claim_official_multi2_genq"
    assert "level L3 of 4" in cfg["description"]
    assert sweep.sweep_name("multi2_grad_noel", "genq", False, "dense", True, 1) == (
        "corpus_sweep_multi2_grad_noel_k1_genq_pcs_preliminary"
    )
