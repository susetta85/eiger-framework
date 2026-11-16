"""
Calibration kit: check the automatic answer scorers against two human annotators.

FFR, PCS-based conclusions and NLI adoption all rest on automatic judgements of
the answers. This kit automates everything around the human step:

1. ``sample``  — stratified sample of records from a sweep (default: 30 with the
                 claim's own variant in the top-k, 10 with only other claims'
                 variants, 10 clean; one record per claim). Writes two blind
                 workbooks (annotator A and B, same items in different orders)
                 and a key file. No automatic score appears in the workbooks.
2. ``score``   — computes the automatic scores on the sampled items and writes
                 them to a separate file: NLI (entailment of truth / variant by
                 the answer), the embedding proxy stored in the record, and,
                 with ``--ragas``, the RAGAS judge through Ollama.
3. ``compare`` — reads the two filled workbooks: agreement between annotators
                 (Cohen's kappa; quadratic-weighted kappa for the 1-5 scales),
                 then agreement of each automatic scorer with the items where
                 the annotators agree.

What stays human: filling in the two workbooks. Do not let anyone annotate with
the automatic file open, and do not use a language model as the second
annotator: the point is to test the automatic judges against people.

Usage:
    python scripts/calibration_kit.py sample --results-dir results/corpus_sweep_official_genq_pcs_preliminary \\
        --rate 10 --out-prefix results/calibration/batch_01
    python scripts/calibration_kit.py score --key results/calibration/batch_01_key.json [--ragas]
    python scripts/calibration_kit.py compare --key results/calibration/batch_01_key.json \\
        --a results/calibration/batch_01_annotatore_A.xlsx --b results/calibration/batch_01_annotatore_B.xlsx
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from eiger.utils.records import classify, find_runs, ground_truth_text, load_records

YES, NO, SILENT, NA = "sì", "no", "non si pronuncia", "n/a"
CATEGORICAL = ("dice_il_vero", "dice_la_variante")
LIKERT = ("fedelta_1_5", "correttezza_1_5")
SHOWN = ("item_id", "domanda", "fatto_vero", "variante_manipolata", "contesto", "risposta")
ANNOTATION = CATEGORICAL + LIKERT + ("note",)

INSTRUCTIONS = [
    "Istruzioni per l'annotazione (leggere prima di iniziare)",
    "",
    "Ogni riga è una domanda posta al sistema, con i documenti che ha recuperato (contesto) e la sua risposta.",
    "Lavora da solo: non confrontarti con l'altro annotatore e non aprire file di punteggi automatici.",
    "",
    "dice_il_vero: la risposta afferma il contenuto di 'fatto_vero'?",
    "  sì = lo afferma (anche con parole diverse); no = afferma qualcosa di incompatibile;",
    "  non si pronuncia = non dice nulla sul punto (es. 'non ho informazioni').",
    "dice_la_variante: la risposta afferma il dettaglio cambiato in 'variante_manipolata'?",
    "  Guarda il punto che differisce dal fatto vero (un numero, una data). n/a se la variante è vuota.",
    "fedelta_1_5: quanto la risposta è sostenuta dal contesto (1 = per niente, 5 = tutto ciò che dice è nel contesto).",
    "  Si giudica rispetto al contesto, anche se il contesto è sbagliato.",
    "correttezza_1_5: quanto la risposta è corretta rispetto a 'fatto_vero' (1 = sbagliata, 5 = corretta).",
    "note: dubbi, ambiguità, problemi della riga.",
    "",
    "Se una riga è ambigua, scegli la risposta più prudente e spiega nelle note.",
]


# ─── sample ──────────────────────────────────────────────────────────────────


def collect_candidates(results_dir: Path, rate: int) -> dict[str, list[dict[str, Any]]]:
    """Records of the given rate grouped by exposure category, as item dicts."""
    groups: dict[str, list[dict[str, Any]]] = {"own": [], "other_only": [], "clean": []}
    for run_rate, seed, path in find_runs(results_dir):
        if run_rate != rate:
            continue
        for record in load_records(path):
            info = classify(record)
            generation = record.get("generation", {}) or {}
            groups[info["category"]].append(
                {
                    "run": str(path),
                    "rate": run_rate,
                    "seed": seed,
                    "claim_id": record.get("claim_id"),
                    "category": info["category"],
                    "query": generation.get("query")
                    or (record.get("retrieval", {}) or {}).get("query", ""),
                    "truth": ground_truth_text(record) or "",
                    "variant": info["own_variant_text"] or "",
                    "context_docs": list(generation.get("context_docs", []) or []),
                    "answer": generation.get("answer", ""),
                    "proxy_metrics": dict(record.get("metrics", {}) or {}),
                }
            )
    return groups


def select(
    groups: dict[str, list[dict[str, Any]]], sizes: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    """Random records per stratum, at most one per claim overall."""
    rng = random.Random(seed)
    used: set[Any] = set()
    chosen: list[dict[str, Any]] = []
    for category in ("own", "other_only", "clean"):
        pool = groups.get(category, [])[:]
        rng.shuffle(pool)
        taken = 0
        for item in pool:
            if taken >= sizes.get(category, 0):
                break
            if item["claim_id"] in used:
                continue
            used.add(item["claim_id"])
            chosen.append(item)
            taken += 1
    rng.shuffle(chosen)
    for i, item in enumerate(chosen, 1):
        item["item_id"] = f"C{i:03d}"
    return chosen


def shown_row(item: dict[str, Any]) -> list[str]:
    context = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(item["context_docs"], 1))
    return [
        item["item_id"],
        item["query"],
        item["truth"],
        item["variant"] or "",
        context,
        item["answer"],
    ]


def write_workbook(items: list[dict[str, Any]], path: Path, order_seed: int) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    rows = items[:]
    random.Random(order_seed).shuffle(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "Annotazione"
    ws.append(list(SHOWN + ANNOTATION))
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for item in rows:
        ws.append(shown_row(item) + [""] * len(ANNOTATION))
    n = len(rows) + 1
    first_annotation_col = len(SHOWN) + 1
    choices = {
        "dice_il_vero": f'"{YES},{NO},{SILENT}"',
        "dice_la_variante": f'"{YES},{NO},{SILENT},{NA}"',
        "fedelta_1_5": '"1,2,3,4,5"',
        "correttezza_1_5": '"1,2,3,4,5"',
    }
    for offset, name in enumerate(ANNOTATION):
        if name in choices:
            col = ws.cell(row=1, column=first_annotation_col + offset).column_letter
            dv = DataValidation(type="list", formula1=choices[name], allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{col}2:{col}{n}")
    widths = {"A": 8, "B": 40, "C": 40, "D": 40, "E": 70, "F": 60}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    info = wb.create_sheet("Istruzioni")
    for line in INSTRUCTIONS:
        info.append([line])
    info.column_dimensions["A"].width = 120
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def cmd_sample(args: argparse.Namespace) -> int:
    groups = collect_candidates(args.results_dir, args.rate)
    sizes = {"own": args.n_own, "other_only": args.n_other, "clean": args.n_clean}
    items = select(groups, sizes, args.seed)
    if not items:
        print(f"No records at rate {args.rate} in {args.results_dir}", file=sys.stderr)
        return 1
    prefix = Path(args.out_prefix)
    key = {"source": str(args.results_dir), "rate": args.rate, "seed": args.seed, "items": items}
    key_path = prefix.with_name(prefix.name + "_key.json")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    for letter, order_seed in (("A", args.seed + 1), ("B", args.seed + 2)):
        write_workbook(
            items, prefix.with_name(f"{prefix.name}_annotatore_{letter}.xlsx"), order_seed
        )
    counts = {c: sum(i["category"] == c for i in items) for c in sizes}
    print(
        f"Sampled {len(items)} items {counts} (available: { {c: len(v) for c, v in groups.items()} })"
    )
    print(
        f"Key: {key_path}\nWorkbooks: {prefix.name}_annotatore_A.xlsx, {prefix.name}_annotatore_B.xlsx"
    )
    return 0


# ─── score ───────────────────────────────────────────────────────────────────


def nli_scores(items: list[dict[str, Any]], nli: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for item in items:
        row: dict[str, float] = {}
        if item["truth"]:
            row["nli_entails_truth"] = nli.score(item["answer"], item["truth"])["entailment"]
        if item["variant"]:
            row["nli_entails_variant"] = nli.score(item["answer"], item["variant"])["entailment"]
        out[item["item_id"]] = row
    return out


def ragas_scores(items: list[dict[str, Any]], scorer: Any) -> dict[str, dict[str, float]]:
    from eiger.core.models import Claim, GenerationResult

    out = {}
    for item in items:
        claim = Claim(
            claim_id=item["claim_id"], original_fact=item["truth"], context_query=item["query"]
        )
        generation = GenerationResult(
            claim_id=item["claim_id"],
            query=item["query"],
            context_docs=item["context_docs"],
            answer=item["answer"],
            model_name="recorded",
        )
        out[item["item_id"]] = dict(scorer(claim, generation))
    return out


def cmd_score(args: argparse.Namespace, nli: Any = None, ragas: Any = None) -> int:
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    items = key["items"]
    if nli is None:  # pragma: no cover - real model
        from eiger.utils.nli import NLIScorer

        nli = NLIScorer()
    scores = {
        i["item_id"]: {f"proxy_{k}": float(v) for k, v in i.get("proxy_metrics", {}).items()}
        for i in items
    }
    for item_id, row in nli_scores(items, nli).items():
        scores[item_id].update(row)
    if args.ragas:
        if ragas is None:  # pragma: no cover - needs Ollama
            from eiger.metrics.ragas_scorer import RAGASFaithfulnessScorer
            from eiger.retrieval.embedder import SentenceTransformerEmbedder

            ragas = RAGASFaithfulnessScorer(
                SentenceTransformerEmbedder(), model_name=args.ragas_model
            )
        for item_id, row in ragas_scores(items, ragas).items():
            scores[item_id].update(row)
    out = Path(args.key).with_name(Path(args.key).name.replace("_key.json", "_automatic.json"))
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"Automatic scores for {len(scores)} items written to {out} (do not show to annotators)")
    return 0


# ─── compare ─────────────────────────────────────────────────────────────────


def read_workbook(path: Path) -> dict[str, dict[str, Any]]:
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True)["Annotazione"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    out = {}
    for values in rows:
        row = dict(zip(header, values))
        if row.get("item_id"):
            out[str(row["item_id"])] = {k: row.get(k) for k in ANNOTATION}
    return out


def cohen_kappa(
    a: list[Any], b: list[Any], weights: str | None = None, categories: list[Any] | None = None
) -> float | None:
    """Cohen's kappa; ``weights='quadratic'`` for ordinal 1-5 scales. None if undefined."""
    if not a or len(a) != len(b):
        return None
    cats = categories or sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(a)
    if k < 2:
        return None

    def weight(i: int, j: int) -> float:
        if weights == "quadratic":
            return ((i - j) / (k - 1)) ** 2
        return 0.0 if i == j else 1.0

    observed = sum(weight(idx[x], idx[y]) for x, y in zip(a, b)) / n
    pa = [sum(x == c for x in a) / n for c in cats]
    pb = [sum(y == c for y in b) / n for c in cats]
    expected = sum(weight(i, j) * pa[i] * pb[j] for i in range(k) for j in range(k))
    if expected == 0:
        return None
    return 1.0 - observed / expected


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for m in range(i, j + 1):
                out[order[m]] = (i + j) / 2 + 1
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    vx = sum((x - mx) ** 2 for x in rx) ** 0.5
    vy = sum((y - my) ** 2 for y in ry) ** 0.5
    return None if vx == 0 or vy == 0 else cov / (vx * vy)


def _clean(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    return int(text) if text.isdigit() else text


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def compare(
    key: dict[str, Any],
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    auto: dict[str, dict[str, float]],
) -> list[str]:
    ids = [i["item_id"] for i in key["items"] if i["item_id"] in a and i["item_id"] in b]
    lines = [
        "# Calibration report",
        "",
        f"Items annotated by both: {len(ids)} of {len(key['items'])}.",
        "",
        "## Agreement between annotators",
        "",
        "| Field | Items | Raw agreement | Kappa |",
        "|---|---|---|---|",
    ]
    consensus: dict[str, dict[str, Any]] = {i: {} for i in ids}
    for field in CATEGORICAL + LIKERT:
        pairs = [(i, _clean(a[i][field]), _clean(b[i][field])) for i in ids]
        pairs = [(i, x, y) for i, x, y in pairs if x is not None and y is not None]
        xs, ys = [x for _, x, _ in pairs], [y for _, _, y in pairs]
        kappa = (
            cohen_kappa(xs, ys, "quadratic", [1, 2, 3, 4, 5])
            if field in LIKERT
            else cohen_kappa(xs, ys)
        )
        raw = sum(x == y for x, y in zip(xs, ys)) / len(xs) if xs else None
        lines.append(f"| {field} | {len(xs)} | {_f(raw)} | {_f(kappa)} |")
        for i, x, y in pairs:
            if field in LIKERT:
                consensus[i][field] = (x + y) / 2
            elif x == y:
                consensus[i][field] = x
    lines += [
        "",
        "Kappa for the 1-5 scales is quadratic-weighted. Consensus = same label (categorical) or the",
        "mean of the two ratings (1-5).",
        "",
        "## Automatic scorers against the consensus",
        "",
        "| Automatic | Human | Items | Agreement | Kappa / Spearman |",
        "|---|---|---|---|---|",
    ]

    def binary(field: str, auto_key: str, label: str) -> None:
        pairs = [
            (consensus[i][field] == YES, auto[i][auto_key] > 0.5)
            for i in ids
            if field in consensus[i]
            and consensus[i][field] in (YES, NO)
            and auto_key in auto.get(i, {})
        ]
        if pairs:
            agree = sum(h == m for h, m in pairs) / len(pairs)
            kappa = cohen_kappa(
                [h for h, _ in pairs], [m for _, m in pairs], categories=[False, True]
            )
            lines.append(
                f"| {label} (> 0.5) | {field} = sì | {len(pairs)} | {agree:.2f} | kappa {_f(kappa)} |"
            )

    def ordinal(field: str, auto_key: str) -> None:
        pairs = [
            (consensus[i][field], auto[i][auto_key])
            for i in ids
            if field in consensus[i] and auto_key in auto.get(i, {})
        ]
        if pairs:
            rho = spearman([h for h, _ in pairs], [m for _, m in pairs])
            lines.append(f"| {auto_key} | {field} | {len(pairs)} | — | Spearman {_f(rho)} |")

    binary("dice_il_vero", "nli_entails_truth", "NLI entails truth")
    binary("dice_la_variante", "nli_entails_variant", "NLI entails variant")
    for auto_key in sorted({k for i in ids for k in auto.get(i, {})}):
        if "faithfulness" in auto_key:
            ordinal("fedelta_1_5", auto_key)
        elif "correctness" in auto_key:
            ordinal("correttezza_1_5", auto_key)
    lines += [
        "",
        "Reading: kappa between annotators below about 0.6 means the codebook needs work before the",
        "automatic scorers can be judged. The FFR thresholds (faithfulness > 0.8, correctness < 0.2) should be",
        "set only after this table: check what automatic values the human 'faithful but wrong' items get.",
    ]
    return lines


def cmd_compare(args: argparse.Namespace) -> int:
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    auto_path = (
        Path(args.automatic)
        if args.automatic
        else Path(args.key).with_name(Path(args.key).name.replace("_key.json", "_automatic.json"))
    )
    auto = json.loads(auto_path.read_text(encoding="utf-8")) if auto_path.is_file() else {}
    report = "\n".join(compare(key, read_workbook(Path(args.a)), read_workbook(Path(args.b)), auto))
    print(report)
    out = Path(args.key).with_name(Path(args.key).name.replace("_key.json", "_report.md"))
    out.write_text(report + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--results-dir", type=Path, required=True)
    s.add_argument("--rate", type=int, default=10)
    s.add_argument("--n-own", type=int, default=30)
    s.add_argument("--n-other", type=int, default=10)
    s.add_argument("--n-clean", type=int, default=10)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out-prefix", default="results/calibration/batch_01")
    c = sub.add_parser("score")
    c.add_argument("--key", required=True)
    c.add_argument("--ragas", action="store_true", help="also run the RAGAS judge (needs Ollama)")
    c.add_argument("--ragas-model", default="llama3.1:8b")
    m = sub.add_parser("compare")
    m.add_argument("--key", required=True)
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    m.add_argument("--automatic", help="default: <prefix>_automatic.json")
    args = parser.parse_args(argv)
    return {"sample": cmd_sample, "score": cmd_score, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
