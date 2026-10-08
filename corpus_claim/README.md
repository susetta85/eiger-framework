# RAG claim corpus

This repository contains **4,620 paired claim instances** for RAG experiments. Each instance includes a clean/original claim and a controlled manipulated counterpart.

## Recommended files

- `rag_claims_engineering.csv`: complete machine-readable dataset for the RAG pipeline.
- `claims_part_*.md`: human-readable GitHub previews split into small chunks.

## Core fields

| Field | Description |
|---|---|
| `claim_id` | Shared identifier for the paired clean/poisoned instance. |
| `clean_doc_id` | Unique identifier for the clean document. |
| `poisoned_doc_id` | Unique identifier for the poisoned document. |
| `source_platform` | Fact-checking organization/source. |
| `factcheck_url` | Original fact-check URL, when available. |
| `clean_claim` | Original fact-checked claim used in the clean condition. |
| `poisoned_claim` | Controlled manipulated claim used in the poisoned condition. |
| `language` | ISO language code. |
| `ground_truth_label` | Normalized original verdict (`verified_true` / `verified_false`). |
| `manipulation_category` | Controlled manipulation category (`M01`, `M02`, `M04`, `M06`). |
| `topic` | Main topic. |
| `subtopic` | Subtopic. |
| `risk_level` | Risk level assigned during corpus preparation. |
| `sensitivity_class` | Sensitivity class. |
| `requires_human_review` | Whether the instance requires manual review. |
| `source_version` | Version of the source corpus. |

## Human-readable parts

- [claims_part_01_0001-0200.md](claims_part_01_0001-0200.md) — rows 1–200
- [claims_part_02_0201-0400.md](claims_part_02_0201-0400.md) — rows 201–400
- [claims_part_03_0401-0600.md](claims_part_03_0401-0600.md) — rows 401–600
- [claims_part_04_0601-0800.md](claims_part_04_0601-0800.md) — rows 601–800
- [claims_part_05_0801-1000.md](claims_part_05_0801-1000.md) — rows 801–1000
- [claims_part_06_1001-1200.md](claims_part_06_1001-1200.md) — rows 1001–1200
- [claims_part_07_1201-1400.md](claims_part_07_1201-1400.md) — rows 1201–1400
- [claims_part_08_1401-1600.md](claims_part_08_1401-1600.md) — rows 1401–1600
- [claims_part_09_1601-1800.md](claims_part_09_1601-1800.md) — rows 1601–1800
- [claims_part_10_1801-2000.md](claims_part_10_1801-2000.md) — rows 1801–2000
- [claims_part_11_2001-2200.md](claims_part_11_2001-2200.md) — rows 2001–2200
- [claims_part_12_2201-2400.md](claims_part_12_2201-2400.md) — rows 2201–2400
- [claims_part_13_2401-2600.md](claims_part_13_2401-2600.md) — rows 2401–2600
- [claims_part_14_2601-2800.md](claims_part_14_2601-2800.md) — rows 2601–2800
- [claims_part_15_2801-3000.md](claims_part_15_2801-3000.md) — rows 2801–3000
- [claims_part_16_3001-3200.md](claims_part_16_3001-3200.md) — rows 3001–3200
- [claims_part_17_3201-3400.md](claims_part_17_3201-3400.md) — rows 3201–3400
- [claims_part_18_3401-3600.md](claims_part_18_3401-3600.md) — rows 3401–3600
- [claims_part_19_3601-3800.md](claims_part_19_3601-3800.md) — rows 3601–3800
- [claims_part_20_3801-4000.md](claims_part_20_3801-4000.md) — rows 3801–4000
- [claims_part_21_4001-4200.md](claims_part_21_4001-4200.md) — rows 4001–4200
- [claims_part_22_4201-4400.md](claims_part_22_4201-4400.md) — rows 4201–4400
- [claims_part_23_4401-4600.md](claims_part_23_4401-4600.md) — rows 4401–4600
- [claims_part_24_4601-4620.md](claims_part_24_4601-4620.md) — rows 4601–4620
