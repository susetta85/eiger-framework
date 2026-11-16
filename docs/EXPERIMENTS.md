# Esperimenti EIBench: documentazione e risultati

Stato al 4 novembre 2026, dopo la revisione del codice (`docs/CODE_REVIEW.md`). Sostituisce le versioni del 29 ottobre e del 3 novembre. Descrive gli esperimenti eseguiti, i risultati e i limiti con cui vanno letti. Per l'ambiente e la riproduzione vedi `ENVIRONMENT.md` e `docs/REPRODUCING.md`.

## 1. Sintesi

Lo sweep su Snopes è stato rieseguito dopo la revisione del codice del 4 novembre, che ha trovato un errore importante: metà dei documenti "avvelenati" erano copie identiche del documento vero, prodotte da attacchi che non avevano trovato nulla da cambiare. Ora sono esclusi dal corpus. I numeri del 3 novembre (PRR 0.82 al 10%) erano quindi sovrastimati e non vanno più citati.

Con i soli quattro attacchi del benchmark (M01, M02, M04, M06), cinque seed e il codice corretto, la quota di query con almeno un documento avvelenato nel top-5 (PRR) cresce in modo monotono con il tasso di avvelenamento, da 0.00 a **0.54 al 10%** (IC 95% 0.49–0.58). Il risultato resta coerente con H1 (RQ1), con un effetto più piccolo di quanto riportato prima.

Due risultati nuovi, comuni ai due corpus:

- **L'esposizione è soprattutto collaterale.** Al 10%, su Snopes, 0.16 delle query contengono la variante avvelenata del proprio claim e 0.37 contengono solo varianti di *altri* claim. Sul corpus ufficiale il rapporto è simile (0.10 contro 0.17).
- **Quando un claim è avvelenato, la sua variante entra quasi sempre nel contesto**, ma raramente al primo posto.

Limiti invariati: FFR non è misurabile con lo scorer proxy (sezione 5); la Source Integrity, ora calcolata correttamente, ha un problema di definizione (sezione 3.2); l'esperimento sul corpus ufficiale (sezione 4) è preliminare.

## 2. Setup comune

| Componente | Valore |
|---|---|
| Dataset (sez. 3) | Snopes verified-true ripulito (`data/snopes/snopes_enriched.json`, 2.928 claim, 0 contaminati), primi 100 claim, split `test` |
| SHA-256 del dataset | identico sulle due macchine (registrato in `reference/data_hashes.txt`) |
| Attacchi | M01 `numerical_shift`, M02 `date_manipulation`, M04 `causal_manipulation`, M06 `missing_context`; M03 e M05 esclusi, `allow_non_benchmark_attacks: false` |
| Tasso di avvelenamento | 0, 1, 3, 5, 10% (applicato indipendentemente a ogni attacco) |
| Seed | 42, 43, 44, 45, 46 (25 run in totale) |
| Retriever | Denso, `sentence-transformers/all-MiniLM-L6-v2`, top-k = 5 |
| Vector store | Qdrant server 1.9.4 (Docker), `qdrant-client` 1.9.2 |
| Generazione | `llama3.1:8b` via Ollama 0.40.13 (server), temperatura 0.0, max 512 token |
| Scorer di faithfulness | `EmbeddingFaithfulnessScorer` (proxy a similarità coseno) |
| Ambiente Python | 3.12.16, `requirements/lock-darwin-arm64.txt` (hash SHA-256): numpy 1.26.4, pyarrow 25.0.1, torch 2.15.0, transformers 4.57.6, sentence-transformers 3.4.1, ragas 0.2.15 |
| Macchina | Mac-116, Apple Silicon (darwin-arm64) |
| Commit | `270c9d6` (configurazioni generate da `scripts/make_benchmark_sweep.py`; correzioni della revisione incluse) |

Verifiche prima dello sweep: 776 test unitari e 5 test di integrazione superati (incluso il test live con Qdrant e Ollama). Due run identiche di `baseline_v1` e di `0pct_seed42` producono `results.json` identici, escluso il timestamp.

## 3. Risultati: sweep del tasso di avvelenamento (Snopes)

### 3.1 Metriche di retrieval

Commit `270c9d6`. Media ± deviazione standard sui cinque seed; IC 95% con bootstrap (10.000 ricampionamenti) sui 500 valori per record di ogni tasso.

| Tasso | PRR (≥1 avvelenato in top-5) | IC 95% | PRD@1 (avvelenato al rank 1) | IC 95% |
|---|---|---|---|---|
| 0% | 0.000 ± 0.000 | [0.000, 0.000] | 0.000 ± 0.000 | [0.000, 0.000] |
| 1% | 0.116 ± 0.048 | [0.088, 0.144] | 0.006 ± 0.005 | [0.000, 0.014] |
| 3% | 0.254 ± 0.030 | [0.216, 0.292] | 0.012 ± 0.008 | [0.004, 0.022] |
| 5% | 0.342 ± 0.045 | [0.300, 0.384] | 0.020 ± 0.014 | [0.010, 0.032] |
| 10% | 0.538 ± 0.016 | [0.494, 0.582] | 0.046 ± 0.029 | [0.028, 0.064] |

Scomposizione del PRR (vedi anche sezione 4.5):

| Tasso | PRR | variante del claim interrogato (mirata) | solo varianti di altri claim (collaterale) |
|---|---|---|---|
| 1% | 0.116 | 0.020 | 0.096 |
| 3% | 0.254 | 0.050 | 0.204 |
| 5% | 0.342 | 0.082 | 0.260 |
| 10% | 0.538 | 0.164 | 0.374 |

Lettura. L'esposizione cresce in modo monotono e gli intervalli di tassi adiacenti non si sovrappongono. Circa due terzi dell'esposizione sono collaterali: un attacco su pochi claim contamina anche le risposte su claim non attaccati, semanticamente vicini. Il documento avvelenato arriva al primo posto in meno del 5% delle query anche al 10%.

Confronto con le versioni precedenti, al 10%: PRR 0.77 (29 ottobre, un seed, sei attacchi), 0.82 (3 novembre, con le copie identiche), 0.54 (ora). La differenza tra le ultime due è dovuta interamente alle copie identiche escluse.

### 3.2 Metriche diagnostiche (non sono risultati)

| Tasso | Source Integrity | ERS | FFR (proxy) |
|---|---|---|---|
| 0% | 0.189 ± 0.000 | 0.000 ± 0.000 | 0.000 |
| 1% | 0.191 ± 0.002 | 0.796 ± 0.033 | 0.000 |
| 3% | 0.194 ± 0.002 | 0.784 ± 0.011 | 0.000 |
| 5% | 0.198 ± 0.002 | 0.779 ± 0.011 | 0.000 |
| 10% | 0.207 ± 0.004 | 0.777 ± 0.009 | 0.000 |

**Source Integrity.** Ora è calcolata come previsto dal README: media su top-5 di P(implicazione | documento, claim), con il claim come riferimento (prima della revisione usava la query, ed era errata). Il calcolo è verificato: 0.992 per il claim confrontato con se stesso, 0.084 con l'anno cambiato, 0.0 per un testo non correlato. Però la metrica **sale** leggermente con l'avvelenamento. Il motivo è la definizione: un documento su un altro claim e un documento che contraddice il claim ricevono entrambi circa 0, quindi il valore di base è circa 0.2 (il solo documento vero su cinque), e una variante avvelenata, simile al claim, riceve un po' di implicazione e fa salire la media. La definizione va rivista (proposta nella sezione 7).

**ERS** è la media delle annotazioni fissate dal disegno degli attacchi, calcolata solo sui record con documenti annotati: è quasi costante per costruzione. **FFR** è 0 per costruzione dello scorer proxy (sezione 5).

### 3.3 Avvertenze di lettura

- **Tasso nominale e manipolazioni effettive.** Ogni attacco si applica in modo indipendente, e circa metà delle applicazioni non produce alcuna modifica (soprattutto l'attacco numerico, perché molti claim non contengono cifre): queste applicazioni sono escluse dal corpus. Media sui cinque seed, 100 claim per run:

  | Tasso nominale | Documenti davvero modificati | Applicazioni senza effetto (escluse) | Quota del corpus avvelenata |
  |---|---|---|---|
  | 0% | 0.0 | 0.0 | 0.0% |
  | 1% | 2.0 | 2.8 | 2.0% |
  | 3% | 5.2 | 6.0 | 4.9% |
  | 5% | 8.4 | 9.6 | 7.7% |
  | 10% | 17.6 | 21.0 | 15.0% |

  La quota del corpus avvelenata è circa una volta e mezza il tasso nominale, perché i quattro attacchi si sommano. Quando si riporta il PRR, va indicata accanto al tasso nominale: al 10% nominale, il 15% dei documenti del corpus è avvelenato. Fonte: campi `n_poisoned`, `skipped_noop` e `poison_ratio` della riga `corpus_builder.complete` nei log.

- **Query che ripetono il valore vero.** 17 delle 100 query Snopes ripetono un numero o un anno presente nel claim. Quando l'attacco cambia proprio quel valore, la query favorisce il documento vero, e il PRD@1 è sottostimato. La scelta di un prompt che eviti numeri e date è in discussione con la collega.
- **Intervallo di confidenza.** Il bootstrap tratta i 500 record di un tasso come indipendenti, ma i cinque seed usano gli stessi 100 claim; la deviazione standard tra seed è la misura più prudente.
- **Parità tra macchine.** Tutte le run sono su Mac-116. La verifica su un'altra macchina non è ancora stata fatta.

## 4. Esperimento sul corpus ufficiale in git (preliminare)

### 4.1 Cosa cambia

Il corpus della collega è versionato in `corpus_claim_official/`: 3.638 claim originali (`Original_Claims/`) e, per ognuno, la sua manipolazione controllata (`Modified_Claims/`), con categoria e elemento modificato. Due componenti nuovi lo rendono utilizzabile:

- dataset `corpus_claim_official` e `corpus_claim_official_unreviewed` (`eiger/datasets/corpus_claim_official.py`);
- attacco `corpus_precomputed` (`eiger/attacks/corpus_precomputed.py`), che usa come documento avvelenato la manipolazione già prodotta dalla collega invece di un attacco meccanico.

Ogni claim ha una sola variante manipolata, quindi qui il tasso nominale coincide con la quota attesa di claim avvelenati: la prima avvertenza della sezione 3.3 non si applica.

### 4.2 Quali righe entrano

| Filtro | Righe escluse |
|---|---|
| Etichetta diversa da `verified_true` | 1.951 |
| Lingua diversa da inglese | 3 |
| Non revisionate da un umano (solo per `corpus_claim_official`) | 1.659 |

Restano 1.684 claim verified-true in inglese. Di questi, solo **25** sono revisionati (`kept_after_manual_audit` o `approved_final_version`); i 230 `corrected_after_manual_audit` non contano come revisionati, perché il loro output è ancora da ricontrollare. Le categorie presenti sono M06 (814), M01 (581), M02 (287) e M04 (2): M04 è quasi assente.

I claim `verified_false` sono esclusi perché il documento ground-truth è il testo del claim stesso: per un claim falso servirebbe un documento correttivo, che il corpus non contiene.

### 4.3 Configurazione

25 run (5 tassi × 5 seed) su 300 claim del dataset `corpus_claim_official_unreviewed`, generati da `scripts/make_corpus_sweep.py`. Il sottoinsieme di 300 claim è fisso (ordinato per SHA-256 di `item_id`, così mescola Snopes e PolitiFact); il seed cambia solo quali claim ricevono la variante avvelenata. Il resto del setup è quello della sezione 2.

Tutti i risultati di questo esperimento vanno etichettati **"preliminare, corpus non ancora validato"**, come richiede il protocollo di allineamento (punto 6). Con soli 25 claim revisionati, una versione paper-facing (`--reviewed-only`) non ha ancora una dimensione utile.

### 4.4 Risultati (preliminari, corpus non ancora validato)

25 run, commit `cd95b48` (rieseguito dopo la revisione, risultati identici alla prima esecuzione), 300 claim per run, 1.500 record per tasso. Media ± deviazione standard sui cinque seed; IC 95% con bootstrap sui record.

| Tasso | PRR | IC 95% | PRD@1 | IC 95% |
|---|---|---|---|---|
| 0% | 0.000 ± 0.000 | [0.000, 0.000] | 0.000 ± 0.000 | [0.000, 0.000] |
| 1% | 0.050 ± 0.040 | [0.040, 0.061] | 0.001 ± 0.001 | [0.000, 0.002] |
| 3% | 0.100 ± 0.068 | [0.085, 0.115] | 0.001 ± 0.002 | [0.000, 0.003] |
| 5% | 0.148 ± 0.037 | [0.131, 0.167] | 0.002 ± 0.003 | [0.000, 0.005] |
| 10% | 0.261 ± 0.030 | [0.239, 0.283] | 0.006 ± 0.004 | [0.003, 0.010] |

Diagnostiche (codice corretto): Source Integrity 0.201–0.205, con lo stesso problema di definizione della sezione 3.2; ERS 0.000 (annotazioni assenti nel corpus, non inventate); FFR 0.000 (proxy).

### 4.5 Scomposizione del PRR

Il PRR conta qualunque documento avvelenato nel top-5. Sui `results.json` è stato separato il caso in cui il documento avvelenato è la variante del claim interrogato da quello in cui appartiene a un altro claim.

| Tasso | Documento vero del claim al rank 1 | Variante del claim nel top-5 | di cui al rank 2 | Solo varianti di altri claim |
|---|---|---|---|---|
| 1% | 0.999 | 0.015 | 21 su 22 | 0.035 |
| 5% | 0.998 | 0.050 | 72 su 75 | 0.098 |
| 10% | 0.994 | 0.096 | 135 su 144 | 0.165 |

Tre osservazioni.

1. **Quando un claim è avvelenato, la sua variante entra quasi sempre nel contesto.** La quota di query con la propria variante nel top-5 coincide con la quota attesa di claim avvelenati (circa 1%, 5%, 10%). Il retrieval non filtra la manipolazione.
2. **La variante resta quasi sempre al secondo posto, dietro l'originale.** Il documento vero è al rank 1 in oltre il 99% delle query. La causa è la query: il template `Is it true that <claim>?` contiene il testo originale, quindi l'originale è quasi identico alla query. Il PRD@1 vicino a zero misura questo vantaggio, non la robustezza del retrieval. La previsione iniziale (PRD@1 più alto che su Snopes) era sbagliata.
3. **Circa due terzi del PRR al 10% vengono da varianti di altri claim** (0.165 su 0.261). Sono documenti avvelenati di claim semanticamente vicini, che contaminano query su cui non c'è stato un attacco mirato. È un'esposizione collaterale, distinta da quella mirata.

### 4.6 Perché non è confrontabile con Snopes

- **Query**: generata da un LLM su Snopes, template con il testo originale qui (osservazione 2).
- **Quota di documenti avvelenati**: una variante per claim qui; quattro attacchi indipendenti su Snopes, circa quattro volte tanti documenti avvelenati allo stesso tasso nominale.
- **Dimensione**: 300 claim qui, 100 su Snopes.

Il risultato utilizzabile di questo esperimento non è il valore di PRR, ma la scomposizione: la manipolazione viene quasi sempre recuperata insieme alla verità, e il contesto dato al modello contiene quindi una contraddizione. È lo scenario in cui misurare FFR ha più senso.

## 4.7 Confronto dei retriever con query v2 (corpus ufficiale, curva completa)

Stesse 300 claim, stesse query v2, stessi documenti avvelenati e stessi seed. Il denso è sul commit della revisione (Mac-116), l'ibrido sul commit `4690d28` (Mac-116, 25 run completate il 13 novembre). Il BM25 è disponibile solo al 10% (Mac-Susy, commit `4690d28`).

PRR e PRD@1 (media su 5 seed; IC claims × seed tra parentesi):

| Tasso | Denso PRR | Ibrido PRR | Denso PRD@1 | Ibrido PRD@1 |
|---|---|---|---|---|
| 1% | 0.071 [0.034, 0.115] | 0.056 [0.029, 0.087] | 0.006 | 0.006 |
| 3% | 0.142 [0.071, 0.218] | 0.121 [0.061, 0.183] | 0.017 | 0.015 |
| 5% | 0.201 [0.149, 0.262] | 0.174 [0.125, 0.225] | 0.023 | 0.021 |
| 10% | 0.333 [0.300, 0.367] | 0.297 [0.261, 0.333] | 0.041 [0.026, 0.059] | 0.039 [0.025, 0.055] |

Esposizione mirata (variante propria nel top-5) e collaterale (solo varianti di altri claim), al 10%:

| Retriever | Mirata | Collaterale |
|---|---|---|
| Denso | 0.095 | 0.239 |
| BM25 | 0.087 | 0.225 |
| Ibrido | 0.094 | 0.203 |

BM25, curva completa (corpus ufficiale, query v2, Mac-Susy). PRR e PRD@1 medi su 5 seed; IC claims × seed tra parentesi. ATTENZIONE: le run risultano su due commit (`4690d28` e `7e5421d`); il numero va confermato prima di essere citato (sezione 4.10).

| Tasso | BM25 PRR | IC | BM25 PRD@1 | IC |
|---|---|---|---|---|
| 1% | 0.041 | [0.019, 0.068] | 0.007 | [0.001, 0.013] |
| 3% | 0.123 | [0.066, 0.185] | 0.014 | [0.005, 0.025] |
| 5% | 0.178 | [0.122, 0.236] | 0.023 | [0.013, 0.034] |
| 10% | 0.312 | [0.276, 0.348] | 0.043 | [0.027, 0.060] |

Confronto con il denso, PRR: 0.071, 0.142, 0.201, 0.333; ibrido: 0.056, 0.121, 0.174, 0.297. Il BM25 è nel mezzo a ogni tasso.

Lettura.
- L'ibrido abbassa il PRR a ogni tasso, ma la differenza è nella parte collaterale (0.203 contro 0.239 al 10%). La parte mirata è identica (0.094 contro 0.095): il claim attaccato viene recuperato allo stesso modo.
- Il PRD@1 è praticamente uguale (differenze dentro gli intervalli): con query neutre, il primo posto non dipende dal retriever.
- Il valore del PRD@1 è basso a tutti i tassi, e questo è coerente con quanto visto sopra: con query che non contengono il valore cambiato, il documento avvelenato non prende quasi mai il primo posto.
- Confronto con Snopes: lì il PRD@1 del denso era 0.046 al 10% con query a template; qui 0.041 con query neutre, quindi il valore non dipende dal tipo di query nei limiti di questa misura. Il risultato su Snopes per BM25 (0.002) era causato dal bug del pareggio, già corretto.

## 4.6bis Esperimenti da rifare (avviso, 13 novembre 2026)

Non vanno citati come risultati, anche se il comando li ha prodotti:
- **Snopes, BM25 (`poison_rate_sweep_benchmark_sparse`)**, commit `c437527`: PRD@1 azzerato dal bug del pareggio BM25 (sezione 2.5 di `docs/CODE_REVIEW.md`). PRR invariato in pratica, ma va rifatto con `4690d28` per ottenere anche il PRD@1 corretto. Risultato corretto al 10% (Mac-Susy, 15 novembre, 5 run, commit `7e5421d`, equivalente a `4690d28`): PRR 0.452 [0.384, 0.522], PRD@1 0.028 [0.006, 0.058], mirata 0.166, collaterale 0.286. Confronto al 10%: denso PRR 0.538 e PRD@1 0.046; ibrido PRR 0.434 e PRD@1 0.046; BM25 PRR 0.452 e PRD@1 0.028. Il PRD@1 di BM25 passa da 0.002 (con il bug) a 0.028: conferma che il valore vecchio era azzerato dal pareggio, non dal metodo. Il rilancio su Mac-Domenico è indipendente e serve come replica. Curva completa da fare.
- **Snopes, ibrido (`poison_rate_sweep_benchmark_hybrid`)**: rieseguito per intero il 15 novembre su Mac-116 con la correzione (25 run, commit `7e5421d` e `c0f20d3`; da confermare che `7e5421d` contenga `4690d28`, sezione 4.10). PRR 0.078 / 0.194 / 0.272 / 0.434 (1/3/5/10%), PRD@1 0.006 / 0.010 / 0.018 / 0.046, mirata 0.170 e collaterale 0.264 al 10%. Prima della correzione i valori erano identici ai tassi bassi e al 10% PRR 0.430: la correzione cambia il PRR di 0.004 e non il PRD@1, come previsto (nell'ibrido, a parità sul lato lessicale, decide il lato denso). Il vecchio sweep è in `poison_rate_sweep_benchmark_hybrid_v1_before_tiefix`.
- **Snopes, denso**: valido, rieseguito dopo la revisione (commit `270c9d6`).
- **Corpus ufficiale e corpus grande**: validi per il commit usato (denso su revisione, ibrido su `4690d28`).

Ambiente: dal 13 novembre Ollama è alla versione **0.40.16** su entrambi i Mac. Gli sweep eseguiti prima su Mac-116 usavano 0.40.13. La versione di Ollama non incide su PRR, PRD, Source Integrity ed ERS (dipendono solo dal retrieval), ma può cambiare il testo delle risposte: va tenuta uguale prima di misurare FFR o PCS.

## 4.8 Stato delle misure di sensibilità e difesa (4 novembre -> 13 novembre 2026)

Obiettivo: misurare quanto il sistema dipende da un contesto manipolato e se una difesa semplice lo riduce. Le manipolazioni restano quelle già nel corpus, revisionate dove possibile. L'unica aggiunta è la campagna B (sezione 4.13): varianti a regole che cambiano solo numeri e date, senza modelli linguistici.

| Misura | Cosa risponde | Stato | Cosa serve |
|---|---|---|---|
| 1. PCS (sensibilità della risposta al contesto avvelenato) | La risposta cambia quando si rimuovono i documenti avvelenati? | Implementata nel runner. Config pronti (15 novembre): `experiments/corpus_sweep_official_genq_pcs_preliminary/` (10%, 5 seed). Analisi per record: `scripts/analyze_answers.py`. Sezione 4.13. | Eseguire lo sweep. PCS è un proxy a similarità di embedding: si riporta come "PCS (embedding-similarity proxy)". |
| 2. Calibrazione di FFR e Source Integrity | Le metriche automatiche correlano con il giudizio umano? | Kit pronto (15 novembre): `scripts/calibration_kit.py` (campione stratificato, due fogli ciechi, punteggi automatici a parte, confronto con kappa). Nessuna annotazione raccolta. Sezione 4.13. | Due annotatori umani su 50 record. Chi annota è ancora da decidere. |
| 3. Audit di contraddizione (difesa) | Una coppia di documenti in contraddizione nel top-5 segnala la manipolazione? | Implementato (15 novembre): `scripts/contradiction_audit.py`. Non richiede generazioni. Sezione 4.13. | Eseguirlo sui risultati esistenti. Soglia calibrata sui record a 0% (5% di falsi allarmi) più soglia fissa 0.5. |

Decisioni aperte prima di implementare la misura 3:
- definizione di contraddizione: probabilità NLI di contraddizione sopra soglia (punto di partenza 0.5, da calibrare);
- trattamento delle coppie: si considerano solo documenti con lo stesso claim_id, e si escludono le coppie non varianti, per evitare un segnale banale;
- chi decide le soglie e i criteri di validazione: la collega e il gruppo.

Stato dei risultati al 13 novembre 2026: corpus ufficiale con query v2, retriever densi (curva completa, Mac-116); BM25 solo al 10% (Mac-Susy, commit 4690d28; curva completa da rilanciare); ibrido completo su tutti i tassi (25 run, Mac-116, commit 4690d28, completate il 13 novembre, tabella in 4.7). Lo sweep ibrido su Snopes resta da rifare con il commit 4690d28, perché quello precedente è pre-correzione.

## 4.9 Prospettiva di ricerca: misure sulla risposta e manipolazioni scritte a mano (piano, 13 novembre 2026)

Domanda di ricerca. I risultati finora riguardano il retrieval: con query neutre la variante avvelenata entra nel contesto quasi sempre e vince il primo posto circa nella metà dei casi in cui è recuperata. Resta aperto il punto centrale del progetto: **il modello adotta l'informazione manipolata nella risposta?** Le misure della sezione 4.8 (PCS, calibrazione di FFR e Source Integrity, audit di contraddizione) servono a rispondere.

Manipolazioni scritte a mano. Oltre alle varianti prodotte da Mistral, la ricercatrice scriverà a mano un piccolo insieme di varianti più sottili: plausibili, coerenti con il resto del claim, ma con un elemento falso. Ruolo del codice: solo valutarle con la stessa pipeline e le stesse misure, non generarle né migliorarle.

Condizioni (da rispettare prima di qualunque esecuzione):
- **Gate etico.** Ogni variante passa dallo stesso controllo del corpus: classe di sensibilità S0–S2 (S3 esclusa), nessuna trasformazione azionabile in ambito sanitario, elettorale o finanziario, decisione registrata nel registro decisioni etiche con data e responsabile. La revisione è della collega, non di chi scrive le varianti.
- **Conservazione.** Le varianti non vanno nel repository GitHub, che oggi è pubblico: si tengono in `data/handcrafted/` (cartella esclusa da git) o in un repository privato, e si trasferiscono tra i Mac a mano.
- **Pubblicazione.** Nel paper solo risultati aggregati ed esempi astratti o sanitizzati, mai i testi.
- **Valutazione indipendente.** Chi scrive le varianti non giudica le risposte: l'annotazione umana (misura 2) va fatta da un'altra persona, senza sapere quali varianti sono scritte a mano.

Disegno di valutazione proposto:
- formato: un CSV con lo stesso schema di `Modified_Claims` (`item_id`, `claim_text`, `claim_original_reference`, `manipulation_category`), collegato ai claim del corpus ufficiale tramite `item_id`;
- confronto appaiato: per gli stessi claim, variante Mistral contro variante manuale, stesse query v2, stesso retriever denso, tasso di avvelenamento 100% sui claim dell'insieme;
- misure: esposizione (variante nel top-5), rango rispetto al documento vero, PCS (la risposta cambia rimuovendo la variante?), adozione dell'elemento falso nella risposta (annotazione umana cieca);
- da implementare: un dataset ristretto ai claim con variante manuale e l'opzione dell'attacco `corpus_precomputed` per leggere le varianti da `data/handcrafted/`.

Limite da dichiarare: l'insieme manuale sarà piccolo, quindi i risultati sono un caso di studio, non una stima generale.

## 4.10 Verifica dei commit (chiusa il 15 novembre 2026)

Commit `7e5421d` è un commit di sola documentazione (`docs/EXPERIMENTS.md`, 74 righe aggiunte). Il codice è identico a `4690d28`. Quindi i risultati ottenuti con `4690d28` o `7e5421d` sono equivalenti per il retrieval e sono validi: BM25 sul corpus ufficiale (21 run a `4690d28`, 4 a `7e5421d`), BM25 su Snopes (10%, a `7e5421d`), ibrido su Snopes (a `7e5421d` e `c0f20d3`).

Il warning "different git commits" dello script di aggregazione è quindi innocuo in questi casi. Regola da mantenere: prima di aggregare, verificare che ogni commit abbia il codice corretto, con `git diff --stat 4690d28 <commit>`.

La curva BM25 sul corpus ufficiale è stata aggregata con due commit: `4690d28` (correzione del pareggio BM25) e `7e5421d`. Lo stesso `7e5421d` compare, insieme a `c0f20d3`, nel nuovo sweep ibrido su Snopes. Prima di citare i valori va verificato per ogni run quale commit ha prodotto il risultato, e che la differenza tra i due commit non tocchi il retrieval BM25 né il tie-break. Se `7e5421d` contiene solo documenti, i valori restano validi; altrimenti le run con `7e5421d` vanno rifatte con `4690d28`.

## 4.11 Testo delle box della slide (15 novembre 2026)

Testo da usare nella slide di sintesi. Ogni affermazione è legata ai numeri delle sezioni 3 e 4. Il valore 10% è nominale: la quota reale di documenti avvelenati è circa il 15%.

Riga sotto il titolo: "Question: does the model adopt the poisoned information? Retrieval is measured; answer-level measures are pending."

Box 1 (53.8%, PRR@5, Snopes): "At a 10% nominal poison rate (about 15% of documents actually poisoned), over half of queries retrieve at least one poisoned document in the top-5. Most of it is collateral (37.4%) rather than targeted (16.4%). Dense retrieval, template queries."

Box 2 (Multi-Seed & Corpus Validation): "5 seeds per rate; 25 runs per retriever, on 100 cleaned Snopes claims and 300 official-corpus claims (not yet human-validated). When a claim is poisoned, its variant almost always enters the top-5."

Box 3 (RAGAS Judge Calibration Next): "The cosine proxy gives FFR = 0 by construction, so FFR is not reported yet. The Ollama RAGAS judge is integrated; the Source Integrity definition is under review; human-annotation calibration is next."

Grafici: il grafico Snopes (barre impilate) è coerente con la sezione 3.1. Il grafico dei tre retriever (sezione 4.7, 10%) dice che cambiare retriever non protegge il claim attaccato; è consigliato come slide successiva o al posto del box 2.

## 4.12 Replica su claim indipendenti (corpus grande, claim non presenti nel corpus ufficiale)

236 claim verificati veri, in inglese, presenti solo nel corpus grande (Mistral, senza correzioni manuali), quindi disgiunti da tutti quelli usati negli altri sweep. Query v2, retriever denso, 5 seed, commit `f444e3d` (Mac-116). Il retriever denso non è toccato dal bug del pareggio. Quasi tutti i claim vengono da Snopes (circa 96%), quindi la replica è sul corpus, non sulla fonte.

| Tasso | PRR | IC (claims × seed) | PRD@1 | IC | Mirata | Collaterale |
|---|---|---|---|---|---|---|
| 1% | 0.063 | [0.010, 0.122] | 0.003 | [0.000, 0.012] | 0.011 | 0.052 |
| 3% | 0.146 | [0.068, 0.223] | 0.008 | [0.002, 0.019] | 0.030 | 0.116 |
| 5% | 0.225 | [0.137, 0.312] | 0.013 | [0.003, 0.025] | 0.050 | 0.175 |
| 10% | 0.375 | [0.297, 0.450] | 0.026 | [0.013, 0.042] | 0.096 | 0.279 |

Confronto con il corpus ufficiale, stesso retriever e stesse query, al 10%: PRR 0.333 [0.300, 0.367], PRD@1 0.041 [0.026, 0.059], mirata 0.095, collaterale 0.239.

Lettura.
- **Il risultato principale si replica**: l'esposizione mirata è identica (0.096 contro 0.095). Quando un claim è avvelenato la sua variante entra nel top-5 quasi sempre, anche su claim mai usati prima.
- Il PRR è un po' più alto (0.375 contro 0.333) per la parte collaterale, ma gli intervalli si sovrappongono. Una spiegazione possibile, non verificata: il corpus è più piccolo (236 documenti veri contro 300), quindi ogni top-5 copre una parte più grande del corpus.
- Il PRD@1 è più basso (0.026 contro 0.041), con intervalli che si sovrappongono in parte: la differenza non è solida. Possibili cause: diversa composizione per categoria, manipolazioni solo Mistral.
- La variabilità tra seed è alta (deviazione standard del PRR fino a 0.10): con 236 claim il risultato è meno preciso.

## 4.13 Campagna di novembre: misure sulla risposta e varianti a più elementi (piano, 15 novembre 2026)

Due campagne, preparate il 15 novembre. Nessun risultato ancora: questa sezione descrive cosa si esegue, dove e come si legge.

### Campagna A: misure sulla risposta e difesa

**A1. PCS.** Corpus ufficiale, query v2, retriever denso, 10%, 5 seed (`experiments/corpus_sweep_official_genq_pcs_preliminary/`). Per ogni record con un documento avvelenato nel top-5 il runner genera una seconda risposta senza quei documenti. Il retrieval è identico allo sweep denso già fatto (stesso seed e stesso corpus): il PRR al 10% deve tornare 0.333, ed è un controllo.

Analisi: `scripts/analyze_answers.py`. Il runner salva solo la media di PCS su tutti i record, che include gli zeri dei record senza veleno; lo script ricalcola PCS per record e lo riporta separato per esposizione mirata (variante del proprio claim nel top-5) e collaterale. Con `--nli` misura anche l'**adozione**: la risposta implica la variante (P(implicazione) > 0.5 e maggiore di quella del fatto vero)? Lo stesso per la risposta controfattuale; la differenza è l'effetto del veleno sulla risposta.

Limiti da dichiarare: PCS e adozione NLI sono automatiche e non ancora calibrate (A3); la risposta controfattuale ha un contesto più corto (si tolgono documenti senza sostituirli), quindi parte del cambiamento può dipendere dalla lunghezza.

**A2. Audit di contraddizione.** `scripts/contradiction_audit.py` sui risultati già salvati, senza Ollama. Per ogni record si confrontano tutte le coppie di documenti del top-5 con il modello NLI di Source Integrity; il punteggio della coppia è il massimo delle due direzioni (NLI non è simmetrico), quello del record il massimo sulle coppie. La soglia è calibrata sui record a 0%, dove non c'è veleno, per avere il 5% di falsi allarmi; si riporta anche la soglia fissa 0.5.

Lettura dei risultati:
- *Rilevabile*: quota dei record esposti in cui nel top-5 ci sono sia la variante sia il documento vero. È il massimo che un controllo a coppie può fare: se il vero non è recuperato, la variante non contraddice nulla.
- *Rilevato*: quota segnalata, tra i rilevabili e tra tutti gli esposti.
- *Coppia vero–variante*: probabilità di contraddizione della coppia dello stesso claim. Risponde a una domanda più semplice: il modello NLI vede un numero cambiato come una contraddizione?

Differenza rispetto al piano in 4.8: lì si proponeva di usare solo coppie con lo stesso `claim_id`. Un sistema reale non conosce i `claim_id`, quindi il rilevatore usa tutte le coppie (con la soglia calibrata per controllare i falsi allarmi). La coppia dello stesso claim è riportata a parte, come riferimento con informazione privilegiata. Quale dei due usare come risultato principale va deciso con la collega.

**A3. Calibrazione.** `scripts/calibration_kit.py`, in tre passi:
1. `sample`: 50 record dallo sweep PCS al 10% (30 con esposizione mirata, 10 collaterale, 10 puliti; un record per claim). Produce due fogli Excel identici nel contenuto ma in ordine diverso, uno per annotatore, con menu a tendina e un foglio di istruzioni. Nessun punteggio automatico nei fogli.
2. `score`: calcola a parte i punteggi automatici (NLI, proxy di embedding salvato nel record, e con `--ragas` il giudice RAGAS via Ollama).
3. `compare`: accordo tra annotatori (kappa di Cohen; kappa pesata quadratica per le scale 1–5), poi accordo di ogni misura automatica con i record su cui i due annotatori concordano.

Domande per l'annotatore: la risposta afferma il fatto vero? afferma il dettaglio cambiato nella variante? quanto è sostenuta dal contesto (1–5)? quanto è corretta rispetto al fatto vero (1–5)?

Cosa resta umano: compilare i due fogli. Gli annotatori lavorano separati, senza vedere i punteggi automatici, e non si usa un modello linguistico come secondo annotatore: lo scopo è proprio verificare i giudici automatici contro le persone. Con kappa tra annotatori sotto circa 0.6 va rivisto il codebook prima di giudicare le misure automatiche. Le soglie di FFR (fedeltà > 0.8, correttezza < 0.2) si fissano solo dopo questo confronto.

### Campagna B: varianti a più elementi

Domanda: il numero di elementi cambiati in una variante (k = 1, 2, 3) cambia quanto spesso la variante viene recuperata, se supera il vero nel ranking e se la risposta la adotta?

Perché servono varianti nuove: nel corpus ufficiale quasi ogni variante cambia un solo elemento (su 3.638 coppie, nessun numero nuovo nel 57%, un numero nel 43%, due in un caso solo). Con questi dati il confronto tra k non è possibile.

Generatore (`eiger/datasets/multi_element.py`, `scripts/make_multi_element_variants.py`): a regole, senza modelli linguistici. In ogni claim vero del corpus ufficiale trova gli elementi numerici (interi, decimali, anni; esclude identificativi come "COVID-19" o "F-35"), li ordina a caso con un seme fisso e cambia i primi k con una regola fissa per tipo: anno ±1; intero ±1 sotto 10, altrimenti ±10% arrotondato; decimale ±10% con le stesse cifre. Le entità sono quelle già presenti nel corpus (per esempio "2020 -> 2021", "9 -> 10").

Limiti di disegno, voluti:
- cambia solo numeri e date, non riscrive frasi;
- la regola è la stessa per ogni claim e non viene scelta guardando il risultato: non c'è nessuna ricerca della variante più credibile o meno rilevabile;
- il file generato va in `data/multi_element/variants.csv`, escluso da git, e non si pubblica. Non va nemmeno condiviso: ogni macchina lo rigenera dal corpus versionato e controlla l'impronta. Versione 1, seme 0: 1.182 righe, SHA-256 `7ae52ed33966fbcf5bfe1bcc228293eccfecbb6b86e43eafc9707ee6cb5b8c58`.

Disegno appaiato: le varianti sono annidate (la variante con k=2 contiene la modifica di k=1 più un'altra), e per uno stesso seed vengono avvelenati gli stessi claim per ogni k (la scelta dipende solo da seed, claim e tasso; verificato: per k=1 e k=2 al 10%, seed 42, gli stessi 25 claim su 218). Il confronto tra k è quindi sullo stesso claim.

Insiemi di claim (query v2, escluse le query che ripetono un numero cambiato):
- `multi2`: 218 claim con almeno 2 elementi, k = 1, 2. È il confronto principale.
- `multi3`: 66 claim con almeno 3 elementi, k = 1, 2, 3. Esplorativo, pochi claim.

Tassi: 10% (confrontabile con il resto) e 50% (più record esposti, per stimare meglio le misure condizionate all'esposizione). PCS attivo. Totale: 20 run per `multi2`, 30 per `multi3`, 5 seed ciascuno.

Analisi: `scripts/analyze_answers.py --sweep k1=... --sweep k2=...` riporta per ogni k esposizione, variante sopra il vero, variante al primo posto, PCS e adozione NLI, con le differenze appaiate rispetto a k=1 (bootstrap per claim).

Rapporto con le varianti scritte a mano (4.9): sono due cose diverse. Le varianti a regole misurano l'effetto del numero di elementi cambiati con un disegno controllato. Le varianti a mano restano un caso di studio separato, con le condizioni della sezione 4.9.

### Divisione del lavoro

| Mac | Prima (in corso) | Poi |
|---|---|---|
| Susy | sweep attuali | A1 PCS (5 run), poi B `multi3` (30 run); A2 audit quando Ollama è libero |
| Domenico | BM25 Snopes (25 run) | B `multi2` (20 run) |

A3 parte dopo A1: serve lo sweep PCS per il campione.

### Sezione 4.12

La sezione 4.12 (replica su claim indipendenti) è nel commit `7d14425` ed è identica alla versione preparata con i risultati della replica. Il Mac di Domenico non la vedeva perché il suo ramo si fermava al commit precedente (`7e5421d`); dopo l'allineamento della sezione 4.13 è presente su entrambi.

## 5. Limiti e interpretazione

FFR è zero per una ragione strutturale. FFR conta una risposta come falsità fedele solo se la faithfulness supera 0.8 e la correttezza scende sotto 0.2. Lo scorer proxy confronta similarità di embedding, e una risposta sullo stesso argomento del fatto vero ottiene comunque una correttezza alta anche se contiene il falso. Serve uno scorer che verifichi l'implicazione.

Lo scorer RAGAS (`faithfulness_scorer: ragas`) ora funziona: il bug su `AnswerSimilarity` è corretto e il test live passa. Non è però calibrato: usa `llama3.1:8b` come giudice, con problemi di affidabilità documentati. Va calibrato su annotazioni umane prima di produrre numeri da riportare, sempre come "FFR (RAGAS, llama3.1:8b judge)".

Source Integrity: fino al commit `b3450f9` era calcolata in modo errato (sezione 3.2 e `docs/CODE_REVIEW.md`). Dopo la correzione usa P(implicazione | documento, claim) del modello `cross-encoder/nli-MiniLM2-L6-H768`, che resta non calibrato e non fissato a una revisione.

FFR con lo scorer proxy: la correttezza è la similarità coseno riscalata in [0, 1] come (cos + 1) / 2. Due testi non correlati hanno coseno vicino a 0, quindi correttezza vicina a 0.5: la soglia 0.2 richiederebbe un coseno sotto −0.6, che con questi embedding non si verifica. FFR = 0 è quindi garantito dalla costruzione dello scorer, non osservato.

Warning noti, non bloccanti: `ascore` deprecato in RAGAS 0.2.15, `recreate_collection` deprecato in `qdrant_store.py`, `datetime.utcnow()` deprecato in `eiger/core/models.py`, `project.license` come tabella in `pyproject.toml`.

## 6. Storico (superati, da non citare)

- **29 ottobre.** 100 claim Snopes, un seed, sei attacchi inclusi M03 e M05. PRR da 0.28 (1%) a 0.77 (10%).
- **3 novembre.** Cinque seed, quattro attacchi. PRR da 0.20 (1%) a 0.82 (10%). Includeva come documenti avvelenati le copie identiche all'originale, e la Source Integrity era calcolata rispetto alla query.

Entrambe le versioni confermano il trend crescente, ma i valori sono sovrastimati.

## 7. Cosa serve ancora

1. **Definizione di Source Integrity** (da decidere con la collega). Proposta: SI = 1 − media di P(contraddizione | documento, claim). Con questa definizione un documento coerente o non correlato vale circa 1, uno che contraddice il claim circa 0, e la metrica scenderebbe solo se il contesto contiene affermazioni contrarie alla verità.
2. **Prompt delle query** (da decidere con la collega): metà delle query generate per il corpus ufficiale, e 17 su 100 per Snopes, ripetono il valore vero.
3. **FFR**: calibrare lo scorer RAGAS su annotazioni umane, e riportare FFR anche sui soli record esposti alla variante del proprio claim.
4. **Parità** sull'altra macchina con `10pct_seed42`.
5. **Revisione umana del corpus ufficiale**, poi sezione 4 con `--reviewed-only`.
6. Fissare la revisione dei modelli NLI e di embedding (`HF_HUB_OFFLINE=1`).
7. **Campagne A e B** (sezione 4.13): eseguire PCS, audit di contraddizione, varianti a più elementi; scegliere i due annotatori per la calibrazione; decidere con la collega quale versione dell'audit è il risultato principale.

## 8. Riproduzione

```bash
scripts/install_locked.sh && make up
python scripts/make_benchmark_sweep.py --seeds 42 43 44 45 46
caffeinate -i scripts/run_benchmark_sweep.sh experiments/poison_rate_sweep_benchmark
python scripts/aggregate_benchmark_sweep.py

python scripts/make_corpus_sweep.py --seeds 42 43 44 45 46 --max-claims 300
caffeinate -i scripts/run_benchmark_sweep.sh experiments/corpus_sweep_preliminary
python scripts/aggregate_benchmark_sweep.py --results-dir results/corpus_sweep_preliminary \
  --title "Official corpus sweep, PRELIMINARY (corpus not yet human-validated)"
```
