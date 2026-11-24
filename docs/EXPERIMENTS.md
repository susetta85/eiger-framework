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

Due campagne, preparate il 15 novembre ed eseguite il 16 sul Mac di Susy (A1, A2, `multi3`) e su quello di Domenico (`multi2`). I risultati sono nelle sottosezioni "Risultati"; sono preliminari, perché le misure sulla risposta non sono ancora calibrate (A3).

### Campagna A: misure sulla risposta e difesa

**A1. PCS.** Corpus ufficiale, query v2, retriever denso, 10%, 5 seed (`experiments/corpus_sweep_official_genq_pcs_preliminary/`). Per ogni record con un documento avvelenato nel top-5 il runner genera una seconda risposta senza quei documenti. Il retrieval è identico allo sweep denso già fatto (stesso seed e stesso corpus): il PRR al 10% deve tornare 0.333, ed è un controllo.

Analisi: `scripts/analyze_answers.py`. Il runner salva solo la media di PCS su tutti i record, che include gli zeri dei record senza veleno; lo script ricalcola PCS per record e lo riporta separato per esposizione mirata (variante del proprio claim nel top-5) e collaterale. Con `--nli` misura anche l'**adozione**: la risposta implica la variante (P(implicazione) > 0.5 e maggiore di quella del fatto vero)? Lo stesso per la risposta controfattuale; la differenza è l'effetto del veleno sulla risposta.

Limiti da dichiarare: PCS e adozione NLI sono automatiche e non ancora calibrate (A3); la risposta controfattuale ha un contesto più corto (si tolgono documenti senza sostituirli), quindi parte del cambiamento può dipendere dalla lunghezza.

#### Risultati A1 (16 novembre, Mac Susy, codice `7034901`, 5 run, 1.500 record)

Controllo superato: 142 record con esposizione mirata e 358 solo collaterali, 500 su 1.500, PRR 0,333 come nello sweep denso.

| Misura (10%) | Valore [IC 95%, bootstrap per claim] |
|---|---|
| Variante del proprio claim nel top-5 | 0,095 [0,079; 0,109] (n=1500) |
| Variante sopra il vero (dato esposizione) | 0,437 [0,345; 0,525] (n=142) |
| Variante al primo posto | 0,040 [0,029; 0,051] (n=1500) |
| PCS, esposizione mirata | 0,043 [0,034; 0,053] (n=142) |
| PCS, solo collaterale | 0,022 [0,015; 0,029] (n=358) |
| Adozione NLI, risposta reale | 0,317 [0,232; 0,411] (n=142) |
| Adozione NLI, risposta controfattuale | 0,211 [0,135; 0,293] (n=142) |
| P(risposta reale implica il vero) | 0,404 [0,321; 0,490] (n=142) |
| P(risposta controfattuale implica il vero) | 0,584 [0,492; 0,671] (n=142) |

Lettura, preliminare: (1) PCS è piccolo (0,043) e circa la metà nei collaterali, dove può pesare il contesto più corto del controfattuale. (2) Con il veleno la risposta implica il vero meno spesso, circa −0,18; gli intervalli in tabella sono marginali, l'intervallo sulla differenza appaiata si calcola con la versione di `analyze_answers.py` della sezione 4.14. (3) L'adozione NLI non è specifica: la risposta controfattuale, senza veleno, risulta "adottare" la variante nel 21% dei casi. Non va usata come risultato principale finché A3 non dice cosa misura.

**Differenze appaiate e baseline senza attacco (17 novembre, Mac Susy).** Riferimento: le run a 0% di `results/corpus_sweep_official_genq_preliminary/` (stessi claim, contesto completo). Differenze record per record sui 142 record con esposizione mirata, IC 95% bootstrap per claim:

| Differenza | Avvelenata − controfattuale | Avvelenata − senza attacco |
|---|---|---|
| P(la risposta implica il vero) | −0,180 [−0,263; −0,099] | −0,195 [−0,275; −0,110] |
| Adozione NLI della variante | +0,106 [0,023; 0,193] | +0,106 [0,014; 0,196] |
| P(la risposta implica la variante) | +0,028 [−0,057; 0,118] | +0,024 [−0,060; 0,114] |

| Controllo | Valore |
|---|---|
| 1 − sim(risposta, risposta senza attacco), record puliti | 0,002 [0,001; 0,004] (n=1000) |
| 1 − sim(risposta, risposta senza attacco), esposizione mirata | 0,042 [0,034; 0,049] |
| 1 − sim(risposta, risposta senza attacco), solo collaterale | 0,024 [0,017; 0,033] |
| Risposta senza attacco: P(implica il vero) / adozione | 0,598 / 0,211 |

Lettura, preliminare:
1. Controllo superato: sui record puliti la risposta è praticamente uguale a quella senza attacco (0,002), quindi la generazione è deterministica e le differenze sui record esposti non sono rumore.
2. Il contesto più corto del controfattuale non spiega i risultati: risposta controfattuale e risposta senza attacco danno quasi gli stessi valori (P(vero) 0,584 contro 0,598; adozione 0,211 in entrambi; PCS 0,043 contro 0,042). Il limite dichiarato in A1 si può ridimensionare. Anche l'effetto collaterale (0,024) resta.
3. Effetto del veleno: la probabilità che la risposta implichi il vero scende di circa 0,19, con intervallo lontano da zero.
4. La probabilità che la risposta implichi la variante non sale in modo misurabile (+0,02, intervallo che include zero). L'aumento dell'adozione (+0,11) viene quindi soprattutto dal calo dell'implicazione del vero, che fa scattare la seconda condizione della definizione, non da risposte che affermano la variante. Ipotesi da verificare con A3: con vero e variante nello stesso contesto, la risposta diventa meno netta, più che sbagliata.
5. L'adozione NLI resta poco specifica (21% anche senza veleno).

**A2. Audit di contraddizione.** `scripts/contradiction_audit.py` sui risultati già salvati, senza Ollama. Per ogni record si confrontano tutte le coppie di documenti del top-5 con il modello NLI di Source Integrity; il punteggio della coppia è il massimo delle due direzioni (NLI non è simmetrico), quello del record il massimo sulle coppie. La soglia è calibrata sui record a 0%, dove non c'è veleno, per avere il 5% di falsi allarmi; si riporta anche la soglia fissa 0.5.

Lettura dei risultati:
- *Rilevabile*: quota dei record esposti in cui nel top-5 ci sono sia la variante sia il documento vero. È il massimo che un controllo a coppie può fare: se il vero non è recuperato, la variante non contraddice nulla.
- *Rilevato*: quota segnalata, tra i rilevabili e tra tutti gli esposti.
- *Coppia vero–variante*: probabilità di contraddizione della coppia dello stesso claim. Risponde a una domanda più semplice: il modello NLI vede un numero cambiato come una contraddizione?

Differenza rispetto al piano in 4.8: lì si proponeva di usare solo coppie con lo stesso `claim_id`. Un sistema reale non conosce i `claim_id`, quindi il rilevatore usa tutte le coppie (con la soglia calibrata per controllare i falsi allarmi). La coppia dello stesso claim è riportata a parte, come riferimento con informazione privilegiata. Quale dei due usare come risultato principale va deciso con la collega.

**Primo controllo (16 novembre, 20 record per run, Mac Susy).** Il modello NLI vero risponde nel formato atteso. Con tutte le coppie, però, a 0% il 95% dei record puliti ha una coppia con P(contraddizione) > 0,5, e la soglia calibrata sale a 0,998: a quella soglia non si rileva nessuno dei 17 record rilevabili del campione. È un limite noto dei modelli NLI addestrati su SNLI/MNLI, che tendono a classificare come contraddizione due frasi su argomenti diversi; prendendo il massimo su 20 coppie orientate, quasi ogni record supera la soglia. La coppia vero–variante dello stesso claim ha in media P(contraddizione) tra 0,50 e 0,70: anche l'oracolo vede la modifica di un numero solo circa la metà delle volte. Numeri da campione piccolo, da confermare con l'audit completo.

**Filtro dei quasi duplicati** (`--min-overlap 0.4`). Una variante e il suo vero sono quasi duplicati: nel corpus ufficiale il 98,6% delle coppie vero–variante ha sovrapposizione di parole (Jaccard) ≥ 0,4, mentre due claim presi a caso non superano mai 0,33. Con il filtro si valutano solo le coppie quasi duplicate. Un sistema reale può calcolarlo, quindi resta un rilevatore realistico, senza bisogno dei `claim_id`. I file di output prendono il suffisso `_overlap0.4` e non sovrascrivono il rapporto su tutte le coppie. Si eseguono entrambe le versioni e si riportano tutte e due.

#### Risultati A2 (16 novembre, Mac Susy, 25 run, 7.500 record, corpus ufficiale, query v2, denso)

Tutte le coppie:

| Tasso | Rilevabili (vero e variante nel top-5) | Rilevati, soglia calibrata 0,998 | Coppia vero–variante: P(contr.) media | Coppia sopra 0,5 |
|---|---|---|---|---|
| 1% | 22/22 | 0/22 | 0,424 | 45% (10/22) |
| 3% | 49/49 | 0/49 | 0,424 | 45% (22/49) |
| 5% | 73/75 | 0/73 | 0,391 | 41% (30/73) |
| 10% | 139/142 | 2/139 (1,4%) | 0,374 | 38% (53/139) |

Falsi allarmi sui puliti: 5% alla soglia calibrata (per costruzione), 98,7% alla soglia fissa 0,5. Il retrieval porta quasi sempre nel top-5 sia la variante sia il vero; il limite è il modello NLI, che anche sulla coppia giusta vede il numero cambiato come contraddizione in meno della metà dei casi. Su tutte le coppie, come difesa, non funziona.

Solo coppie quasi duplicate (`--min-overlap 0.4`):

| Tasso | Falsi allarmi (puliti) | Rilevati tra gli esposti: qualunque coppia quasi duplicata | Rilevati: coppia quasi duplicata e P(contr.) > 0,5 | Collaterali segnalati |
|---|---|---|---|---|
| 0% | 1,0% (15/1500) | – | – | – |
| 1% | 1,1% | 100% (22/22) | 45% (10/22) | 64% |
| 3% | 1,1% | 100% (49/49) | 45% (22/49) | 62% |
| 5% | 1,2% | 96% (72/75) | 41% (31/75) | 65% |
| 10% | 1,4% | 98% (139/142) | 42% (60/142) | 74% |

La soglia calibrata risulta 0,000 perché il 99% dei record puliti non ha coppie quasi duplicate: il rilevatore segnala ogni record con una coppia quasi duplicata. A rilevare è il segnale strutturale, non il giudizio NLI. I collaterali segnalati non sono falsi allarmi (il loro top-5 contiene vero e variante di un altro claim). Limite: vale per varianti che sono modifiche minime del vero; varianti scritte diversamente avrebbero sovrapposizione più bassa (da verificare con le varianti scritte a mano, 4.9).

**A3. Calibrazione.** `scripts/calibration_kit.py`, in tre passi:
1. `sample`: 50 record dallo sweep PCS al 10% (30 con esposizione mirata, 10 collaterale, 10 puliti; un record per claim). Produce due fogli Excel identici nel contenuto ma in ordine diverso, uno per annotatore, con menu a tendina e un foglio di istruzioni. Nessun punteggio automatico nei fogli.
2. `score`: calcola a parte i punteggi automatici (NLI, proxy di embedding salvato nel record, e con `--ragas` il giudice RAGAS via Ollama).
3. `compare`: accordo tra annotatori (kappa di Cohen; kappa pesata quadratica per le scale 1–5), poi accordo di ogni misura automatica con i record su cui i due annotatori concordano.

Domande per l'annotatore: la risposta afferma il fatto vero? afferma il dettaglio cambiato nella variante? quanto è sostenuta dal contesto (1–5)? quanto è corretta rispetto al fatto vero (1–5)?

Cosa resta umano: compilare i due fogli. Gli annotatori lavorano separati, senza vedere i punteggi automatici, e non si usa un modello linguistico come secondo annotatore: lo scopo è proprio verificare i giudici automatici contro le persone. Con kappa tra annotatori sotto circa 0.6 va rivisto il codebook prima di giudicare le misure automatiche. Le soglie di FFR (fedeltà > 0.8, correttezza < 0.2) si fissano solo dopo questo confronto.

**Stato (17 novembre, Mac Susy).** `sample` e `score` fatti: 50 record (30/10/10, disponibili 142/358/1.000), seme 42, in `results/calibration/a3_batch01_*`. Annota una sola persona, la collega: la procedura per un'annotatrice sola è nella sezione 4.14. Domanda specifica emersa da A1: l'adozione NLI segnala anche risposte senza veleno; il confronto con le etichette deve dire se va scartata o ridefinita.

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

#### Risultati `multi3` (16 novembre, Mac Susy, 30 run, 66 claim)

k=1 e k=2 in un primo lancio, k=3 in un secondo (finito alle 15:29), stesso codice. Differenze appaiate rispetto a k=1, IC 95% bootstrap per claim:

| Misura | 50%: k2 − k1 | 50%: k3 − k1 | 10%: k3 − k1 |
|---|---|---|---|
| Variante nel top-5 | 0,000 | +0,003 [−0,009; 0,018] | 0,000 |
| Variante sopra il vero (dato esposizione) | −0,033 [−0,170; 0,099] | −0,100 [−0,245; 0,029] | 0,000 [−0,229; 0,200] (n=37) |
| Variante al primo posto | −0,006 [−0,064; 0,052] | −0,039 [−0,103; 0,021] | +0,003 [−0,018; 0,027] |
| PCS, esposizione mirata | +0,003 [−0,005; 0,010] | +0,002 [−0,006; 0,010] | +0,007 [−0,007; 0,021] |
| Adozione NLI | −0,086 [−0,200; 0,028] | −0,053 [−0,191; 0,079] | −0,027 [−0,167; 0,108] |
| P(risposta implica il vero) | **−0,138 [−0,216; −0,053]** | **−0,152 [−0,260; −0,046]** | **−0,178 [−0,334; −0,034]** |
| P(controfattuale implica il vero) | −0,010 [−0,035; 0,013] | −0,004 [−0,032; 0,023] | 0,000 |

Valori assoluti al 50%: P(risposta implica il vero) 0,394 (k=1), 0,256 (k=2), 0,244 (k=3).

Lettura, preliminare: il retrieval non dipende da k (le stesse varianti entrano nel top-5) e PCS è stabile. Con più elementi cambiati la risposta implica il vero meno spesso; la risposta controfattuale non cambia, quindi la differenza viene dalla risposta avvelenata. L'adozione NLI non cambia in modo significativo (e A1 mostra che non è specifica). Al 10% ci sono solo 37 record esposti. Misure non calibrate e 66 claim: il confronto principale è `multi2`.

#### `multi2` (16–17 novembre, Mac Domenico, codice `7034901`, 20 run, 218 claim)

Differenze appaiate k=2 − k=1, IC 95% bootstrap per claim:

| Misura | 10% | 50% |
|---|---|---|
| Variante nel top-5 | +0,001 [−0,003; 0,006] | 0,000 |
| Variante sopra il vero (dato esposizione) | −0,028 [−0,121; 0,068] (n=109) | −0,023 [−0,093; 0,049] (n=517) |
| Variante al primo posto | −0,003 [−0,012; 0,006] | −0,006 [−0,039; 0,027] |
| PCS, esposizione mirata | 0,000 [−0,007; 0,007] | +0,002 [−0,002; 0,006] |
| Adozione NLI della risposta | **+0,083 [0,009; 0,159]** | +0,004 [−0,052; 0,061] |
| P(la risposta implica il vero) | +0,029 [−0,052; 0,113] | 0,000 [−0,055; 0,055] |
| P(la risposta controfattuale implica il vero) | +0,018 [0,000; 0,046] | +0,008 [−0,006; 0,023] |

Lettura, preliminare:
1. Il retrieval non dipende da k (stesse varianti nel top-5), e PCS è stabile.
2. Il calo di "la risposta implica il vero" osservato in `multi3` non si replica: al 50%, con 517 record esposti, la differenza è 0,000. Il risultato di `multi3` (66 claim) è quindi esplorativo e non va citato come conferma.
3. L'unico segnale significativo è l'adozione NLI al 10%, con intervallo che sfiora lo zero e 109 record. Al 50% sparisce. Va trattato come ipotesi.

Controllo di rumore (Mac Domenico, corretto il 19 novembre): le risposte controfattuali di k=1 e k=2 sono identiche nel 93,7% dei casi (1.342 su 1.432). Il 17 novembre il restante 6% era stato attribuito a rumore di generazione, presumendo che il contesto controfattuale fosse identico per costruzione. Era sbagliato: la risposta controfattuale usa il top-5 meno *tutti* i documenti avvelenati, comprese le varianti degli altri claim, e tra k=1 e k=2 il numero di documenti avvelenati nel top-5 può cambiare, e con esso i documenti puliti che restano. Verifica con `scripts/check_counterfactual_context.py` sui due sweep: 1.277 coppie con stesso contesto e stessa risposta, **0 con stesso contesto e risposta diversa**, 65 con contesto diverso e stessa risposta, 90 con contesto diverso e risposta diversa. La generazione è quindi deterministica (lo confermano anche i controlli sui record puliti della seconda campagna, tutti 0,000), e le differenze appaiate non hanno un rumore di fondo da generazione. L'adozione al 10% (+0,083) non si spiega con il rumore; però è l'unico intervallo significativo tra molti confronti e non si ripete né al 50% né nelle altre famiglie (k2 − k1 al 10%: `numrand` +0,028 [−0,038; 0,096], `mix` −0,019 [−0,082; 0,044]). Va trattata come probabile falso positivo da confronti multipli.

Stato: risultato di `multi2` acquisito. Confronto con la baseline senza attacco di `multi2` da fare (le run a 0% sono sul Mac di Susy).

### Divisione del lavoro (16 novembre)

| Mac | Fatto |
|---|---|
| Susy | A2 (entrambe le versioni), `multi3` (30 run), A1 (5 run, spostato qui perché il Mac era libero), A3 `sample` e `score` |
| Domenico | BM25 Snopes (25 run, 15 novembre), `multi2` (20 run, da verificare) |

### Sezione 4.12

La sezione 4.12 (replica su claim indipendenti) è nel commit `7d14425` ed è identica alla versione preparata con i risultati della replica. Il Mac di Domenico non la vedeva perché il suo ramo si fermava al commit precedente (`7e5421d`); dopo l'allineamento della sezione 4.13 è presente su entrambi.

## 4.14 Seconda campagna: baseline, famiglie di varianti e altri retriever (piano, 17 novembre 2026)

Preparata il 17 novembre. Nessun risultato ancora. Quattro domande:

1. **Baseline.** I risultati delle varianti a regole sono diversi da quelli senza attacco e da quelli delle varianti del corpus ufficiale sugli stessi claim?
2. **Regola fissa o casuale.** Cambia qualcosa se l'ampiezza del cambiamento è estratta a caso invece che fissa?
3. **Tipo di elemento.** Un luogo, un mese, un ordine di grandezza o una direzione cambiati si comportano come un numero cambiato?
4. **Retriever.** Col denso k non cambia il retrieval; con BM25 (che confronta le parole esatte, numeri compresi) e con l'ibrido?

### Famiglie di varianti

Generatore esteso (`eiger/datasets/multi_element.py`, `scripts/make_multi_element_variants.py --all`). Quattro famiglie, un file ciascuna in `data/multi_element/` (escluso da git, non si pubblica, si rigenera su ogni macchina):

| Famiglia | Cosa cambia | Regola | Claim con ≥1 / ≥2 / ≥3 elementi | SHA-256 (seme 0) |
|---|---|---|---|---|
| `num` | numeri e anni | fissa (sezione 4.13) | 861 / 245 / 76 | `7ae52ed3…8c58` (invariata) |
| `numrand` | gli stessi elementi di `num`, nello stesso ordine | ampiezza casuale: anno ±1..10, intero ±1..5 sotto 10, altrimenti ±5..50%, decimale ±5..50% | 861 / 245 / 76 | `d38dc61f…4ab0` |
| `mix` | numeri, stati USA, paesi, mesi, ordini di grandezza (thousand/million/billion/trillion), direzione (increased/decreased, rose/fell, higher/lower, more than/less than, ...) | fissa: voce successiva della lista, mese successivo, ordine di grandezza successivo, contrario | 1.124 / 596 / 277 | `da4f17a7…1003` |
| `mixrand` | gli stessi elementi di `mix` | casuale: un'altra voce qualsiasi della stessa lista; numeri come `numrand`; la direzione ha un solo contrario | 1.124 / 596 / 277 | `8f2a3abe…38a2` |

Impronte complete: `num` `7ae52ed33966fbcf5bfe1bcc228293eccfecbb6b86e43eafc9707ee6cb5b8c58`, `numrand` `d38dc61fe7d7538f4f9002dab0b7a1b10b5f3d1d2265fe9ddda055aa44164ab0`, `mix` `da4f17a706c4bac2013eb29a7c2d7db467101021806cebe592355997a92b1003`, `mixrand` `8f2a3abe303efb335826bf2fa369bcbb4510df5b130aab10dcd8f145f64238a2`.

Disegno appaiato: `numrand` cambia gli stessi elementi di `num` e `mixrand` gli stessi di `mix` (verificato su tutte le righe), quindi fissa contro casuale si confronta sullo stesso claim e sullo stesso elemento. Le famiglie nuove scrivono il tipo di ogni cambio (`Texas -> Utah [us_state]`) e hanno versione 2. Il file `num` è identico a quello usato per `multi2` e `multi3`.

Tipi esclusi, di proposito:
- **nomi di persona e fonti**: cambiare chi ha detto o fatto qualcosa è lo scambio di attribuzione (M03), escluso dal benchmark dal protocollo etico (README, `allow_non_benchmark_attacks`). Non si introduce senza una decisione esplicita con la collega;
- parole ambigue: "Washington" (città, cognome), "Jordan", "Chad", "Georgia" come paese, "May" (anche verbo), "up"/"down", "rise"/"fall" (anche nomi);
- una parola di direzione il cui contrario è già nel claim (cambiarla fonderebbe due elementi).

Restano i limiti della sezione 4.13: nessun modello linguistico, nessuna ricerca della variante più credibile o meno rilevabile, regola uguale per ogni claim.

Filtro delle query: per `num` resta quello concordato (numeri e anni ripetuti). Per le famiglie nuove si escludono anche le query che ripetono una parola cambiata (per esempio il nome dello stato vero), per la stessa ragione: favoriscono il documento vero. Insiemi risultanti: `multi2_numrand` 218 claim (gli stessi di `multi2`), `multi2_mix` e `multi2_mixrand` 394 claim (gli stessi), di cui 300 nello sweep (`max_claims`).

### Baseline

- **Nessun attacco**: una run a 0% per ogni insieme di claim e retriever (`*_noattack_*`). Un seme basta: a 0% il seme non cambia nulla (verificato sullo sweep BM25 Snopes: impronte di retrieval identiche tra i seed a 0%). È la risposta di riferimento con il contesto completo, senza il problema del contesto più corto del controfattuale.
- **Varianti del corpus ufficiale** sugli stessi claim (`*_corpusvar_*`): ogni claim di `multi2` e `multi3` ha la sua variante nel corpus (M01 202 e M02 16 in `multi2`; M01 63 e M02 3 in `multi3`), che cambia un solo elemento. È il riferimento per k=1. Una sola query di `multi2` ripete il numero cambiato dalla variante del corpus; resta nell'insieme per non rompere l'abbinamento.
- **Altri retriever**: BM25 e ibrido sui claim di `multi2` (k=1, k=2, varianti del corpus, nessun attacco). Per `multi3` e per le famiglie nuove solo denso, per contenere i tempi.

Per il corpus ufficiale (A1) la baseline senza attacco esiste già: le run a 0% di `results/corpus_sweep_official_genq_preliminary/`.

### Analisi

`scripts/analyze_answers.py` aggiunge:
- `--noattack DIR`: per ogni record, la risposta della run a 0% sullo stesso claim. Misure: NLI della risposta senza attacco, e 1 − similarità tra risposta e risposta senza attacco. Sui record puliti questa quantità deve essere circa 0 (stesso top-5, generazione a temperatura 0): è un controllo di determinismo.
- **differenze appaiate dentro lo stesso sweep**, record per record, con intervallo sulla differenza: avvelenata − controfattuale e avvelenata − senza attacco, per P(implica il vero), P(implica la variante) e adozione. Risponde al punto lasciato aperto in A1.
- `--by-kind`: misure sui record esposti divise per tipo di elemento cambiato (us_state, month, integer, ...; M01/M02 per le varianti del corpus).

Differenze appaiate per A2 e A3: in A2 le due versioni dell'audit sono rilevatori deterministici sugli stessi record e la differenza è già netta (0–1% contro 96–100%), quindi un test appaiato non aggiunge informazione. In A3 il confronto è già appaiato per costruzione (stessi item).

### A3 con una sola annotatrice

Annota la collega. Un annotatore automatico non può sostituire la persona: A3 serve a controllare le misure automatiche, e confrontarle con un'altra misura automatica sarebbe circolare. `scripts/calibration_kit.py` ora ha:
- `compare` senza `--b`: misure automatiche contro le etichette dell'annotatrice;
- `retest --n 15`: un secondo foglio con 15 degli stessi item in ordine nuovo, da compilare almeno una settimana dopo senza guardare il primo; `compare --b <retest> --retest` dà l'accordo tra le due compilazioni.

Limite da dichiarare: un'annotatrice con retest misura la coerenza, non se un'altra persona etichetterebbe allo stesso modo.

### Esecuzione

145 configurazioni nuove, tutte con PCS tranne le run senza attacco. Prima di lanciare, su ogni Mac: `python scripts/make_multi_element_variants.py --all` e confronto delle quattro impronte con la tabella sopra.

| Blocco | Cartelle (`experiments/`) | Run | Tempo stimato |
|---|---|---|---|
| 1. Nessun attacco, denso | `corpus_sweep_{multi2,multi3,multi2_mix}_noattack_genq_preliminary` | 3 | ~0,5 h |
| 2. Varianti del corpus, denso | `corpus_sweep_{multi3,multi2}_corpusvar_genq_pcs_preliminary` | 20 | ~3,5 h |
| 3. Regola casuale, numeri | `corpus_sweep_multi2_numrand_k{1,2}_genq_pcs_preliminary` | 20 | ~5,5 h |
| 4. Tipi di elemento, fissa | `corpus_sweep_multi2_mix_k{1,2}_genq_pcs_preliminary` | 20 | ~7,5 h |
| 5. Tipi di elemento, casuale | `corpus_sweep_multi2_mixrand_k{1,2}_genq_pcs_preliminary` | 20 | ~7,5 h |
| 6. BM25 | `corpus_sweep_multi2_{noattack,corpusvar,k1,k2}_genq_sparse*_preliminary` | 31 | ~8 h |
| 7. Ibrido | `corpus_sweep_multi2_{noattack,corpusvar,k1,k2}_genq_hybrid*_preliminary` | 31 | ~8 h |

Tempi stimati dalle run del 16 novembre (circa 16 minuti per run su 218 claim con PCS, 5 su 66, circa 22 su 300). Ordine proposto: blocchi 1–5 sul Mac di Susy (circa 24 ore), blocchi 6–7 sul Mac di Domenico (circa 16 ore), dopo la verifica di `multi2`.

**Stato (19 novembre).** Blocchi 1–5 completati sul Mac di Susy il 19 novembre alle 12:43 (83 run). Blocchi 6–7 lanciati il 19 novembre pomeriggio, divisi tra i due Mac: ibrido sul Mac di Susy (31 run), BM25 sul Mac di Domenico (31 run). Il codice del Mac di Domenico è stato allineato a `origin/main` (`0f2defb`) e le quattro impronte delle varianti coincidono.

**Incidente Qdrant e correzione (17–19 novembre).** Durante i blocchi 2–5 alcune run sono fallite all'inizio, nel reset della collezione, con `RocksDB open error: IO error: While open a file for lock` (poi `While open directory`). Gli errori sono diventati via via più frequenti (nella notte del 19 novembre due run su quattro), e il riavvio del container non bastava più. Causa probabile (non verificata sul log di Qdrant, ma coerente con quanto osservato): ogni run crea la propria collezione e non la cancella; sul Mac di Susy se ne erano accumulate centinaia, che Qdrant tiene aperte e ricarica a ogni avvio, fino a esaurire i file che il container può aprire. Dopo la cancellazione delle collezioni `eibench_` (le analisi leggono solo `results/`, e ogni run ricrea la sua collezione da zero) e con la cancellazione a fine run, le ultime 26 run sono andate senza errori e senza nessun nuovo tentativo. Le run fallite sono state rifatte da capo: i risultati non ne risentono. Strumenti aggiunti:
- `scripts/run_pending.sh CARTELLA...`: esegue solo le configurazioni senza `results.json`, cancella la collezione dopo ogni run riuscita, e se una run fallisce riavvia Qdrant e la riprova una volta. Da usare per tutte le campagne lunghe, compresi i blocchi 6–7.
- `scripts/qdrant_drop_collections.sh`: cancella le collezioni `eibench_` tra una campagna e l'altra (rifiuta se c'è una run attiva).
- `docker-compose.yml`: il controllo di salute usava `curl`, che nell'immagine di Qdrant non c'è, per cui il container risultava sempre "unhealthy". Ora apre la porta con bash; vale dopo `make down && make up`.

Confronti previsti (tutti appaiati per claim):

| Domanda | Confronto |
|---|---|
| Baseline | `multi2` k1 contro `corpusvar` e contro `noattack`; lo stesso per `multi3` |
| Fissa o casuale | `num` k1/k2 (cioè `multi2`) contro `numrand` k1/k2; `mix` contro `mixrand` |
| Tipo di elemento | `mix` k1 con `--by-kind` |
| Retriever | `multi2` k1, k2, `corpusvar` con denso, BM25, ibrido |

### Risultati dei blocchi 1–5 (19 novembre, preliminari)

Analisi con `scripts/analyze_answers.py --nli` (e `--noattack`, `--by-kind`), file `results/analisi_*.md` sul Mac di Susy. Tutte le misure di risposta sono automatiche (NLI, embedding) e non ancora calibrate con A3.

**1. Effetto del veleno: solido in tutte le famiglie.** Record con la propria variante nel top-5, differenza record per record tra risposta avvelenata e risposta senza attacco:

| Famiglia (tasso) | n | P(la risposta implica il vero) | Adozione NLI |
|---|---|---|---|
| A1, corpus ufficiale, tutti i claim (10%) | 142 | −0,195 [−0,275; −0,110] | +0,106 [0,014; 0,196] |
| `multi2`, varianti del corpus (50%) | 518 | −0,221 [−0,290; −0,155] | +0,158 [0,109; 0,209] |
| `multi2` `num` k1 (50%) | 517 | −0,255 [−0,326; −0,183] | +0,147 [0,102; 0,197] |
| `multi2` `numrand` k1 (50%) | 518 | −0,263 [−0,337; −0,192] | +0,114 [0,075; 0,156] |
| `mix` k1 (50%) | 732 | −0,158 [−0,210; −0,105] | +0,149 [0,107; 0,194] |
| `mixrand` k1 (50%) | 730 | −0,187 [−0,241; −0,136] | +0,125 [0,084; 0,167] |
| `multi3` k1 (50%) | 151 | −0,152 [−0,257; −0,042] | +0,185 [0,090; 0,287] |
| `multi3`, varianti del corpus (50%) | 149 | −0,162 [−0,277; −0,036] | +0,215 [0,121; 0,320] |

Al 10% i valori sono dello stesso ordine. La risposta controfattuale dà quasi gli stessi valori della risposta senza attacco in tutte le famiglie.

Specificità dell'adozione NLI: la risposta senza attacco "adotta" la variante nel 21% dei casi in A1, ma tra 0 e 2% su `multi2`, `multi3` e `mix`. Sui claim con un elemento cambiato (numeri, date, luoghi) la misura è specifica; la scarsa specificità di A1 riguarda le varianti del corpus ufficiale sull'insieme completo dei claim. Da capire con A3.

**2. Regola fissa o casuale: nessuna differenza robusta.** Differenze appaiate al 50%: `numrand` − `num`, adozione −0,035 [−0,086; 0,012], P(vero) −0,009 [−0,053; 0,032], variante sopra il vero +0,019 [−0,063; 0,105] (PCS +0,005 [0,002; 0,009], trascurabile); `mixrand` − `mix` (k1), adozione −0,021 [−0,067; 0,026], P(vero) −0,028 [−0,063; 0,007]. Unico segnale: a k=2 e 10% la variante casuale supera il vero meno spesso (−0,101 [−0,192; −0,012]), ma al 50% la differenza non è significativa (−0,037 [−0,104; 0,032]).

**3. Numero di elementi cambiati: nessun effetto robusto.** k2 − k1 al 50%: `num` adozione +0,004, P(vero) 0,000; `numrand` +0,004, +0,003; `mix` −0,033 [−0,082; 0,014], −0,035 [−0,082; 0,012]. Il retrieval non dipende da k. Il calo di P(vero) in `multi3` (66 claim) resta isolato.

**4. Varianti del corpus ufficiale contro varianti a regole.** Sulla risposta l'effetto è simile. Nel retrieval le varianti ufficiali sembrano piazzarsi più in alto: su `multi3` (appaiato, 50%) variante sopra il vero +0,181 [−0,013; 0,387], variante al primo posto +0,088 [0,000; 0,176], al limite della significatività. Su `multi2` (appaiato, 218 claim, 50%) la direzione è la stessa ma più debole e non significativa: variante sopra il vero +0,075 [−0,022; 0,178], variante al primo posto +0,042 [−0,005; 0,089]; sulla risposta adozione +0,021 [−0,050; 0,090], P(vero) +0,034 [−0,022; 0,088]. Lettura: le varianti ufficiali (quasi tutte M01) potrebbero essere un po' più competitive nel retrieval di quelle a regole, ma nessuno dei due insiemi lo conferma; sulla risposta le due famiglie si equivalgono. Il confronto `multi2` mette insieme run dei due Mac (k1 su quello di Domenico, varianti del corpus su quello di Susy), che per la pipeline densa danno risposte identiche (punto 6).

**5. Tipo di elemento: esplorativo.** Pochi record per tipo, intervalli larghi. Al 50% in `mix` k1 l'adozione è più alta per interi (0,19), mesi (0,21) e anni (0,18), più bassa per ordini di grandezza (0,03) e stati USA (0,03).

**6. Determinismo e parità tra i due Mac.** Sui record puliti, 1 − similarità tra risposta e risposta senza attacco è 0,000 [0,000; 0,000] in tutti gli sweep. Lo stesso vale tra le run `multi2` del Mac di Domenico e la baseline senza attacco del Mac di Susy (0,000 su 660 record al 10%): per la pipeline densa i due Mac danno risposte identiche.

### Blocchi 6–7 (BM25 e ibrido) e revisione del 20 novembre

**Stato.** 62 run complete: BM25 sul Mac di Domenico (codice `0f2defb`), ibrido sul Mac di Susy (codice `75b53b0`; tra i due commit cambia solo la documentazione).

**Controlli di integrità** (tutte le 62 run): 218 claim per run, sempre lo stesso insieme; top-5 sempre pieno; nessuna risposta vuota; risposta controfattuale presente esattamente quando c'è veleno nel top-5; nessun documento avvelenato nelle run a 0%. Le configurazioni BM25 e ibride differiscono da quelle dense solo per il tipo di retriever (e per nome, collezione e cartella). L'insieme avvelenato dipende solo da seme, tasso e claim (`CorpusBuilder`, `derive_seed`), non dal retriever né da k: il confronto appaiato tra retriever è valido.

**Retrieval, `multi2`, 50%** (1.090 record per sweep):

| Varianti | Retriever | Variante nel top-5 | Variante sopra il vero | Variante al primo posto |
|---|---|---|---|---|
| `num` k1 | denso | 0,474 | 0,489 | 0,220 |
| `num` k1 | BM25 | 0,455 | 0,478 | 0,200 |
| `num` k1 | ibrido | 0,479 | 0,494 | 0,226 |
| corpus ufficiale | denso | 0,475 | 0,566 | 0,262 |
| corpus ufficiale | BM25 | 0,455 | 0,448 | 0,187 |
| corpus ufficiale | ibrido | 0,483 | 0,570 | 0,262 |

Ibrido − BM25, appaiato: variante nel top-5 +0,024 [0,010; 0,039] (`num` k1) e +0,028 [0,013; 0,044] (corpus ufficiale); variante al primo posto +0,026 [−0,023; 0,075] e +0,075 [0,023; 0,129]. k2 − k1: nessun effetto né con BM25 (variante sopra il vero +0,004 [0,000; 0,012]) né con l'ibrido (−0,021 [−0,088; 0,049]). Confronto appaiato con il denso: sotto.

**Punto emerso dalla revisione: con queste query nessun retriever separa il vero dalla variante.** In nessuno dei record con la propria variante nel top-5 la query contiene il valore cambiato (l'insieme `multi2` esclude per costruzione i claim la cui query lo ripete). Conseguenze:
- BM25: il vero e la variante a regole hanno esattamente lo stesso punteggio nel 100% dei casi `num` k1 (circa 90% con le varianti ufficiali, che a volte cambiano anche altre parole). Chi sta sopra lo decide la regola di spareggio (hash del `doc_id`, introdotta per non favorire sempre il vero). Per BM25 "variante sopra il vero" e "variante al primo posto" valgono circa 0,5 per costruzione e non misurano una proprietà del retriever; per lo stesso motivo k non può cambiare il retrieval BM25.
- Ibrido: pari merito nel 45–50% dei casi, risolti a favore del documento che compare prima nella lista densa.
- Denso: nessun pari merito esatto, ma "variante sopra il vero" resta vicino a 0,5 (0,489).

Lettura: quando la query non nomina il valore cambiato, vero e variante entrano insieme nel top-5 e il conflitto passa al generatore. Le misure informative sono l'esposizione (variante nel top-5) e gli effetti sulla risposta; "variante sopra il vero" e PRD@1 vanno riportati con questa avvertenza, e per BM25 non vanno interpretati.

**Nota di disegno sull'ibrido.** La fusione RRF usa solo i primi 5 documenti di ciascun retriever (non liste più lunghe) e risolve i pari merito a favore del denso. Va bene per questa campagna, ma per una versione futura conviene fondere liste più lunghe (per esempio 50) e usare lo stesso spareggio del BM25. Cambierebbe i risultati, quindi solo in una nuova campagna e con un nuovo nome di retriever.

**Effetto sulla risposta con BM25 e ibrido** (analisi del 20 novembre sul Mac di Susy, `results/analisi_bm25.md`, `analisi_ibrido.md`, `analisi_retriever_k1.md`, `analisi_retriever_corpusvar.md`). Record con la propria variante nel top-5, avvelenata − senza attacco (stesso retriever), 50%:

| Retriever | Varianti | n | P(la risposta implica il vero) | Adozione NLI |
|---|---|---|---|---|
| denso | `num` k1 | 517 | −0,255 [−0,326; −0,183] | +0,147 [0,102; 0,197] |
| BM25 | `num` k1 | 496 | −0,227 [−0,292; −0,164] | +0,135 [0,090; 0,184] |
| ibrido | `num` k1 | 522 | −0,213 [−0,280; −0,145] | +0,119 [0,082; 0,159] |
| denso | corpus ufficiale | 518 | −0,221 [−0,290; −0,155] | +0,158 [0,109; 0,209] |
| BM25 | corpus ufficiale | 496 | −0,201 [−0,266; −0,141] | +0,137 [0,088; 0,189] |
| ibrido | corpus ufficiale | 526 | −0,208 [−0,266; −0,152] | +0,146 [0,094; 0,200] |

Differenze appaiate tra retriever (stessi claim, semi e documenti avvelenati), 50%:
- `num` k1, BM25 − denso: variante nel top-5 −0,019 [−0,037; −0,002]; adozione −0,012 [−0,071; 0,043]; P(vero) +0,036 [−0,006; 0,080]. Ibrido − denso: top-5 +0,005 [−0,005; 0,015]; adozione −0,025 [−0,060; 0,008]; P(vero) +0,029 [−0,005; 0,062].
- Corpus ufficiale, BM25 − denso: top-5 −0,020 [−0,038; −0,002]; variante al primo posto −0,075 [−0,130; −0,022]; adozione −0,019 [−0,079; 0,036]; P(vero) +0,032 [−0,009; 0,078]. Ibrido − denso: top-5 +0,007 [0,001; 0,017]; nessuna differenza sulla risposta.
- Al 10% l'unica differenza significativa è ibrido − denso su P(vero) con `num` k1 (+0,062 [0,009; 0,116], 109 record), che non si ripete al 50%.
- k2 − k1 e corpus ufficiale − `num` k1, dentro BM25 e dentro l'ibrido: nessuna differenza significativa sulla risposta.

Lettura, preliminare: il retriever cambia di poco quanto spesso la variante entra nel contesto (BM25 circa 2 punti in meno del denso) e dove la mette (con BM25 le varianti ufficiali finiscono meno spesso al primo posto), ma non cambia l'effetto sulla risposta una volta che la variante è nel contesto. La risposta alla domanda 4 è quindi: il risultato principale non dipende dal retriever. Nell'ibrido l'ordine tra vero e variante coincide con quello del denso (differenza 0,000), coerente con la nota di disegno sopra.

**Parità tra i Mac: completa.** Run a 0% ripetute sull'altro Mac e confrontate con `scripts/compare_runs.py`: BM25 (Mac di Susy contro Mac di Domenico) 218 su 218 con lo stesso retrieval e la stessa risposta; ibrido (Mac di Domenico contro Mac di Susy) 218 su 218. Con il denso (sezione 4.14, risultati punto 6), i due Mac danno risultati identici con tutti e tre i retriever.

**Effetto dell'avvelenamento sul retrieval di claim non toccati (BM25 e ibrido).** Con il denso, sui record puliti la risposta coincide con quella senza attacco (0,000). Con BM25 no (0,007), e non è rumore: i punteggi BM25 dipendono dalle statistiche dell'intero corpus (frequenza delle parole nei documenti, lunghezza media), e aggiungere i documenti avvelenati le cambia. Sui record puliti di `num` k1, 237 su 807 hanno con BM25 un top-5 diverso da quello della run senza attacco; con l'ibrido 107 su 815; con il denso nessuno. In tutti i casi con lo stesso top-5 anche la risposta è uguale, quindi la generazione resta deterministica. Conseguenze: (1) con i retriever lessicali l'avvelenamento di una parte del corpus sposta il retrieval anche per domande non toccate, un effetto collaterale da riportare; (2) con BM25 e ibrido la run senza attacco è un riferimento un po' meno pulito che col denso, e per questo con BM25 avvelenata − controfattuale (−0,156) e avvelenata − senza attacco (−0,227) differiscono più che col denso.

## 4.15 Terza campagna: generatori, profondità del contesto, dose–risposta (20–24 novembre 2026)

Nessuno di questi esperimenti dipende da A3 o dalle decisioni aperte con la collega (Source Integrity, prompt delle query, M03): usano solo la misura primaria, che non è l'adozione NLI.

**Piano statistico, dichiarato prima delle run.**
- Misura primaria: P(la risposta implica il vero), avvelenata − senza attacco, record con la propria variante nel top-5, differenza record per record con IC 95% bootstrap per claim (`scripts/analyze_answers.py --nli --noattack`).
- Tutto il resto (adozione NLI, PCS, retrieval, tipo di elemento) è esplorativo e non si cita come risultato senza replica.
- Ipotesi: H1 (E1) il calo di P(vero) è presente con tutti i generatori, cioè l'intervallo esclude lo zero per ciascuno; le differenze tra generatori sono esplorative. H2 (E4) con top-10 il calo è minore che con top-5, con top-3 maggiore. H3 (E5) il calo per record esposto non dipende dal tasso; cresce invece la quota di record esposti, e con BM25 la quota di claim non toccati il cui top-5 cambia.

**Insieme di claim e confronti.** Sempre `multi2` (218 claim), varianti `num` k1 e varianti del corpus ufficiale, query `genq`, 5 semi. Stessi claim, semi e documenti avvelenati in tutte le run: i confronti tra generatori, tra top-k e tra tassi sono appaiati.

| Esperimento | Cosa cambia | Cartelle | Run |
|---|---|---|---|
| E1, altri generatori | `qwen2.5:7b` e `mistral:7b` al posto di `llama3.1:8b` (denso, 50%) | `corpus_sweep_multi2_{noattack,k1,corpusvar}_genq_{qwen2-5-7b,mistral-7b}_…` | 22 |
| E4, profondità del contesto | top-3 e top-10 al posto di top-5 (denso, `num` k1, 50%) | `corpus_sweep_multi2_{noattack,k1}_genq_{top3,top10}_…` | 12 |
| E5, dose–risposta | tassi 1, 5 e 25% (10 e 50 ci sono già), denso e BM25 | nelle cartelle esistenti `corpus_sweep_multi2_k1_genq_pcs_preliminary` e `…_sparse_pcs_preliminary` | 30 |

Configurazioni generate con le nuove opzioni `--llm-model` e `--top-k` di `scripts/make_corpus_sweep.py`; con i valori predefiniti i nomi delle cartelle restano quelli di prima (verificato: nessuna configurazione esistente cambia). Per E1 i modelli vanno scaricati prima (`ollama pull qwen2.5:7b`, `ollama pull mistral:7b`). Tutte le run nuove registrano versione di Ollama e impronta del modello (`environment` in `results.json`).

### Esecuzione e controlli

64 run (E1 ed E4 sul Mac di Susy il 21 novembre; E5 sul Mac di Domenico il 21 e ripetuto sul Mac di Susy il 22), codice `246b12d`, Ollama 0.40.16, impronte dei modelli registrate (llama3.1:8b `46e0c10c…`, qwen2.5:7b `845dbda0…`, mistral:7b `6577803a…`). Tutte con 218 claim, il numero giusto di documenti nel contesto, nessuna risposta vuota, controfattuale presente esattamente quando c'è veleno nel contesto. Con i tre generatori il retrieval è identico: il confronto tra generatori è appaiato. Analisi con `scripts/analyze_answers.py --nli` (`results/analisi_e1_*`, `analisi_e4_*`, `analisi_e5_*`).

### Risultati (preliminari; misure di risposta non ancora calibrate con A3)

**E1 — H1 confermata.** Avvelenata − senza attacco (stesso generatore), record con la propria variante nel top-5, 50%:

| Generatore | Varianti | n | P(la risposta implica il vero) | Adozione NLI | P(vero) senza attacco |
|---|---|---|---|---|---|
| llama3.1:8b | `num` k1 | 517 | −0,255 [−0,326; −0,183] | +0,147 | 0,506 |
| qwen2.5:7b | `num` k1 | 517 | −0,129 [−0,195; −0,062] | +0,191 | 0,508 |
| mistral:7b | `num` k1 | 517 | −0,087 [−0,146; −0,024] | +0,116 | 0,357 |
| llama3.1:8b | corpus ufficiale | 518 | −0,221 [−0,290; −0,155] | +0,158 | 0,507 |
| qwen2.5:7b | corpus ufficiale | 518 | −0,132 [−0,190; −0,070] | +0,172 | 0,507 |
| mistral:7b | corpus ufficiale | 518 | −0,078 [−0,144; −0,017] | +0,129 | 0,358 |

Il calo c'è con tutti e tre i generatori e con entrambi i tipi di varianti. L'entità cambia (esplorativo): llama perde circa metà del P(vero) di partenza, qwen e mistral circa un quarto. Qwen adotta la variante più spesso di llama ma perde meno il vero (P(la risposta implica la variante), avvelenata − senza attacco, +0,106 [0,042; 0,170]): le sue risposte sembrano riportare entrambe le versioni. Mistral risponde in modo meno netto anche senza veleno. Confronto appaiato sulle risposte avvelenate: qwen − llama P(vero) +0,128 [0,059; 0,189], mistral − llama +0,018 [−0,048; 0,084].

**E4 — H2 confermata solo in parte.** Avvelenata − senza attacco (stesso top-k), `num` k1, 50%:

| Contesto | n | P(la risposta implica il vero) | Adozione NLI |
|---|---|---|---|
| top-3 | 510 | −0,257 [−0,329; −0,183] | +0,118 [0,075; 0,172] |
| top-5 | 517 | −0,255 [−0,326; −0,183] | +0,147 [0,102; 0,197] |
| top-10 | 527 | −0,180 [−0,248; −0,109] | +0,167 [0,120; 0,217] |

Con top-10 il calo è minore (risposte avvelenate, top-10 − top-5: P(vero) +0,103 [0,056; 0,155], contro +0,025 senza attacco): più contesto attenua l'effetto. Con top-3 il calo è uguale a top-5: la seconda metà di H2 non si conferma.

**E5 — H3 confermata.** Avvelenata − senza attacco, record con la propria variante nel top-5:

| Tasso | Denso: n | Denso: P(vero) | BM25: n | BM25: P(vero) | Variante propria nel top-5 (denso / BM25) | BM25: claim non toccati con top-5 cambiato |
|---|---|---|---|---|---|---|
| 1% | 17 | −0,209 [−0,392; −0,052] | 16 | −0,097 [−0,298; 0,079] | 0,016 / 0,015 | 9,1% |
| 5% | 49 | −0,325 [−0,467; −0,197] | 47 | −0,268 [−0,408; −0,126] | 0,045 / 0,043 | 18,6% |
| 10% | 110 | −0,261 [−0,358; −0,168] | 106 | −0,240 [−0,337; −0,146] | 0,101 / 0,097 | 27,7% |
| 25% | 260 | −0,234 [−0,321; −0,151] | 250 | −0,216 [−0,295; −0,138] | 0,239 / 0,229 | 40,6% |
| 50% | 517 | −0,255 [−0,326; −0,183] | 496 | −0,227 [−0,292; −0,164] | 0,474 / 0,455 | 49,2% |

Il calo per record esposto non ha un andamento col tasso (gli intervalli si sovrappongono; all'1% ci sono solo 16–17 record esposti). Cresce invece, quasi in proporzione al tasso, la quota di record esposti, e quindi l'effetto complessivo sul sistema (circa esposizione × calo: 0,003 all'1%, 0,12 al 50% col denso). Con BM25 l'avvelenamento cambia il top-5 anche dei claim non toccati, già dal 9% all'1%; col denso mai.

### Rumore di fondo della generazione tra run diverse

Nelle run E5 all'1 e al 5%, 20 record puliti (4 col denso, 16 con BM25, su circa 4.200 con lo stesso top-5 della baseline) hanno domanda e contesto identici a quelli della baseline senza attacco ma una risposta leggermente diversa; nelle run al 10 e al 50% (campagna 2) nessuno. Verifiche: le baseline densa e BM25 rifatte il 22 novembre sono identiche a quelle del 19–20 (218 su 218); le 30 run E5 rifatte sul Mac di Susy danno le stesse risposte di quelle del Mac di Domenico (6.539 su 6.540, stesso retrieval in tutti i 6.540 record); le differenze non dipendono dal fatto che il record precedente fosse avvelenato. Quindi: la generazione è riproducibile per una data run su qualsiasi Mac, ma la stessa domanda con lo stesso contesto può dare una risposta leggermente diversa in run diverse. Causa probabile: Ollama riusa i calcoli della richiesta precedente (cache del prompt) e il risultato numerico dipende da quale richiesta c'era prima. Il rumore di fondo tra run è dello 0,3% dei record puliti col denso e dell'1,4% con BM25: piccolo rispetto agli effetti misurati, da riportare come limite. Per le prossime campagne: aggiornamento automatico di Ollama disattivato, versione registrata (fatto dal 20 novembre).

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

In ordine di priorità; la roadmap completa con gli esperimenti proposti è nel documento di progetto `EIGER_roadmap_novembre_2026`.

1. **Seconda campagna: chiusa il 20 novembre** (sezione 4.14). Tutti i blocchi completati e analizzati; parità tra i Mac verificata per denso, BM25 e ibrido.
   **Terza campagna: chiusa il 24 novembre** (sezione 4.15): H1 confermata, H2 in parte, H3 confermata; rumore di fondo della generazione tra run misurato.
2. **A3, calibrazione umana** (sezione 4.13): annotazione della collega, poi `compare` e `retest`. È il prerequisito per qualunque affermazione sull'adozione NLI: in A1 la misura "adotta" la variante nel 21% delle risposte senza veleno.
3. **Decisioni con la collega**: definizione di Source Integrity (proposta: SI = 1 − media di P(contraddizione | documento, claim)); prompt delle query (metà delle query generate per il corpus ufficiale, e 17 su 100 per Snopes, ripetono il valore vero; l'insieme `multi2` le esclude per costruzione); versione principale dell'audit A2; cambi di nomi di persona (M03): Susy è favorevole (20 novembre); prima di introdurli serve il passaggio dalla gate etica concordata con la collega (protocollo di allineamento, punto 2) e una voce nel registro delle decisioni etiche.
4. **Piano statistico** prima della prossima campagna: un endpoint primario dichiarato in anticipo (proposta: P(la risposta implica il vero), avvelenata − senza attacco, record con la propria variante nel top-5, 50%), tutto il resto esplorativo; correzione per confronti multipli o, in alternativa, replica obbligatoria su un secondo insieme di claim prima di citare un effetto.
5. **Riproducibilità**: dal 20 novembre `results.json` registra in `environment` anche la versione del server Ollama (`ollama_version`), l'impronta del modello effettivamente servito (`llm_model_digest`, che cambia se `ollama pull` aggiorna il tag), le versioni di numpy, rank-bm25, qdrant-client, sentence-transformers, transformers, torch e httpx (`pkg_*`) e il valore di `HF_HUB_OFFLINE`. Resta da decidere se fissare `HF_HUB_OFFLINE=1` nelle campagne (oggi viene solo registrato). Le run precedenti non hanno questi campi: per loro valgono il commit e la parità verificata tra i due Mac.
6. **FFR**: calibrare lo scorer RAGAS su annotazioni umane, e riportare FFR anche sui soli record esposti alla variante del proprio claim.
7. **Revisione umana del corpus ufficiale**, poi sezione 4 con `--reviewed-only`.
8. **Parità BM25 Snopes** tra i due Mac: ora si può fare con `scripts/compare_runs.py` sulle due run `10pct_seed42`.

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
