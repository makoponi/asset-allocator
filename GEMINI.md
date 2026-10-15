# GEMINI.md — Specifiche Architetturali e Direttive per Agenti AI (Transatlantic Edition)

## 1. Contesto e Obiettivo del Progetto
Costruzione di un sistema quantitativo end-to-end per la gestione di un portafoglio azionario **Long-Only Transatlantico** (**USA + Canada + Europa**, *ex-Financials*), progettato per massimizzare il *Tail Risk Alpha* e minimizzare i Drawdown severi tramite:
1. Selezione cross-sectional non lineare (**Machine Learning** su fondamentali Point-in-Time neutralizzati per Regione e Settore).
2. Depurazione spettrale della matrice di covarianza multi-fuso orario (**Random Matrix Theory / Marcenko-Pastur** su rendimenti multi-day sincronizzati).
3. Allocazione topologica e orientata alla coda (**Hierarchical Risk Parity / Mean-CVaR**).
4. Copertura dinamica bipartita (*Dual-Sleeve Hedging*) tramite Future su Indici (`ES`/`MES` per il Nord America e `FXXP`/`FESX` per l'Europa) guidata da sensori di regime indipendenti (**Expanding-Window HMM** e **Turbolenza di Mahalanobis**).

---

## 2. Vincoli Tecnologici e Protocollo di Esecuzione Agente (MANDATORY)

### 2.1 Toolchain Obbligatorio
Qualsiasi agente AI (es. Antigravity) che generi o modifichi codice in questo repository DEVE rispettare i seguenti vincoli. Codice non conforme verrà rigettato:
* **Python Version:** `>= 3.12` (utilizzare esclusivamente la sintassi moderna per i tipi: `type`, `list[str]`, `dict[str, float]`, `X | None`, mai `typing.List`, `typing.Dict` o `typing.Optional`).
* **Package & Environment Manager:** Esclusivamente **`uv`**.
  * Esecuzione script/test: `uv run pytest`, `uv run mypy --strict src tests`, `uv run ruff check --fix .`
  * Aggiunta pacchetti: `uv add <package>` (o `uv add --dev <package>`).
  * Vietato l'uso diretto di `pip`, `poetry` o `conda`.
* **Linter & Formatter:** **`ruff`**. Lunghezza linea max: 100 caratteri. Regole attive: `E`, `F`, `B`, `I`, `UP`, `SIM`, `TCH`.
* **Static Type Checking:** **`mypy` in modalità `--strict`**. Nessun uso di `Any` non documentato. Tutti gli input/output tabellari devono avere schemi validati o type hint espliciti (`pd.DataFrame`, `np.ndarray[Any, np.dtype[np.float64]]`).
* **Testing Framework:** **`pytest`**. I test non devono MAI interrogare il terminale Bloomberg reale (usare fixture e mock deterministici in `tests/conftest.py`).
* **Version Control:** **`git`** con *Conventional Commits* (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).

### 2.2 Protocollo Operativo per Agenti a Ragionamento Medio (Atomic Tasking)
Per garantire la massima precisione matematica ed evitare allucinazioni strutturali, l'agente deve seguire tassativamente il ciclo **TDD Atomico**:
1. **Un Modulo alla Volta:** Non generare mai più di un modulo applicativo e il relativo file di test in un singolo turno operativo.
2. **Test-First Verification:** Prima o contestualmente all'implementazione di un algoritmo matematico (es. Marcenko-Pastur, HRP, Purged CV), scrivere un test unitario con matrici giocattolo a risultato analitico noto (es. verifica che la traccia della matrice di correlazione prima e dopo il denoising sia identica a $\text{tol} = 10^{-8}$).
3. **Gate di Validazione Obbligatorio:** Nessun task può essere dichiarato concluso dall'agente se i seguenti tre comandi, eseguiti in sequenza nel terminale, non restituiscono `Exit Code 0`:
   ```bash
   uv run ruff check --fix src tests && uv run ruff format src tests
   uv run mypy --strict src tests
   uv run pytest -v
   ```

---

## 3. Regole Auree di Ingegneria Finanziaria Transatlantica (Anti-Bias)
Gli agenti devono implementare controlli assertivi (`assert` o eccezioni custom) nel codice per prevenire i seguenti 6 errori metodologici:

1. **Valuta Base Unificata (EUR Base) e Deduplicazione Cross-Listing:**
   * L'universo copre titoli quotati in EUR, USD, CAD, GBP, CHF, SEK, DKK, NOK. Tutti i prezzi, i volumi e gli indici Total Return (`TOT_RETURN_INDEX_GROSS_DVDS`) devono essere convertiti in **EUR** alla data di riferimento.
   * Eventuali società con doppia quotazione tra indici (es. quotate sia a Toronto/New York che a Londra/Francoforte) devono essere deduplicate tramite identificativo globale (`FIGI` / `ISIN`), mantenendo la linea di quotazione più liquida.
2. **Sincronizzazione dei Fusi Orari per la Covarianza (3-Day / Weekly Returns):**
   * Le borse europee chiudono alle 17:30 CET, quelle nordamericane alle 22:00 CET. Calcolare la matrice di covarianza su rendimenti giornalieri *Close-to-Close* sottostima gravemente la correlazione tra USA ed Europa.
   * Il modulo di rischio (Marcenko-Pastur e HRP) deve operare obbligatoriamente su **rendimenti a 3 giorni lavorativi non sovrapposti (o rendimenti settimanali Friday-to-Friday)** in EUR.
3. **Neutralizzazione Doppia `(Region_Bloc, GICS_Sector)`:**
   * I principi contabili (US GAAP vs IFRS) e i multipli strutturali differiscono tra Nord America ed Europa. La standardizzazione cross-sectional (Z-Score) delle feature NON deve mai avvenire sull'intero universo globale insieme, ma rigorosamente all'interno dei gruppi `(region_bloc, gics_sector)` alla data $t$, dove `region_bloc` $\in \{\text{'NA'}, \text{'EU'}\}$.
4. **Esclusione del Settore Finanziario (GICS 40):**
   * Banche, assicurazioni e financial services presentano bilanci incompatibili con multipli industriali (`EV/EBITDA`, `Gross Margin`, `Accruals`). Il settore GICS `40` deve essere filtrato a monte dall'universo investibile in entrambe le regioni.
5. **Zero Look-Ahead Bias nei Fondamentali e nei Sensori HMM:**
   * *Fondamentali:* Nello split temporale al tempo $t$, sono interrogabili solo i record con `filing_date <= t`.
   * *HMM:* È severamente vietato fare `.fit()` dell'Hidden Markov Model sull'intera serie storica. L'addestramento deve avvenire in **Expanding Window** (ricalibrazione mensile usando solo dati fino a $t$) e la probabilità di stato in $t$ deve essere calcolata tramite *forward filtering* puro, mai tramite algoritmo di Viterbi sull'intero campione.
6. **Ponte Bloomberg Ibrido Realistico:**
   * L'API Python `blpapi` desktop non espone nativamente BQL fuori dall'ambiente BQuant. L'ingestione massiva storica BQL viene gestita tramite loader dedicati da export BQL/Excel (`data/raw_bql_exports/*.parquet` o `.csv`) verso SQLite, mentre gli aggiornamenti delta usano `xbbg` (`bdp`/`bdh`) con caching idempotente rigoroso.
   * Nelle interrogazioni storiche `bdh`, assicurare sempre l'append del suffisso `Equity` (es. `f"{t} Equity"`) in input e il recupero pulito in output per prevenire errori `BAD_SEC` del DAPI.
   * Tutti i download massivi di serie giornaliere devono eseguire il checkpointing immediato batch-by-batch su Parquet con supporto a `--skip-existing`.

---

## 4. Struttura della Directory del Progetto
```text
quant-portfolio/
├── GEMINI.md
├── pyproject.toml
├── uv.lock
├── data/
│   ├── raw_bql_exports/      # Drop-zone per estrazioni storiche BQL (SPX, SPTSX60, SXXP)
│   ├── db/
│   │   └── quant_store.db    # Database SQLite (universo PiT, fondamentali PiT, serie macro)
│   └── parquet/              # Serie giornaliere Total Return in EUR partizionate per anno
├── src/
│   └── quant_portfolio/
│       ├── __init__.py
│       ├── config.py         # Configurazione tipizzata (Pydantic), pesi regionali e costanti
│       ├── storage/          # Layer SQLite e Parquet idempotente con deduplicazione ISIN
│       ├── ingestion/        # Generatore query BQL, parser export e client delta xbbg
│       ├── features/         # Feature tecniche e Z-score per (region_bloc, gics_sector)
│       ├── alpha/            # Modelli ML (LightGBM / RF) e Purged Walk-Forward CV
│       ├── risk/             # Resampling multi-day, Marcenko-Pastur, HRP e Mean-CVaR
│       ├── regime/           # Dual Expanding HMM (Sensori NA + Sensori EU) e Turbolenza
│       ├── hedging/          # Dual-Sleeve Beta-Adjusted Hedge Ratio (ES/MES + FXXP/FESX)
│       └── backtest/         # Motore Walk-Forward multi-valuta, slippage e reportistica CLI
└── tests/
    ├── conftest.py           # Fixture sintetiche transatlantiche prive di chiamate esterne
    └── test_*.py
```

---

## 5. Piano Operativo a Fasi (Step-by-Step)

### Fase 1: Setup Infrastruttura, Storage Locale e Schemi Dati Transatlantici
* **Obiettivo:** Inizializzare il progetto con `uv`, configurare `ruff`, `mypy --strict`, `pytest` e costruire il layer di persistenza locale idempotente.
* **Deliverables:**
  1. `pyproject.toml` completo con dipendenze (`polars`/`pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `cvxpy`, `hmmlearn`, `pydantic`).
  2. Modulo `src/quant_portfolio/storage/schema.py` con gestione SQLite per 3 tabelle core:
     * `universe_pit`: `(as_of_date TEXT, ticker TEXT, isin TEXT, region_bloc TEXT CHECK(region_bloc IN ('NA', 'EU')), source_index TEXT, gics_sector INT, local_currency TEXT, is_financial BOOL, PRIMARY KEY (as_of_date, ticker))`
     * `fundamentals_pit`: `(as_of_date TEXT, filing_date TEXT, ticker TEXT, metric TEXT, value REAL, PRIMARY KEY (as_of_date, ticker, metric))`
     * `macro_series`: `(date TEXT, region_bloc TEXT, series_id TEXT, value REAL, PRIMARY KEY (date, region_bloc, series_id))`
  3. Gestore Parquet per `daily_market_eur`: `(date, ticker, region_bloc, px_last_eur, tot_return_idx_eur, volume_eur)`.
  4. Unit test per verificare l'idempotenza degli `UPSERT`, il vincolo `filing_date <= as_of_date` e la deduplicazione per `isin`.

### Fase 2: Ricostruzione Universo Point-in-Time (USA + Canada + Europa) e Serie in EUR
* **Obiettivo:** Popolare i costituenti storici mensili degli ultimi 15 anni per i 3 indici di riferimento (inclusi i titoli delistati/acquisiti), escludere il settore GICS 40 (`Financials`) e consolidare le serie Total Return in EUR.
* **Perimetro dell'Universo:**
  * **Blocco `NA` (Nord America):** **S&P 500** (`SPX Index`) + **S&P/TSX 60** (`SPTSX60 Index`).
  * **Blocco `EU` (Europa):** **STOXX Europe 600** (`SXXP Index`).
  * Dimensione attesa universo attivo netto (ex-GICS 40): $\approx 900$ azioni totali ($\approx 450$ `NA`, $\approx 450$ `EU`).
* **Deliverables:**
  1. Modulo `src/quant_portfolio/ingestion/bql_templates.py` che genera le stringhe BQL esatte (con parametro `currency='EUR'` e `dates=range(-15y, 0d, frq=m)`) per esportare i costituenti storici e i fondamentali da terminale/Excel.
  2. Parser d'ingestione per caricare gli export BQL e client incrementale `xbbg` (a blocchi di 50 ticker con caching locale) per aggiornare `TOT_RETURN_INDEX_GROSS_DVDS` e `PX_VOLUME` in EUR.
  3. Allineamento calendari: gestione delle festività asincrone tra NYSE, TSX, LSE ed Euronext tramite calendario master e forward-fill controllato (max 3 giorni consecutivi).

### Fase 3: Ingestione Parsimoniosa dei Fondamentali e Neutralizzazione Doppia
* **Obiettivo:** Costruire la matrice dei predittori $X_{i,t}$ combinando 10 feature fondamentali Point-in-Time e 5 feature tecniche/di rischio calcolate in locale, eliminando il bias contabile USA vs Europa.
* **Set di 15 Feature Core:**
  * *Fondamentali (10 - da BQL PiT):* `PE_RATIO`, `PX_TO_BOOK_RATIO`, `EV_TO_T12M_EBITDA`, `FREE_CASH_FLOW_YIELD`, `RETURN_COM_EQY` (ROE), `OPER_MARGIN`, `ASSET_TURNOVER`, `TOT_DEBT_TO_TOT_EQY`, `SALES_REV_TURN_GROWTH_1YR`, `ACCRUALS_RATIO`.
  * *Prezzo/Rischio (5 - calcolate in locale su serie EUR):* `MOM_1M` (Reversal a 21gg), `MOM_12M_1M` (Rendimento da $t-252$ a $t-21$), `IDIO_VOL_252D` (Volatilità residua CAPM contro il rispettivo indice regionale `SPX` o `SXXP`), `BETA_REGIONAL_252D`, `AMIHUD_ILLIQUIDITY_21D`.
* **Trasformazione Cross-Sectional:**
  1. Winsorization all'1% e 99% all'interno di ciascun `region_bloc` alla data $t$.
  2. Standardizzazione in **Z-Score per `(as_of_date, region_bloc, gics_sector)`**:
     $$z_{i,t} = \frac{x_{i,t} - \mu_{\text{region, sector}, t}}{\sigma_{\text{region, sector}, t}}$$
  3. Imputazione di eventuali `NaN` residui a `0.0` (media neutrale del settore nella specifica regione).

### Fase 4: Motore di Machine Learning e Selezione Bilanciata per Sleeve
* **Obiettivo:** Addestrare il modello non lineare per ordinare i titoli e selezionare i candidati per il portafoglio evitando che una singola regione cannibalizzi l'intero capitale.
* **Deliverables:**
  1. Target $y_{i,t}$: Rendimento forward a 21 giorni lavorativi in eccesso rispetto alla mediana cross-sectional del proprio blocco regionale `(region_bloc)` in $t+1$.
  2. Implementazione di `PurgedWalkForwardCV` con *embargo period* di 5 giorni lavorativi.
  3. Modello Ensemble (**LightGBM Regressor** / **Random Forest** con iperparametri anti-overfitting: `max_depth=4`, `min_child_samples=30`, regolarizzazione L1/L2).
  4. **Selezione Stratified Top-Decile:** Estrazione dei migliori **30–35 titoli del blocco `NA`** e dei migliori **20–25 titoli del blocco `EU`** (totale: **50–60 azioni**), garantendo una diversificazione strutturale transatlantica prima dell'ottimizzazione dei pesi.

### Fase 5: Resampling Sincronizzato, Denoising di Marcenko-Pastur e Allocazione (HRP / CVaR) [COMPLETED]
* **Obiettivo:** Stimare la vera struttura di correlazione transatlantica sui 50-60 titoli selezionati ed eseguire l'allocazione ottimale del capitale.
* **Deliverables Realizzati:**
  1. **Sincronizzatore Multi-Day (`MultiDaySynchronizer`):** Trasforma i rendimenti giornalieri in EUR degli ultimi 2 anni ($T_{\text{daily}} = 504$) in **rendimenti a 3 giorni lavorativi non sovrapposti** ($T_{3d} \approx 168$) per annullare l'effetto del fuso orario (17:30 CET vs 22:00 CET).
  2. **Modulo `MarcenkoPasturDenoiser`:**
     * Calcolo della matrice di correlazione empirica $\mathbf{C}$ sui rendimenti sincronizzati.
     * Calcolo della soglia del rumore $\lambda_+ = \sigma^2 \left(1 + \sqrt{\frac{N}{T}}\right)^2$.
     * Sostituzione degli autovalori $\lambda_i \le \lambda_+$ con la loro media costante $\bar{\lambda}_{\text{noise}}$ (preservando analiticamente la traccia $\text{Tr}(\tilde{\mathbf{C}}) = N$) e riscalatura diagonale per garantire $C_{ii} = 1.0$. Ricostruzione della matrice di covarianza depurata $\tilde{\mathbf{\Sigma}} = \mathbf{S} \tilde{\mathbf{C}} \mathbf{S}$.
  3. **Modulo `HierarchicalRiskParity` (HRP):**
     * Distanza di correlazione $d_{ij} = \sqrt{0.5(1 - \rho_{ij})}$, Ward linkage, quasi-diagonalizzazione e recursive bisection.
     * Proiezione convessa quadratica (QP) post-bisezione per garantire rigorosamente i vincoli:
       * Singolo titolo: $0.005 \le w_i \le 0.035$ (max 3.5%, min 0.5%).
       * Pesi regionali: Sleeve `NA` $\in [50\%, 65\%]$; Sleeve `EU` $\in [35\%, 50\%]$.
       * Budget: $\sum w_i = 1.0$.
  4. **Modulo `MeanCVaROptimizer` ($\beta = 0.95$):**
     * Formulazione convessa Rockafellar-Uryasev su `cvxpy` con solver `CLARABEL` e rispetto identico dei vincoli di titolo e sleeve.
  5. **Pipeline e CLI (`allocate`):**
     * Modulo `PortfolioAllocationPipeline` e comando `quant-portfolio allocate` per calcolo allocazione, volatilità annualizzata 3d, CVaR ed MCTR.
     * Test suite estesa a 89 test unitari e di integrazione passati (100%).

### Fase 6: Doppio Sensore di Regime (NA + EU) e Dual-Sleeve Beta-Adjusted Hedging
* **Obiettivo:** Proteggere indipendentemente la componente Nord Americana e la componente Europea del portafoglio tramite i rispettivi contratti future su indici.
* **Input Macro Regionali (da FRED / Bloomberg, scaricati con caching):**
  * *Blocco `NA`:* `VIX Index`, US High Yield OAS (`BAMLH0A0HYM2`), Curva US Treasury 10Y-2Y (`T10Y2Y`), Turbolenza di Mahalanobis sui settori S&P 500.
  * *Blocco `EU`:* `V2TX Index` (Euro Stoxx 50 Volatility), EUR High Yield OAS (`BAMLHE00EHYIOAS`), Curva Bund 10Y-2Y, Turbolenza di Mahalanobis sui settori STOXX 600.
* **Deliverables:**
  1. Modulo `DualExpandingRegimeDetector`:
     * Due istanze Gaussian HMM a 3 stati addestrate in finestra espansiva (una per `NA`, una per `EU`) con ordinamento automatico degli stati per volatilità crescente (anti-*label switching*).
  2. Calcolo del **Dual-Sleeve Beta-Adjusted Hedge Ratio**:
     * Calcolo separato del valore di mercato e del beta ex-ante per le due maniche del portafoglio:
       $$\beta_{\text{Sleeve NA}, t} = \sum_{i \in \text{NA}} w_{i,t}^{\text{rel}} \beta_{i, \text{SPX}}, \quad \beta_{\text{Sleeve EU}, t} = \sum_{j \in \text{EU}} w_{j,t}^{\text{rel}} \beta_{j, \text{SXXP}}$$
     * Strumenti di copertura: Future **E-mini / Micro E-mini S&P 500 (`ES` / `MES`)** convertito al tasso di cambio EUR/USD corrente per la Sleeve `NA`; Future **STOXX Europe 600 (`FXXP`) o Euro Stoxx 50 (`FESX`)** per la Sleeve `EU`.
     * Regola di attivazione indipendente per ciascuna regione con **Banda di Inerzia a 3 giorni** (conferma di 3 chiusure consecutive sopra soglia 70% per aprire, sotto 40% per chiudere):
       * *Stato 0 (Calmo):* Copertura regionale = $0\%$.
       * *Stato 1 (Transizione/Allerta):* Copertura regionale = $50\% \times \beta_{\text{Sleeve}}$.
       * *Stato 2 (Stress Severo):* Copertura regionale = $100\% \times \beta_{\text{Sleeve}}$.

### Fase 7: Motore di Backtest Walk-Forward Multi-Valuta e Pipeline CLI di Produzione
* **Obiettivo:** Simulare l'intero ecosistema dal 2015 a oggi tenendo conto dei costi reali e generare il comando CLI per l'operatività mensile/giornaliera.
* **Deliverables:**
  1. Simulatore Walk-Forward che modella:
     * Buffer di turnover mensile sulle azioni (vendita solo se il titolo scende sotto il terzo decile del ranking regionale).
     * Costi di transazione differenziati (es. 5 bps su azioni USA/Canada, 10 bps + eventuale FTT/Stamp Duty su azioni europee, 1.5 bps sui future `ES` e `FXXP` incluso il roll trimestrale).
  2. Reportistica quantitativa: Sharpe, Sortino, Calmar, Max Drawdown, scomposizione dell'Alpha per regione (`NA` vs `EU`) e contributo netto dei due overlay future.
  3. Script CLI (`uv run quant-portfolio --mode delta`) che aggiorna i dati recenti via `xbbg`, calcola i pesi azionari e restituisce il numero esatto di contratti `ES`/`MES` e `FXXP`/`FESX` da vendere a copertura.