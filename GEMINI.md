# GEMINI.md - Istruzioni di Architettura, Vincoli Operativi e Piano di Sviluppo

Documento normativo e piano di implementazione per lo sviluppo del framework quantitativo di asset allocation long-only (STOXX Europe 600) con modulo di denoising spettrale, machine learning cross-sectional e copertura dinamica tramite future su indici azionari.

---

## 1. Vincoli Tecnologici e Standard di Ingegneria del Software

Tutti gli agenti software, script e moduli generati all'interno di questo repository devono conformarsi tassativamente ai seguenti standard. Nessuna eccezione è ammessa senza esplicita approvazione dell'utente.

### 1.1 Stack Tecnologico di Riferimento
*   **Runtime:** Python `>= 3.12`. È vietato l'uso di costrutti deprecati o compatibilità retroattiva con versioni `<= 3.11`.
*   **Gestione dell'Ambiente e Pacchetti:** `uv` come unico package manager e tool di isolamento virtuale. Tutti i comandi di esecuzione devono essere prefissati con `uv run` (es. `uv run pytest`, `uv run ruff check .`).
*   **Controllo di Versione:** `git`. Ogni modulo completato deve essere accompagnato da commit atomici, descrittivi e strutturati secondo la convenzione *Conventional Commits* (`feat:`, `fix:`, `refactor:`, `test:`, `perf:`).

### 1.2 Qualità del Codice e Tipizzazione Statica
*   **Static Type Checking:** `mypy` configurato in modalità rigorosa (`--strict`).
    *   Tutte le funzioni e i metodi devono avere annotazioni di tipo complete per argomenti e valori di ritorno (`Callable`, `Sequence`, `Mapping`, `Optional`, ecc.).
    *   Vietato l'uso di `Any` se non strettamente inevitabile e debitamente documentato con commento motivato.
    *   Tutti i DataFrame e Series devono essere semanticamente descritti o annotati tramite strumenti come `pandera` o type hinting avanzato (`pd.DataFrame`).
*   **Linting e Formatting:** `ruff` come unico orchestratore per linting e formattazione del codice.
    *   Regole minime attive: `E` (Pyflakes/pycodestyle), `F`, `B` (flake8-bugbear), `I` (isort), `UP` (pyupgrade), `SIM` (flake8-simplify), `TCH` (flake8-type-checking).
    *   Lunghezza massima linea: 100 caratteri.
    *   Nessun blocco di formattazione manuale: il codice deve superare `uv run ruff check --fix` e `uv run ruff format`.

### 1.3 Test Unitari e Validazione Statistica
*   **Framework di Test:** `pytest`.
*   **Copertura Minima:** 90% sui moduli deterministici (I/O dati, denoising matriciale, calcolo indici finanziari, partizionamento temporale).
*   **Test di Integrità Finanziaria:** Ogni algoritmo matematico (es. Marcenko-Pastur, HRP, calcolo Z-score) deve includere test di regressione contro dataset sintetici noti con tolleranze analitiche finite (`pytest.approx` o asserzioni su autovalori/tracce invarianti).
*   **Zero Leakage Enforcement:** I test unitari devono verificare espressamente che non vi sia look-ahead bias nei dataset di input (ad esempio controllando che `as_of_date >= filing_date`).

### 1.4 Regole di Gestione Dati e API Bloomberg
*   **Caching e Idempotenza Assoluta:** Qualsiasi estrazione dati da Bloomberg (BQL/BLPAPI) deve passare attraverso un layer di persistenza locale (`SQLite` o file `Parquet`).
*   Se un record identificato da `(ticker, date)` è già presente nel database locale, la richiesta API **non deve** essere eseguita.
*   È severamente proibito inserire chiamate dinamiche a Bloomberg nei moduli di backtesting, ottimizzazione o machine learning. Bloomberg è esclusivamente una sorgente "cold-storage" da interrogare in batch preliminari.

---

## 2. Architettura del Sistema Locale

```text
├── data/
│   ├── raw/                  # File temporanei di staging (estrazioni BQL bulk)
│   ├── processed/            # File Parquet partizionati per anno (prezzi e volumi EOD)
│   └── market_store.db       # Database SQLite (costituenti storici, fondamentali PiT, macro)
├── src/
│   ├── ingestion/            # Script di estrazione idempotente Bloomberg BQL e connettori
│   ├── features/             # Calcolo fattori cross-sectional e Z-score settoriali
│   ├── models/               # Machine Learning (LightGBM/RF) e Purged Time-Series Split
│   ├── risk/                 # RMT Denoising, Hierarchical Risk Parity, Mean-CVaR Solver
│   ├── overlay/              # Hidden Markov Models (HMM), sensori macro e regole future
│   └── execution/            # Motore di backtesting walk-forward e ribilanciamento delta
├── tests/                    # Suite di test pytest unitari e di integrazione
├── pyproject.toml            # Configurazione uv, ruff, mypy, pytest
└── GEMINI.md                 # Piano operativo e linee guida
```

---

## 3. Piano Operativo Dettagliato a Fasi

### Fase 1: Setup dell'Infrastruttura, Database Locale e Layer Idempotente
*   **Obiettivo:** Creare il database locale SQLite e le tabelle Parquet predisponendo l'ambiente `uv` con le dipendenze richieste (`blpapi`, `polars`/`pandas`, `pyarrow`, `scikit-learn`, `scipy`, `cvxpy`, `hmmlearn`, `ruff`, `mypy`, `pytest`).
*   **Schema Database SQLite (`market_store.db`):**
    *   `universe_members`: `(date TEXT, ticker TEXT, gics_sector TEXT, is_active INTEGER, PRIMARY KEY (date, ticker))`
    *   `fundamentals_pit`: `(as_of_date TEXT, filing_date TEXT, ticker TEXT, metric_name TEXT, metric_value REAL, PRIMARY KEY (as_of_date, ticker, metric_name))`
    *   `macro_series`: `(date TEXT, series_id TEXT, value REAL, PRIMARY KEY (date, series_id))`
*   **Tabella Parquet (`data/processed/prices_eod.parquet`):**
    *   Colonne: `date (date32)`, `ticker (string)`, `px_last (float64)`, `volume (float64)`, `ret_1d (float64)`. Partizionamento: `year`.
*   **Requisito Tecnico:** Scrivere moduli in `src/ingestion/storage.py` con controlli preventivi di presenza record per garantire l'idempotenza di ogni operazione di scrittura.

### Fase 2: Universo Storico STOXX Europe 600 Point-in-Time (Survivorship-Bias Free)
*   **Obiettivo:** Estrarre l'esatta composizione storica mensile dell'indice `SXXP Index` per gli ultimi 15 anni, identificando tutti i titoli che ne hanno fatto parte (inclusi i delistati).
*   **Implementazione:**
    *   Eseguire una query aggregata via Bloomberg Query Language (BQL) con frequenza mensile (`frq=m`) sull'universo `members('SXXP Index', dates=range(-15y, 0d, frq=m))`.
    *   Salvare la composizione mese per mese in `universe_members`.
    *   Estrarre la lista distinta dei ticker unici storici (circa 900–1.100 strumenti).
    *   Scaricare in blocco (bulk call) le serie storiche giornaliere di prezzo rettificato (`PX_LAST`) e volume per i soli ticker unici, salvandole in formato Parquet.
*   **Verifica:** Test di coerenza per verificare che i titoli falliti/fusi storici (es. Wirecard, Credit Suisse) siano presenti nel database fino alla data esatta di esclusione.

### Fase 3: Estrazione Parsimoniosa dei Fondamentali (Set Ristretto a 12-15 Feature)
*   **Obiettivo:** Ridurre il carico computazionale e il consumo di API scaricando solo i fattori core non derivabili dai prezzi, delegando i calcoli complessi al cloud Bloomberg.
*   **Query BQL sui Fondamentali (Frequenza Trimestrale/Mensile PiT):**
    *   *Valuation:* `PE_RATIO`, `EV_TO_T12M_EBITDA`, `FREE_CASH_FLOW_YIELD`.
    *   *Quality & Profitability:* `RETURN_COM_EQY` (ROE), `GROSS_PROFIT_MARGIN`, `OPER_MARGIN`.
    *   *Solvency & Risk:* `TOT_DEBT_TO_TOT_EQY`, `QUICK_RATIO`.
    *   *Growth & Investment:* `SALES_REV_TURN_GROWTH_1YR`, `ASSET_GROWTH` (YoY).
    *   *Earnings Quality:* Differenziale `NET_INCOME - CF_CASH_FROM_OPER` scalato su `TOTAL_ASSETS`.
*   **Feature Calcolate in Locale (Zero Chiamate Bloomberg):**
    *   *Momentum 1m (Reversal):* Rendimento a 21 giorni lavorativi.
    *   *Momentum 12m:* Rendimento cumulato a 252 giorni escluso l'ultimo mese.
    *   *Idiosyncratic Volatility:* Deviazione standard dei residui della regressione CAPM a 252 giorni contro lo STOXX 600.
    *   *Market Beta:* Covarianza titolo/benchmark scalata sulla varianza benchmark su finestra 252 giorni.
*   **Strategia di Esecuzione:** Download a blocchi triennali o settoriali salvati immediatamente in `market_store.db`.

### Fase 4: Feature Engineering Cross-Sectional e Machine Learning Ranker
*   **Obiettivo:** Costruire il predittore statistico per il ranking mensile dei titoli.
*   **Normalizzazione Dati:**
    *   Alla fine di ogni mese $t$, applicare la trasformazione in Z-Score cross-sectional per ogni feature $k$ raggruppata per settore GICS:
        $$z_{i,k,t} = \frac{x_{i,k,t} - \mu_{gics, k, t}}{\sigma_{gics, k, t}}$$
    *   Applicare Winsorization a $\pm 3\sigma$ per neutralizzare gli outlier contabili estremi.
*   **Target di Modello:** Rendimento relativo forward a 21 giorni ($t+1$) rispetto alla mediana cross-sectional dell'universo attivo.
*   **Modello:** Ensemble basato su alberi decisionali (`LightGBM` o `RandomForestClassifier` / `Regressor`).
*   **Cross-Validation:** Purged Group Time-Series Split con embargo di almeno 5 giorni lavorativi tra i fold per azzerare l'autocorrelazione seriale dei rendimenti.
*   **Output:** Selezione mensile del primo decile (top 10%, circa 50-60 titoli con l'alpha atteso più elevato).

### Fase 5: Denoising Spettrale (RMT) e Allocazione del Rischio (HRP / CVaR)
*   **Obiettivo:** Ottimizzare i pesi dei 50 titoli selezionati eliminando il rumore empirico dalla matrice di covarianza.
*   **Denoising tramite Teoria delle Matrici Casuali (Marcenko-Pastur):**
    1.  Calcolo della matrice di correlazione empirica $\mathbf{C}$ sui rendimenti giornalieri a 504 giorni ($T=504, N=50$).
    2.  Scomposizione spettrale: $\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$.
    3.  Calcolo del limite teorico del rumore stocastico:
        $$\lambda_{max} = \sigma^2 \left(1 + \sqrt{\frac{N}{T}}\right)^2$$
    4.  Sostituzione di tutti gli autovalori $\lambda_i \le \lambda_{max}$ con la loro media aritmetica $\bar{\lambda}$, mantenendo invariata la traccia di $\mathbf{C}$.
    5.  Ricostruzione della matrice di covarianza denoisata $\mathbf{\Sigma}_{denoised}$.
*   **Algoritmo di Allocazione:**
    *   Applicazione della **Hierarchical Risk Parity (HRP)**: calcolo della distanza topologica $d_{i,j} = \sqrt{\frac{1 - \rho_{i,j}}{2}}$, clustering gerarchico (Single/Ward Linkage), quasi-diagonalizzazione e Recursive Bisection.
    *   Alternativa: Risoluzione convessa con programmazione lineare per minimizzare il Conditional Value at Risk (Mean-CVaR al 95%) con vincolo di peso massimo al 4% per singolo emittente.
*   **Output:** Vettore dei pesi percentuali target $\mathbf{w}_t$ (cardinalità finale: 35-50 azioni).

### Fase 6: Sensori Macroeconomici e Modulo di Copertura con Future (Overlay)
*   **Obiettivo:** Determinare dinamicamente la frazione di capitale da coprire vendendo contratti future sull'Euro Stoxx 50 (`FESX`) o STOXX 600 (`FXXP`) per proteggersi dai crolli di mercato.
*   **Input Macro (da FRED o Bloomberg, scaricati una tantum):**
    *   Spread High Yield europeo (Option-Adjusted Spread).
    *   Pendenza della curva dei tassi (Spread decennale/biennale Bund: 10Y - 2Y).
    *   Indice di volatilità implicita V2TX.
*   **Rilevamento dei Regimi:**
    *   Addestramento di un **Hidden Markov Model (HMM)** gaussiano a 3 stati latenti:
        *   Stato 0: Regime Calmo / Bull Market.
        *   Stato 1: Regime di Transizione / Incertezza Elevata.
        *   Stato 2: Regime di Stress Sistemico / Bear Market Severo.
*   **Regole di Hedging Discreto:**
    *   Se $P(\text{Stato 0}) > 0.65$: Hedge Ratio = $0\%$.
    *   Se $P(\text{Stato 1}) > 0.65$: Hedge Ratio = $40\%$.
    *   Se $P(\text{Stato 2}) > 0.65$: Hedge Ratio = $80\% - 100\%$.
*   **Bande di Isteresi (Anti-Overtrading):** La variazione della posizione di copertura scatta solo se la transizione di stato probabilistica permane per almeno 3 giorni lavorativi consecutivi, impedendo costose oscillazioni transazionali sul future.

### Fase 7: Motore di Backtest Walk-Forward e Routine Delta di Produzione
*   **Obiettivo:** Verificare la redditività corretta per il rischio storica e predisporre lo script di produzione eseguibile a regime.
*   **Simulazione Walk-Forward:**
    *   Finestra rolling con riaddestramento del Machine Learning ogni 6 mesi e ribilanciamento mensile dei pesi di portafoglio.
    *   Modellazione realistica degli attriti: commissioni fisse per contratto/eseguito, tassa sulle transazioni finanziarie (Tobin Tax ove applicabile), e slippage proporzionale al volume.
    *   Metriche di validazione target: Sharpe Ratio $> 1.2$, Maximum Drawdown ridotto di almeno il 50% rispetto al benchmark passivo STOXX 600.
*   **Routine di Produzione "Delta" (Fine Mese):**
    *   Lo script interroga Bloomberg esclusivamente per l'ultimo mese:
        1.  Aggiornamento componenti attivi dell'indice ($1$ chiamata BQL).
        2.  Aggiornamento prezzi ultimi 20 giorni e nuovi bilanci rilasciati.
        3.  Scrittura dei delta su SQLite e Parquet.
        4.  Generazione della distinta pesi target azionari e numero di contratti future da vendere/comprare.
    *   Tempo di esecuzione stimato a regime: $< 90$ secondi.