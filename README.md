# GMAB Model Risk

Questo repository contiene il codice Python sviluppato per la tesi magistrale:

**Pricing e hedging di una GMAB a scadenza fissa: un'analisi del rischio di modello**

L'analisi studia il rischio di modello nel pricing e nella copertura di una Guaranteed Minimum Accumulation Benefit (GMAB) a scadenza fissa, confrontando il modello di Black--Scholes con il modello di Heston.

## Struttura del codice

- `parametri.py`  
  Contiene i parametri contrattuali, di mercato e numerici condivisi dai diversi moduli.

- `01_dati.py`  
  Scarica e prepara i dati di mercato utilizzati nelle analisi successive.

- `02_volatilita.py`  
  Calcola le stime di volatilità storica su differenti finestre temporali.

- `03_pricing_gmab.py`  
  Implementa il pricing e il calcolo del delta della GMAB nel modello di Black--Scholes.

- `04_hedging_backtest.py`  
  Esegue il backtest storico della strategia di delta hedging su finestre decennali.

- `05_montecarlo_heston.py`  
  Stima i parametri del modello di Heston, esegue la simulazione Monte Carlo e confronta le strategie di copertura Heston e Black--Scholes. Implementa inoltre il confronto a capitale iniziale equo e la regressione dell'errore di copertura Heston sulla varianza media realizzata lungo i percorsi simulati.

- `05_montecarlo_heston_gamma.py`  
  Estende l'analisi Monte Carlo introducendo una misura di varianza realizzata ponderata per il gamma della garanzia, calcolato con la volatilità Black--Scholes equivalente al prezzo iniziale Heston. Stima una regressione dell'hedging error Heston sulla varianza media realizzata e sulla varianza gamma-pesata, riportando anche l'incremento dell'R-squared e il partial R-squared associato al secondo regressore.  

- `06_validazione_proxy.py`  
  Analizza la relazione tra l'ETF e l'indice proxy, considerando correlazione, beta, R-squared, tracking error e basis risk residuo.

- `07_vix_mismatch.py`  
  Analizza il maturity mismatch tra il VIX e gli orizzonti temporali più lunghi rilevanti per la GMAB.

## Requisiti

Le librerie Python necessarie sono indicate nel file `requirements.txt`.

Possono essere installate tramite:

```bash
pip install -r requirements.txt
```

## Ordine di esecuzione

Il dataset di mercato deve essere innanzitutto generato eseguendo:

```bash
python 01_dati.py
```

Le principali dipendenze tra i moduli sono:

```text
01 → 02 → 03
01 → 04
01 → 05_montecarlo_heston
01 → 05_montecarlo_heston_gamma
01 → 06
01 + funzioni di 05_montecarlo_heston → 07
```

Gli script successivi possono quindi essere eseguiti in funzione dell'analisi di interesse.

## Dati

I dati di mercato vengono acquisiti programmaticamente da fonti esterne.

L'analisi empirica utilizza dati fino al **20 agosto 2026**. Nel file `parametri.py` la data finale è impostata come:

```python
END_DATE = "2026-08-21"
```

poiché la data finale utilizzata da `yfinance` è esclusiva.

Le principali serie di mercato utilizzate comprendono:

- iShares MSCI USA UCITS ETF;
- S&P 500 Total Return Index;
- Cboe Volatility Index (VIX);
- cambio EUR/USD;
- tassi risk-free in euro.

I file CSV e le figure prodotti dagli script non sono inclusi nel repository e possono essere rigenerati eseguendo i relativi moduli.

## Tesi

Tesi magistrale in Economia e Finanza  
Università degli Studi di Milano-Bicocca

**Autrice:** Irene Balan
