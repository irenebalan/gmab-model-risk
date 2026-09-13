import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import parametri as par


ROLLING_MONTHS = 36

OUTPUT_SUMMARY = (
    par.OUTPUT_DIR
    / "validazione_proxy_sintesi.csv"
)

OUTPUT_MONTHLY = (
    par.OUTPUT_DIR
    / "validazione_proxy_mensile.csv"
)

OUTPUT_PLOT_LEVELS = (
    par.OUTPUT_DIR
    / "validazione_proxy_normalizzata.png"
)

OUTPUT_PLOT_CORRELATION = (
    par.OUTPUT_DIR
    / "validazione_proxy_correlazione_rolling.png"
)

OUTPUT_PLOT_TRACKING = (
    par.OUTPUT_DIR
    / "validazione_proxy_tracking_error_rolling.png"
)

OUTPUT_PLOT_ROLLING = (
    par.OUTPUT_DIR
    / "validazione_proxy_rolling.png"
)

def log_returns(series):
    return np.log(
        series
        / series.shift(1)
    )


def calculate_metrics(
    etf_returns,
    index_returns,
    periods_per_year,
):
    data = pd.concat(
        [
            etf_returns.rename("ETF"),
            index_returns.rename("INDEX"),
        ],
        axis=1,
    ).dropna()

    if len(data) < 2:
        raise RuntimeError(
            "Osservazioni insufficienti "
            "per calcolare le metriche."
        )

    correlation = float(
        data["ETF"].corr(
            data["INDEX"]
        )
    )

    variance_index = float(
        data["INDEX"].var(
            ddof=1
        )
    )

    if variance_index <= 0:
        raise RuntimeError(
            "Varianza dell'indice nulla "
            "o non valida."
        )

    covariance = float(
        data[
            ["ETF", "INDEX"]
        ]
        .cov()
        .loc[
            "ETF",
            "INDEX",
        ]
    )

    beta = (
        covariance
        / variance_index
    )

    alpha_period = float(
        data["ETF"].mean()
        - beta
        * data["INDEX"].mean()
    )

    alpha_annualized = (
        alpha_period
        * periods_per_year
    )

    etf_volatility = float(
        data["ETF"].std(
            ddof=1
        )
        * np.sqrt(
            periods_per_year
        )
    )

    index_volatility = float(
        data["INDEX"].std(
            ddof=1
        )
        * np.sqrt(
            periods_per_year
        )
    )

    active_returns = (
        data["ETF"]
        - data["INDEX"]
    )

    tracking_error = float(
        active_returns.std(
            ddof=1
        )
        * np.sqrt(
            periods_per_year
        )
    )

    active_return_annualized = float(
        active_returns.mean()
        * periods_per_year
    )

    residuals = (
        data["ETF"]
        - (
            alpha_period
            + beta
            * data["INDEX"]
        )
    )

    residual_basis_risk = float(
        residuals.std(
            ddof=1
        )
        * np.sqrt(
            periods_per_year
        )
    )

    r_squared = (
        correlation ** 2
    )

    return {
        "osservazioni": len(data),
        "correlazione": correlation,
        "r_squared": r_squared,
        "beta": beta,
        "alpha_annualizzata": alpha_annualized,
        "vol_etf_annualizzata": etf_volatility,
        "vol_indice_annualizzata": index_volatility,
        "tracking_error_annualizzato": tracking_error,
        "active_return_annualizzato": active_return_annualized,
        "basis_risk_residuo_annualizzato": residual_basis_risk,
    }


def main():
    par.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        par.FILE_DATI_MERCATO,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()

    required = {
        "ETF_EUR",
        "INDEX_EUR",
    }

    missing = required.difference(
        data.columns
    )

    if missing:
        raise RuntimeError(
            f"Mancano le colonne "
            f"{sorted(missing)}."
        )

    prices = (
        data[
            [
                "ETF_EUR",
                "INDEX_EUR",
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
    )

    if len(prices) < 2:
        raise RuntimeError(
            "Periodo comune ETF-indice "
            "insufficiente."
        )

    start_date = (
        prices.index[0]
    )

    end_date = (
        prices.index[-1]
    )

    elapsed_years = (
        end_date
        - start_date
    ).days / 365.25

    if elapsed_years <= 0:
        raise RuntimeError(
            "Periodo comune non valido."
        )

    daily_returns = pd.DataFrame(
        {
            "ETF": log_returns(
                prices["ETF_EUR"]
            ),
            "INDEX": log_returns(
                prices["INDEX_EUR"]
            ),
        }
    ).dropna()

    monthly_prices = (
        prices
        .resample("ME")
        .last()
        .dropna()
    )

    monthly_returns = pd.DataFrame(
        {
            "ETF": log_returns(
                monthly_prices[
                    "ETF_EUR"
                ]
            ),
            "INDEX": log_returns(
                monthly_prices[
                    "INDEX_EUR"
                ]
            ),
        }
    ).dropna()

    daily_metrics = (
        calculate_metrics(
            daily_returns["ETF"],
            daily_returns["INDEX"],
            par.TRADING_DAYS_PER_YEAR,
        )
    )

    monthly_metrics = (
        calculate_metrics(
            monthly_returns["ETF"],
            monthly_returns["INDEX"],
            12,
        )
    )

    cagr_etf = (
        (
            prices[
                "ETF_EUR"
            ].iloc[-1]
            / prices[
                "ETF_EUR"
            ].iloc[0]
        )
        ** (
            1.0
            / elapsed_years
        )
        - 1.0
    )

    cagr_index = (
        (
            prices[
                "INDEX_EUR"
            ].iloc[-1]
            / prices[
                "INDEX_EUR"
            ].iloc[0]
        )
        ** (
            1.0
            / elapsed_years
        )
        - 1.0
    )

    cagr_difference_etf_minus_index = (
        cagr_etf
        - cagr_index
    )

    proxy_drag_annual = (
        cagr_index
        - cagr_etf
    )

    proxy_drag_continuous = np.log(
        (1.0 + cagr_index)
        / (1.0 + cagr_etf)
    )

    summary = pd.DataFrame(
        [
            {
                "ETF": par.ETF_TICKER,
                "indice_proxy": par.LONG_INDEX_TICKER,
                "data_inizio": start_date.date(),
                "data_fine": end_date.date(),
                "anni_overlap": elapsed_years,

                "cagr_etf": cagr_etf,
                "cagr_indice": cagr_index,
                "differenza_cagr_etf_meno_indice": (
                    cagr_difference_etf_minus_index
                ),
                "proxy_drag_annuo": (
                    proxy_drag_annual
                ),
                "proxy_drag_continuo": (
                    proxy_drag_continuous
                ),

                "correlazione_monthly": (
                    monthly_metrics[
                        "correlazione"
                    ]
                ),
                "r_squared_monthly": (
                    monthly_metrics[
                        "r_squared"
                    ]
                ),
                "beta_monthly": (
                    monthly_metrics[
                        "beta"
                    ]
                ),
                "alpha_monthly_annualizzata": (
                    monthly_metrics[
                        "alpha_annualizzata"
                    ]
                ),
                "vol_etf_monthly_annualizzata": (
                    monthly_metrics[
                        "vol_etf_annualizzata"
                    ]
                ),
                "vol_indice_monthly_annualizzata": (
                    monthly_metrics[
                        "vol_indice_annualizzata"
                    ]
                ),
                "tracking_error_monthly": (
                    monthly_metrics[
                        "tracking_error_annualizzato"
                    ]
                ),
                "basis_risk_monthly": (
                    monthly_metrics[
                        "basis_risk_residuo_annualizzato"
                    ]
                ),

                "correlazione_daily": (
                    daily_metrics[
                        "correlazione"
                    ]
                ),
                "r_squared_daily": (
                    daily_metrics[
                        "r_squared"
                    ]
                ),
                "beta_daily": (
                    daily_metrics[
                        "beta"
                    ]
                ),
                "tracking_error_daily": (
                    daily_metrics[
                        "tracking_error_annualizzato"
                    ]
                ),
                "basis_risk_daily": (
                    daily_metrics[
                        "basis_risk_residuo_annualizzato"
                    ]
                ),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    monthly_results = (
        monthly_returns.copy()
    )

    monthly_results[
        "active_return"
    ] = (
        monthly_results["ETF"]
        - monthly_results["INDEX"]
    )

    rolling_correlation = (
        monthly_results[
            "ETF"
        ]
        .rolling(
            ROLLING_MONTHS
        )
        .corr(
            monthly_results[
                "INDEX"
            ]
        )
    )

    rolling_tracking_error = (
        monthly_results[
            "active_return"
        ]
        .rolling(
            ROLLING_MONTHS
        )
        .std(
            ddof=1
        )
        * np.sqrt(12)
    )

    monthly_results[
        f"correlazione_rolling_"
        f"{ROLLING_MONTHS}m"
    ] = rolling_correlation

    monthly_results[
        f"tracking_error_rolling_"
        f"{ROLLING_MONTHS}m"
    ] = rolling_tracking_error

    monthly_results.to_csv(
        OUTPUT_MONTHLY,
        index_label="Date",
    )

    normalized = (
        prices
        / prices.iloc[0]
        * 100.0
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        normalized.index,
        normalized[
            "ETF_EUR"
        ],
        label=(
            f"ETF {par.ETF_TICKER}"
        ),
    )

    plt.plot(
        normalized.index,
        normalized[
            "INDEX_EUR"
        ],
        label=(
            f"Proxy {par.LONG_INDEX_TICKER}"
        ),
    )

    plt.xlabel(
        "Data"
    )

    plt.ylabel(
        "Valore normalizzato"
    )

    plt.title(
        "ETF e indice proxy in EUR "
        "(base 100)"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT_LEVELS,
        dpi=150,
    )

    plt.show()

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        monthly_results.index,
        rolling_correlation,
    )

    plt.axhline(
        y=monthly_metrics[
            "correlazione"
        ],
        linestyle="--",
        label="Correlazione mensile media",
    )

    plt.xlabel(
        "Data"
    )

    plt.ylabel(
        "Correlazione"
    )

    plt.title(
        f"Correlazione mensile rolling "
        f"{ROLLING_MONTHS} mesi"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT_CORRELATION,
        dpi=150,
    )

    plt.show()

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        monthly_results.index,
        rolling_tracking_error
        * 100,
    )

    plt.axhline(
        y=monthly_metrics[
            "tracking_error_annualizzato"
        ]
        * 100,
        linestyle="--",
        label=(
            "Tracking error mensile medio"
        ),
    )

    plt.xlabel(
        "Data"
    )

    plt.ylabel(
        "Tracking error annualizzato (%)"
    )

    plt.title(
        f"Tracking error mensile rolling "
        f"{ROLLING_MONTHS} mesi"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT_TRACKING,
        dpi=150,
    )

    plt.show()

    # --------------------------------------------------------
    # Correlazione e tracking error rolling
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    # Correlazione rolling
    axes[0].plot(
        monthly_results.index,
        rolling_correlation,
    )

    axes[0].axhline(
        y=monthly_metrics[
            "correlazione"
        ],
        linestyle="--",
        linewidth=1,
        label="Correlazione sull'intero periodo",
    )

    axes[0].set_ylabel(
        "Correlazione"
    )

    axes[0].set_title(
        f"Correlazione rolling a "
        f"{ROLLING_MONTHS} mesi"
    )

    axes[0].legend()

    # Tracking error rolling
    axes[1].plot(
        monthly_results.index,
        rolling_tracking_error
        * 100,
    )

    axes[1].axhline(
        y=monthly_metrics[
            "tracking_error_annualizzato"
        ]
        * 100,
        linestyle="--",
        linewidth=1,
        label=(
            "Tracking error "
            "sull'intero periodo"
        ),
    )

    axes[1].set_ylabel(
        "Tracking error annualizzato (%)"
    )

    axes[1].set_xlabel(
        "Data"
    )

    axes[1].set_title(
        f"Tracking error rolling a "
        f"{ROLLING_MONTHS} mesi"
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT_ROLLING,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "=== VALIDAZIONE PROXY STORICA ==="
    )

    print(
        f"ETF: "
        f"{par.ETF_TICKER}"
    )

    print(
        f"Indice proxy: "
        f"{par.LONG_INDEX_TICKER}"
    )

    print(
        f"Periodo comune: "
        f"{start_date.date()} - "
        f"{end_date.date()}"
    )

    print(
        f"Anni di overlap: "
        f"{elapsed_years:.2f}"
    )

    print(
        "\n=== PERFORMANCE DI LUNGO PERIODO ==="
    )

    print(
        f"CAGR ETF: "
        f"{cagr_etf:.4%}"
    )

    print(
        f"CAGR indice proxy: "
        f"{cagr_index:.4%}"
    )

    print(
        f"Differenza ETF - proxy: "
        f"{cagr_difference_etf_minus_index:.4%}"
    )

    print(
        f"Drag annuo della proxy "
        f"rispetto all'ETF: "
        f"{proxy_drag_annual:.4%}"
    )

    print(
        f"Drag continuo equivalente: "
        f"{proxy_drag_continuous:.4%}"
    )

    print(
        "\n=== CONFRONTO MENSILE "
        "(RISULTATO PRINCIPALE) ==="
    )

    print(
        f"Correlazione: "
        f"{monthly_metrics['correlazione']:.6f}"
    )

    print(
        f"R^2: "
        f"{monthly_metrics['r_squared']:.6f}"
    )

    print(
        f"Beta: "
        f"{monthly_metrics['beta']:.6f}"
    )

    print(
        f"Alpha annualizzata: "
        f"{monthly_metrics['alpha_annualizzata']:.4%}"
    )

    print(
        f"Vol ETF annualizzata: "
        f"{monthly_metrics['vol_etf_annualizzata']:.4%}"
    )

    print(
        f"Vol indice annualizzata: "
        f"{monthly_metrics['vol_indice_annualizzata']:.4%}"
    )

    print(
        f"Tracking error annualizzato: "
        f"{monthly_metrics['tracking_error_annualizzato']:.4%}"
    )

    print(
        f"Basis risk residuo "
        f"proxy-ETF annualizzato: "
        f"{monthly_metrics['basis_risk_residuo_annualizzato']:.4%}"
    )

    print(
        "\n=== CONFRONTO GIORNALIERO "
        "(SOLO DIAGNOSTICO) ==="
    )

    print(
        "ATTENZIONE: ETF e indice chiudono "
        "in orari diversi. Le metriche "
        "giornaliere sono influenzate "
        "dall'asincronia temporale."
    )

    print(
        f"Correlazione: "
        f"{daily_metrics['correlazione']:.6f}"
    )

    print(
        f"R^2: "
        f"{daily_metrics['r_squared']:.6f}"
    )

    print(
        f"Beta: "
        f"{daily_metrics['beta']:.6f}"
    )

    print(
        f"Tracking error annualizzato: "
        f"{daily_metrics['tracking_error_annualizzato']:.4%}"
    )

    print(
        f"Basis risk residuo "
        f"proxy-ETF annualizzato: "
        f"{daily_metrics['basis_risk_residuo_annualizzato']:.4%}"
    )

    print(
        "\nInterpretazione:"
    )

    print(
        "Per la validazione economica "
        "della proxy si utilizzano "
        "principalmente le metriche mensili."
    )

    print(
        "Il proxy_drag_annuo misura quanto "
        "l'indice proxy ha sovraperformato "
        "l'ETF nel periodo comune."
    )

    print(
        "Questo drag non viene confuso "
        "con TER e fee contrattuali e puo' "
        "essere utilizzato separatamente "
        "come sensitivita' nel backtest."
    )

    print(
        "\nFile salvati:"
    )

    print(
        f"  - {OUTPUT_SUMMARY}"
    )

    print(
        f"  - {OUTPUT_MONTHLY}"
    )

    print(
        f"  - {OUTPUT_PLOT_LEVELS}"
    )

    print(
        f"  - {OUTPUT_PLOT_CORRELATION}"
    )

    print(
        f"  - {OUTPUT_PLOT_TRACKING}"
    )

    print(
        f"  - {OUTPUT_PLOT_ROLLING}"
    )


if __name__ == "__main__":
    main()
