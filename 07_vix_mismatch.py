from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import parametri as par


BASE_DIR = Path(__file__).resolve().parent

MATURITIES = (
    1,
    3,
    5,
    10,
)

OUTPUT_CURRENT = (
    par.OUTPUT_DIR
    / "vix_mismatch_corrente.csv"
)

OUTPUT_BACKTEST = (
    par.OUTPUT_DIR
    / "vix_mismatch_backtest.csv"
)

OUTPUT_SUMMARY = (
    par.OUTPUT_DIR
    / "vix_mismatch_sintesi.csv"
)

OUTPUT_PLOT_CURRENT = (
    par.OUTPUT_DIR
    / "vix_mismatch_term_structure.png"
)

OUTPUT_PLOT_HISTORICAL = (
    par.OUTPUT_DIR
    / "vix_mismatch_rmse_scadenza.png"
)


def load_heston_module():
    file_path = (
        BASE_DIR
        / "05_montecarlo_heston.py"
    )

    spec = (
        importlib.util.spec_from_file_location(
            "montecarlo_heston",
            file_path,
        )
    )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def bs_put_price(
    S,
    G,
    r,
    fee,
    sigma,
    T,
):
    if T <= 0:
        return max(
            G - S,
            0.0,
        )

    if sigma <= 0:
        return max(
            G * np.exp(-r * T)
            - S * np.exp(-fee * T),
            0.0,
        )

    d1 = (
        np.log(S / G)
        + (
            r
            - fee
            + 0.5 * sigma ** 2
        ) * T
    ) / (
        sigma * np.sqrt(T)
    )

    d2 = (
        d1
        - sigma * np.sqrt(T)
    )

    return (
        G
        * np.exp(-r * T)
        * norm.cdf(-d2)
        - S
        * np.exp(-fee * T)
        * norm.cdf(-d1)
    )


def bs_put_delta(
    S,
    G,
    r,
    fee,
    sigma,
    T,
):
    if T <= 0:
        return (
            -1.0
            if S < G
            else 0.0
        )

    if sigma <= 0:
        future_S = (
            S
            * np.exp(
                (r - fee) * T
            )
        )

        return (
            -np.exp(-fee * T)
            if future_S < G
            else 0.0
        )

    d1 = (
        np.log(S / G)
        + (
            r
            - fee
            + 0.5 * sigma ** 2
        ) * T
    ) / (
        sigma * np.sqrt(T)
    )

    return (
        np.exp(-fee * T)
        * (
            norm.cdf(d1)
            - 1.0
        )
    )


def log_returns(series):
    return np.log(
        series
        / series.shift(1)
    ).dropna()


def first_trading_date_on_or_after(
    index,
    target_date,
):
    position = index.searchsorted(
        pd.Timestamp(target_date),
        side="left",
    )

    if position >= len(index):
        return None

    return index[position]


def load_market_data():
    data = pd.read_csv(
        par.FILE_DATI_MERCATO,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()

    required = {
        "INDEX_EUR",
        "VIX",
    }

    for maturity in MATURITIES:
        required.add(
            f"RiskFree_{maturity}Y"
        )

    missing = required.difference(
        data.columns
    )

    if missing:
        raise RuntimeError(
            f"Mancano le colonne "
            f"{sorted(missing)}."
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data.loc[
        ~data.index.duplicated(
            keep="first"
        )
    ]


def get_rate(
    data,
    date,
    maturity,
):
    column = (
        f"RiskFree_{maturity}Y"
    )

    rates = data.loc[
        data.index <= date,
        column,
    ].dropna()

    if rates.empty:
        raise RuntimeError(
            f"Nessun tasso {maturity}Y "
            f"disponibile alla data "
            f"{date.date()}."
        )

    return float(
        rates.iloc[-1]
    )


def get_vix(
    data,
    date,
):
    vix = data.loc[
        data.index <= date,
        "VIX",
    ].dropna()

    if vix.empty:
        raise RuntimeError(
            f"Nessun VIX disponibile "
            f"alla data {date.date()}."
        )

    return float(
        vix.iloc[-1]
    ) / 100.0


def find_equivalent_sigma(
    target_price,
    maturity,
    risk_free,
):
    def objective(sigma):
        return (
            bs_put_price(
                par.INITIAL_FUND_VALUE,
                par.G,
                risk_free,
                par.TOTAL_FEE_ANNUAL,
                sigma,
                maturity,
            )
            - target_price
        )

    lower = 1e-6
    upper = 3.0

    if (
        objective(lower)
        * objective(upper)
        > 0
    ):
        raise RuntimeError(
            f"Impossibile trovare "
            f"sigma equivalente per "
            f"T={maturity}."
        )

    return float(
        brentq(
            objective,
            lower,
            upper,
        )
    )


def current_mismatch_analysis(
    data,
    heston,
):
    required_columns = [
        "INDEX_EUR",
        "VIX",
    ]

    required_columns += [
        f"RiskFree_{maturity}Y"
        for maturity in MATURITIES
    ]

    valid = data.dropna(
        subset=required_columns
    )

    if valid.empty:
        raise RuntimeError(
            "Nessuna data corrente "
            "con tutti gli input richiesti."
        )

    valuation_date = (
        valid.index[-1]
    )

    index_series = (
        data.loc[
            :valuation_date,
            "INDEX_EUR",
        ]
        .dropna()
    )

    parameters = (
        heston.estimate_heston_parameters(
            index_series
        )
    )

    vix_sigma = get_vix(
        data,
        valuation_date,
    )

    results = []

    for maturity in MATURITIES:
        risk_free = get_rate(
            data,
            valuation_date,
            maturity,
        )

        (
            heston_price,
            heston_delta,
        ) = (
            heston.heston_put_price_delta(
                par.INITIAL_FUND_VALUE,
                par.G,
                parameters["v0"],
                risk_free,
                par.TOTAL_FEE_ANNUAL,
                float(maturity),
                parameters,
            )
        )

        equivalent_sigma = (
            find_equivalent_sigma(
                heston_price,
                float(maturity),
                risk_free,
            )
        )

        vix_price = bs_put_price(
            par.INITIAL_FUND_VALUE,
            par.G,
            risk_free,
            par.TOTAL_FEE_ANNUAL,
            vix_sigma,
            float(maturity),
        )

        vix_delta = bs_put_delta(
            par.INITIAL_FUND_VALUE,
            par.G,
            risk_free,
            par.TOTAL_FEE_ANNUAL,
            vix_sigma,
            float(maturity),
        )

        price_error = (
            vix_price
            - heston_price
        )

        if heston_price != 0:
            price_error_pct = (
                price_error
                / heston_price
                * 100.0
            )
        else:
            price_error_pct = np.nan

        results.append(
            {
                "data_valutazione": (
                    valuation_date
                ),
                "scadenza_anni": maturity,
                "vix_sigma": vix_sigma,
                "sigma_bs_equivalente_heston": (
                    equivalent_sigma
                ),
                "gap_sigma": (
                    vix_sigma
                    - equivalent_sigma
                ),
                "tasso_risk_free": risk_free,
                "prezzo_vix": vix_price,
                "prezzo_heston": heston_price,
                "errore_prezzo_vix": (
                    price_error
                ),
                "errore_prezzo_vix_pct": (
                    price_error_pct
                ),
                "delta_vix": vix_delta,
                "delta_heston": heston_delta,
                "errore_delta": (
                    vix_delta
                    - heston_delta
                ),
            }
        )

    return (
        pd.DataFrame(results),
        parameters,
        valuation_date,
    )


def generate_windows(
    data,
    maturity,
):
    index_series = (
        data[
            "INDEX_EUR"
        ].dropna()
    )

    index = index_series.index

    rate_column = (
        f"RiskFree_{maturity}Y"
    )

    valid_start_data = (
        data[
            [
                "INDEX_EUR",
                "VIX",
                rate_column,
            ]
        ]
        .dropna()
    )

    if valid_start_data.empty:
        return []

    first_start = (
        valid_start_data.index[0]
    )

    last_start = (
        index.max()
        - pd.DateOffset(
            years=maturity
        )
    )

    windows = []

    target_start = (
        first_start
    )

    while (
        target_start
        <= last_start
    ):
        start_date = (
            first_trading_date_on_or_after(
                index,
                target_start,
            )
        )

        if start_date is None:
            break

        target_end = (
            start_date
            + pd.DateOffset(
                years=maturity
            )
        )

        end_date = (
            first_trading_date_on_or_after(
                index,
                target_end,
            )
        )

        if (
            end_date is not None
            and end_date > start_date
        ):
            windows.append(
                (
                    start_date,
                    end_date,
                )
            )

        target_start = (
            start_date
            + pd.DateOffset(
                years=1
            )
        )

    return list(
        dict.fromkeys(
            windows
        )
    )


def historical_mismatch_analysis(
    data,
):
    index_series = (
        data[
            "INDEX_EUR"
        ].dropna()
    )

    results = []

    for maturity in MATURITIES:
        windows = generate_windows(
            data,
            maturity,
        )

        for (
            start_date,
            end_date,
        ) in windows:
            path = index_series.loc[
                start_date:end_date
            ]

            returns = log_returns(
                path
            )

            if len(returns) < 2:
                continue

            realized_sigma = float(
                returns.std(
                    ddof=1
                )
                * np.sqrt(
                    par.TRADING_DAYS_PER_YEAR
                )
            )

            vix_sigma = get_vix(
                data,
                start_date,
            )

            risk_free = get_rate(
                data,
                start_date,
                maturity,
            )

            vix_price = bs_put_price(
                par.INITIAL_FUND_VALUE,
                par.G,
                risk_free,
                par.TOTAL_FEE_ANNUAL,
                vix_sigma,
                float(maturity),
            )

            realized_price = (
                bs_put_price(
                    par.INITIAL_FUND_VALUE,
                    par.G,
                    risk_free,
                    par.TOTAL_FEE_ANNUAL,
                    realized_sigma,
                    float(maturity),
                )
            )

            vix_delta = bs_put_delta(
                par.INITIAL_FUND_VALUE,
                par.G,
                risk_free,
                par.TOTAL_FEE_ANNUAL,
                vix_sigma,
                float(maturity),
            )

            realized_delta = (
                bs_put_delta(
                    par.INITIAL_FUND_VALUE,
                    par.G,
                    risk_free,
                    par.TOTAL_FEE_ANNUAL,
                    realized_sigma,
                    float(maturity),
                )
            )

            sigma_error = (
                vix_sigma
                - realized_sigma
            )

            price_error = (
                vix_price
                - realized_price
            )

            if realized_price != 0:
                price_error_pct = (
                    price_error
                    / realized_price
                    * 100.0
                )
            else:
                price_error_pct = (
                    np.nan
                )

            results.append(
                {
                    "data_inizio": (
                        start_date
                    ),
                    "data_fine": (
                        end_date
                    ),
                    "scadenza_anni": (
                        maturity
                    ),
                    "vix_sigma": (
                        vix_sigma
                    ),
                    "vol_realizzata_futura": (
                        realized_sigma
                    ),
                    "errore_sigma": (
                        sigma_error
                    ),
                    "tasso_risk_free": (
                        risk_free
                    ),
                    "prezzo_vix": (
                        vix_price
                    ),
                    "prezzo_vol_realizzata": (
                        realized_price
                    ),
                    "errore_prezzo": (
                        price_error
                    ),
                    "errore_prezzo_pct": (
                        price_error_pct
                    ),
                    "delta_vix": (
                        vix_delta
                    ),
                    "delta_vol_realizzata": (
                        realized_delta
                    ),
                    "errore_delta": (
                        vix_delta
                        - realized_delta
                    ),
                }
            )

    if not results:
        raise RuntimeError(
            "Nessun backtest VIX "
            "disponibile."
        )

    return pd.DataFrame(
        results
    )


def summarize_historical(
    backtest,
):
    data = backtest.copy()

    data[
        "errore_sigma_assoluto"
    ] = (
        data[
            "errore_sigma"
        ].abs()
    )

    data[
        "errore_sigma_quadrato"
    ] = (
        data[
            "errore_sigma"
        ] ** 2
    )

    data[
        "errore_prezzo_assoluto"
    ] = (
        data[
            "errore_prezzo"
        ].abs()
    )

    data[
        "errore_prezzo_quadrato"
    ] = (
        data[
            "errore_prezzo"
        ] ** 2
    )

    data[
        "vix_sovrastima"
    ] = (
        data[
            "errore_sigma"
        ] > 0
    )

    summary = (
        data.groupby(
            "scadenza_anni",
            as_index=False,
        )
        .agg(
            numero_finestre=(
                "data_inizio",
                "count",
            ),
            vix_medio=(
                "vix_sigma",
                "mean",
            ),
            vol_realizzata_media=(
                "vol_realizzata_futura",
                "mean",
            ),
            bias_sigma=(
                "errore_sigma",
                "mean",
            ),
            mae_sigma=(
                "errore_sigma_assoluto",
                "mean",
            ),
            mse_sigma=(
                "errore_sigma_quadrato",
                "mean",
            ),
            errore_prezzo_medio=(
                "errore_prezzo",
                "mean",
            ),
            mae_prezzo=(
                "errore_prezzo_assoluto",
                "mean",
            ),
            mse_prezzo=(
                "errore_prezzo_quadrato",
                "mean",
            ),
            errore_prezzo_pct_medio=(
                "errore_prezzo_pct",
                "mean",
            ),
            frequenza_sovrastima=(
                "vix_sovrastima",
                "mean",
            ),
            errore_delta_medio=(
                "errore_delta",
                "mean",
            ),
        )
    )

    summary[
        "rmse_sigma"
    ] = np.sqrt(
        summary[
            "mse_sigma"
        ]
    )

    summary[
        "rmse_prezzo"
    ] = np.sqrt(
        summary[
            "mse_prezzo"
        ]
    )

    summary[
        "frequenza_sovrastima_pct"
    ] = (
        summary[
            "frequenza_sovrastima"
        ]
        * 100.0
    )

    return (
        summary.drop(
            columns=[
                "mse_sigma",
                "mse_prezzo",
                "frequenza_sovrastima",
            ]
        )
        .sort_values(
            "scadenza_anni"
        )
        .reset_index(
            drop=True
        )
    )


def create_plots(
    current,
    summary,
):
    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        current[
            "scadenza_anni"
        ],
        current[
            "vix_sigma"
        ] * 100,
        marker="o",
        label="VIX 30d estrapolato",
    )

    plt.plot(
        current[
            "scadenza_anni"
        ],
        current[
            "sigma_bs_equivalente_heston"
        ] * 100,
        marker="o",
        label=(
            "Sigma BS equivalente Heston"
        ),
    )

    plt.xlabel(
        "Scadenza (anni)"
    )

    plt.ylabel(
        "Volatilita' (%)"
    )

    plt.title(
        "VIX 30 giorni e volatilita' "
        "equivalente per scadenza"
    )

    plt.xticks(
        MATURITIES
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT_CURRENT,
        dpi=150,
    )

    plt.show()

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        summary[
            "scadenza_anni"
        ],
        summary[
            "rmse_sigma"
        ] * 100,
        marker="o",
    )

    plt.xlabel(
        "Scadenza (anni)"
    )

    plt.ylabel(
        "RMSE errore volatilita' "
        "(punti percentuali)"
    )

    plt.title(
        "Errore storico del VIX "
        "estrapolato per scadenza"
    )

    plt.xticks(
        MATURITIES
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT_HISTORICAL,
        dpi=150,
    )

    plt.show()


def main():
    par.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_market_data()

    heston = (
        load_heston_module()
    )

    (
        current,
        parameters,
        valuation_date,
    ) = current_mismatch_analysis(
        data,
        heston,
    )

    backtest = (
        historical_mismatch_analysis(
            data
        )
    )

    summary = (
        summarize_historical(
            backtest
        )
    )

    current.to_csv(
        OUTPUT_CURRENT,
        index=False,
        date_format="%Y-%m-%d",
    )

    backtest.to_csv(
        OUTPUT_BACKTEST,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    create_plots(
        current,
        summary,
    )

    print(
        "=== VIX MATURITY MISMATCH ==="
    )

    print(
        f"Data di valutazione: "
        f"{valuation_date.date()}"
    )

    print(
        f"VIX corrente: "
        f"{current['vix_sigma'].iloc[0]:.2%}"
    )

    print(
        f"S0: "
        f"{par.INITIAL_FUND_VALUE:.4f}"
    )

    print(
        f"G: "
        f"{par.G:.4f}"
    )

    print(
        f"Fee: "
        f"{par.TOTAL_FEE_ANNUAL:.4%}"
    )

    print(
        "\n=== PARAMETRI HESTON "
        "USATI COME BENCHMARK ==="
    )

    print(
        f"kappa: "
        f"{parameters['kappa']:.6f}"
    )

    print(
        f"theta: "
        f"{parameters['theta']:.6f}"
    )

    print(
        f"xi: "
        f"{parameters['xi']:.6f}"
    )

    print(
        f"rho: "
        f"{parameters['rho']:.6f}"
    )

    print(
        f"v0: "
        f"{parameters['v0']:.6f}"
    )

    print(
        "\n=== CONFRONTO CORRENTE ==="
    )

    print(
        current[
            [
                "scadenza_anni",
                "vix_sigma",
                "sigma_bs_equivalente_heston",
                "gap_sigma",
                "tasso_risk_free",
                "prezzo_vix",
                "prezzo_heston",
                "errore_prezzo_vix",
                "errore_prezzo_vix_pct",
                "errore_delta",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\n=== BACKTEST STORICO "
        "DEL MISMATCH ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nInterpretazione:"
    )

    print(
        "Nel confronto storico, "
        "la volatilita' realizzata futura "
        "e' usata come benchmark ex post."
    )

    print(
        "Il confronto Heston e' invece "
        "un benchmark di modello e non "
        "una volatilita' implicita "
        "di mercato osservata."
    )

    print(
        "\nFile salvati:"
    )

    print(
        f"  - {OUTPUT_CURRENT}"
    )

    print(
        f"  - {OUTPUT_BACKTEST}"
    )

    print(
        f"  - {OUTPUT_SUMMARY}"
    )

    print(
        f"  - {OUTPUT_PLOT_CURRENT}"
    )

    print(
        f"  - {OUTPUT_PLOT_HISTORICAL}"
    )


if __name__ == "__main__":
    main()
