from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import parametri as par


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
        sigma
        * np.sqrt(T)
    )

    d2 = (
        d1
        - sigma
        * np.sqrt(T)
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

        if future_S < G:
            return (
                -np.exp(
                    -fee * T
                )
            )

        return 0.0

    d1 = (
        np.log(S / G)
        + (
            r
            - fee
            + 0.5 * sigma ** 2
        ) * T
    ) / (
        sigma
        * np.sqrt(T)
    )

    return (
        np.exp(-fee * T)
        * (
            norm.cdf(d1)
            - 1.0
        )
    )


def load_historical_volatilities():
    data = pd.read_csv(
        par.FILE_VOLATILITA_FINESTRE,
        index_col="finestra",
    )

    if (
        "volatilita_annualizzata"
        not in data.columns
    ):
        raise ValueError(
            "Colonna "
            "'volatilita_annualizzata' "
            "non trovata in "
            f"{par.FILE_VOLATILITA_FINESTRE}"
        )

    return {
        f"storica_{label}": float(value)
        for (
            label,
            value,
        ) in data[
            "volatilita_annualizzata"
        ].items()
    }


def load_market_inputs():
    data = pd.read_csv(
        par.FILE_DATI_VOLATILITA,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()

    risk_free_column = (
        f"RiskFree_"
        f"{par.GMAB_MATURITY_YEARS}Y"
    )

    required = {
        "VIX_sigma",
        risk_free_column,
    }

    missing = (
        required.difference(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            f"Mancano le colonne "
            f"{sorted(missing)} in "
            f"{par.FILE_DATI_VOLATILITA}"
        )

    valid = data.dropna(
        subset=[
            "VIX_sigma",
            risk_free_column,
        ]
    )

    if valid.empty:
        raise ValueError(
            "Nessuna data con VIX "
            "e tasso risk-free disponibili."
        )

    valuation_date = (
        valid.index[-1]
    )

    vix_volatility = float(
        valid.loc[
            valuation_date,
            "VIX_sigma",
        ]
    )

    risk_free = float(
        valid.loc[
            valuation_date,
            risk_free_column,
        ]
    )

    return (
        valuation_date,
        vix_volatility,
        risk_free,
    )


def price_gmab(
    G,
    sigma,
    risk_free,
):
    price = bs_put_price(
        par.INITIAL_FUND_VALUE,
        G,
        risk_free,
        par.TOTAL_FEE_ANNUAL,
        sigma,
        par.GMAB_MATURITY_YEARS,
    )

    delta = bs_put_delta(
        par.INITIAL_FUND_VALUE,
        G,
        risk_free,
        par.TOTAL_FEE_ANNUAL,
        sigma,
        par.GMAB_MATURITY_YEARS,
    )

    return (
        price,
        delta,
    )


def main():
    historical_volatilities = (
        load_historical_volatilities()
    )

    (
        valuation_date,
        vix_volatility,
        risk_free,
    ) = load_market_inputs()

    scenarios = (
        historical_volatilities.copy()
    )

    scenarios[
        "proxy_vix_30d"
    ] = (
        vix_volatility
    )

    print(
        "=== Parametri GMAB ==="
    )

    print(
        f"S0: "
        f"{par.INITIAL_FUND_VALUE:.4f} "
        f"{par.CONTRACT_CURRENCY}"
    )

    print(
        f"Scadenza: "
        f"{par.GMAB_MATURITY_YEARS} anni"
    )

    print(
        f"Fee annuale totale: "
        f"{par.TOTAL_FEE_ANNUAL:.4%}"
    )

    print(
        f"Tasso risk-free "
        f"{par.GMAB_MATURITY_YEARS}Y: "
        f"{risk_free:.4%}"
    )

    print(
        f"VIX: "
        f"{vix_volatility * 100:.2f}%"
    )

    print(
        f"Data di valutazione: "
        f"{valuation_date.date()}"
    )

    print(
        "\nVolatilita' storiche "
        "stimate sull'indice proxy in EUR:"
    )

    for (
        scenario,
        sigma,
    ) in historical_volatilities.items():
        print(
            f"  {scenario}: "
            f"{sigma:.4%}"
        )

    results = []

    for guarantee_level in (
        par.GUARANTEE_LEVELS
    ):
        G = (
            par.PREMIO_LORDO
            * guarantee_level
        )

        for (
            scenario,
            sigma,
        ) in scenarios.items():
            (
                price,
                delta,
            ) = price_gmab(
                G,
                sigma,
                risk_free,
            )

            results.append(
                {
                    "livello_garanzia": (
                        guarantee_level
                    ),
                    "G": G,
                    "scenario_volatilita": (
                        scenario
                    ),
                    "sigma": sigma,
                    "tasso_risk_free": (
                        risk_free
                    ),
                    "fee_annua": (
                        par.TOTAL_FEE_ANNUAL
                    ),
                    "prezzo_gmab": (
                        price
                    ),
                    "delta_gmab": (
                        delta
                    ),
                }
            )

    results_table = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "livello_garanzia",
                "sigma",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n=== Prezzo e delta GMAB ==="
    )

    print(
        results_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    results_table.to_csv(
        par.FILE_PREZZO_GMAB,
        index=False,
    )

    print(
        f"\nFile salvato: "
        f"{Path(par.FILE_PREZZO_GMAB).resolve()}"
    )


if __name__ == "__main__":
    main()
