import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import parametri as par


WINDOWS_YEARS = {
    "1y": 1,
    "3y": 3,
    "5y": 5,
}

ROLLING_WINDOW_YEARS = 1


def log_returns(price_series):
    return np.log(
        price_series
        / price_series.shift(1)
    ).dropna()


def historical_volatility(
    returns,
    years,
):
    n_days = (
        years
        * par.TRADING_DAYS_PER_YEAR
    )

    if len(returns) < n_days:
        raise ValueError(
            f"Dati insufficienti per una "
            f"finestra di {years} anni: "
            f"{len(returns)} osservazioni disponibili."
        )

    return (
        returns.iloc[-n_days:]
        .std(ddof=1)
        * np.sqrt(
            par.TRADING_DAYS_PER_YEAR
        )
    )


def rolling_volatility(
    returns,
    years,
):
    window = (
        years
        * par.TRADING_DAYS_PER_YEAR
    )

    return (
        returns
        .rolling(window)
        .std(ddof=1)
        * np.sqrt(
            par.TRADING_DAYS_PER_YEAR
        )
    )


def main():
    par.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        par.FILE_DATI_MERCATO,
        index_col="Date",
        parse_dates=True,
    ).sort_index()

    if "INDEX_EUR" not in data.columns:
        raise RuntimeError(
            "La colonna INDEX_EUR "
            "non e' presente nei dati di mercato."
        )

    returns = log_returns(
        data["INDEX_EUR"]
    )

    returns.name = (
        "log_return_index_eur"
    )

    volatilities = {
        label: historical_volatility(
            returns,
            years,
        )
        for label, years
        in WINDOWS_YEARS.items()
    }

    volatility_table = (
        pd.DataFrame.from_dict(
            volatilities,
            orient="index",
            columns=[
                "volatilita_annualizzata"
            ],
        )
    )

    volatility_table.to_csv(
        par.FILE_VOLATILITA_FINESTRE,
        index_label="finestra",
    )

    rolling = (
        rolling_volatility(
            returns,
            ROLLING_WINDOW_YEARS,
        )
    )

    rolling.name = (
        f"vol_rolling_"
        f"{ROLLING_WINDOW_YEARS}y"
    )

    data = pd.concat(
        [
            data,
            returns,
            rolling,
        ],
        axis=1,
    )

    if "VIX" in data.columns:
        data["VIX_sigma"] = (
            data["VIX"]
            / 100.0
        )

    data.to_csv(
        par.FILE_DATI_VOLATILITA,
        index_label="Date",
    )

    figure_path = (
        par.OUTPUT_DIR
        / "confronto_vol_storica_vix.png"
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        data.index,
        data[rolling.name] * 100,
        label=(
            f"Volatilita' storica rolling "
            f"({ROLLING_WINDOW_YEARS}y)"
        ),
    )

    if "VIX" in data.columns:
        plt.plot(
            data.index,
            data["VIX"],
            label="VIX",
            alpha=0.7,
        )

    plt.xlabel(
        "Data"
    )

    plt.ylabel(
        "Volatilita' (%)"
    )

    plt.title(
        "Volatilita' storica "
        "dell'indice proxy in EUR e VIX"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.show()

    print(
        "\n=== Volatilita' storica "
        "annualizzata ==="
    )

    for (
        label,
        sigma,
    ) in volatilities.items():
        print(
            f"Finestra {label}: "
            f"{sigma:.4f} "
            f"({sigma * 100:.2f}%)"
        )

    print(
        "\nSerie utilizzata: "
        "indice proxy espresso in EUR"
    )

    print(
        "\nFile salvati:"
    )

    print(
        f"  - "
        f"{par.FILE_VOLATILITA_FINESTRE}"
    )

    print(
        f"  - "
        f"{par.FILE_DATI_VOLATILITA}"
    )

    print(
        f"  - "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()
