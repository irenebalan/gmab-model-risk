import pandas as pd
import yfinance as yf

import parametri as par


def download_yahoo_series(
    ticker,
    start,
    end,
    name,
):
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"Nessun dato scaricato per {ticker}"
        )

    series = data["Close"].squeeze()
    series.name = name

    return series


def download_ecb_series(
    series_key,
    start,
    end,
    name,
):
    dataflow, key = (
        series_key.split(".", 1)
    )

    url = (
        f"https://data-api.ecb.europa.eu/service/data/"
        f"{dataflow}/{key}"
        f"?startPeriod={start}&format=csvdata"
    )

    if end is not None:
        url += (
            f"&endPeriod={end}"
        )

    data = pd.read_csv(url)

    if data is None or data.empty:
        raise RuntimeError(
            f"Nessun dato scaricato per "
            f"la serie BCE {series_key}"
        )

    if (
        "TIME_PERIOD" not in data.columns
        or "OBS_VALUE" not in data.columns
    ):
        raise RuntimeError(
            f"Formato inatteso per "
            f"la serie BCE {series_key}"
        )

    series = pd.Series(
        data[
            "OBS_VALUE"
        ].astype(float).values,
        index=pd.to_datetime(
            data["TIME_PERIOD"]
        ),
        name=name,
    )

    series = (
        series[
            ~series.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )

    return series


def download_risk_free_curve(
    start,
    end,
):
    series_list = []

    for tenor in (
        par.RISK_FREE_TENORS_YEARS
    ):
        series_key = (
            par.RISK_FREE_SERIES_TEMPLATE.format(
                tenor=tenor
            )
        )

        series = download_ecb_series(
            series_key,
            start,
            end,
            f"RiskFree_{tenor}Y",
        )

        series = (
            series / 100.0
        )

        series_list.append(
            series
        )

    return pd.concat(
        series_list,
        axis=1,
    ).sort_index()


def main():
    par.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_index_usd = (
        download_yahoo_series(
            par.LONG_INDEX_TICKER,
            par.LONG_INDEX_START_DATE,
            par.END_DATE,
            "INDEX_USD",
        )
    )

    etf_usd = (
        download_yahoo_series(
            par.ETF_TICKER,
            par.ETF_START_DATE,
            par.END_DATE,
            "ETF_USD",
        )
    )

    vix = download_yahoo_series(
        par.VIX_TICKER,
        par.LONG_INDEX_START_DATE,
        par.END_DATE,
        "VIX",
    )

    eurusd = download_ecb_series(
        par.FX_SERIES_KEY,
        par.LONG_INDEX_START_DATE,
        par.END_DATE,
        "EURUSD",
    )

    risk_free = (
        download_risk_free_curve(
            par.LONG_INDEX_START_DATE,
            par.END_DATE,
        )
    )

    market_data = pd.concat(
        [
            long_index_usd,
            etf_usd,
            eurusd,
            vix,
            risk_free,
        ],
        axis=1,
    ).sort_index()

    market_data = (
        market_data.ffill()
    )

    market_data = (
        market_data.reindex(
            long_index_usd.index
        )
    )

    if (
        market_data[
            "EURUSD"
        ].isna().any()
    ):
        raise RuntimeError(
            "Valori mancanti nel cambio "
            "EUR/USD dopo l'allineamento."
        )

    market_data["INDEX_EUR"] = (
        market_data["INDEX_USD"]
        / market_data["EURUSD"]
    )

    market_data["ETF_EUR"] = (
        market_data["ETF_USD"]
        / market_data["EURUSD"]
    )

    fx_data = (
        eurusd.to_frame()
    )

    fx_data.to_csv(
        par.FILE_FX_EURUSD,
        index_label="Date",
    )

    risk_free.to_csv(
        par.FILE_RISK_FREE_RATES,
        index_label="Date",
    )

    market_data.to_csv(
        par.FILE_DATI_MERCATO,
        index_label="Date",
    )

    print(
        "\n=== Dati di mercato ==="
    )

    print(
        f"Indice lungo: "
        f"{par.LONG_INDEX_TICKER}"
    )

    print(
        f"Periodo indice: "
        f"{long_index_usd.index.min().date()} - "
        f"{long_index_usd.index.max().date()}"
    )

    print(
        f"Osservazioni indice: "
        f"{len(long_index_usd)}"
    )

    print(
        f"\nETF: "
        f"{par.ETF_TICKER}"
    )

    print(
        f"Periodo ETF: "
        f"{etf_usd.index.min().date()} - "
        f"{etf_usd.index.max().date()}"
    )

    print(
        f"Osservazioni ETF: "
        f"{len(etf_usd)}"
    )

    print(
        "\nFile salvati:"
    )

    print(
        f"  - {par.FILE_FX_EURUSD}"
    )

    print(
        f"  - {par.FILE_RISK_FREE_RATES}"
    )

    print(
        f"  - {par.FILE_DATI_MERCATO}"
    )


if __name__ == "__main__":
    main()
