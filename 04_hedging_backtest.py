import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import parametri as par


HISTORICAL_WINDOWS_YEARS = {
    "storica_1y": 1,
    "storica_3y": 3,
    "storica_5y": 5,
}

OUTPUT_WINDOWS = par.OUTPUT_DIR / "hedging_gmab_per_finestra.csv"
OUTPUT_SCENARIOS = par.OUTPUT_DIR / "hedging_gmab_sintesi.csv"
OUTPUT_CORRELATIONS = par.OUTPUT_DIR / "hedging_volatility_error_correlation.csv"
OUTPUT_PROXY_DRAG = par.OUTPUT_DIR / "hedging_proxy_drag_sensitivity.csv"
OUTPUT_PLOT = par.OUTPUT_DIR / "hedging_rmse_per_frequenza.png"
OUTPUT_PLOT_VOLATILITY_ERROR = par.OUTPUT_DIR / "hedging_error_vs_volatility_gap.png"
OUTPUT_PLOT_PROXY_DRAG = par.OUTPUT_DIR / "hedging_proxy_drag_sensitivity.png"


def bs_put_price(S, G, r, fee, sigma, T):
    if T <= 0:
        return max(G - S, 0.0)

    if sigma <= 0:
        return max(
            G * np.exp(-r * T) - S * np.exp(-fee * T),
            0.0,
        )

    d1 = (
        np.log(S / G)
        + (r - fee + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return (
        G * np.exp(-r * T) * norm.cdf(-d2)
        - S * np.exp(-fee * T) * norm.cdf(-d1)
    )


def bs_put_delta(S, G, r, fee, sigma, T):
    if T <= 0:
        return -1.0 if S < G else 0.0

    if sigma <= 0:
        future_S = S * np.exp((r - fee) * T)
        if future_S < G:
            return -np.exp(-fee * T)
        return 0.0

    d1 = (
        np.log(S / G)
        + (r - fee + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    return np.exp(-fee * T) * (norm.cdf(d1) - 1.0)


def load_market_data():
    data = pd.read_csv(
        par.FILE_DATI_MERCATO,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()

    risk_free_column = f"RiskFree_{par.GMAB_MATURITY_YEARS}Y"

    required = {
        "INDEX_EUR",
        "VIX",
        risk_free_column,
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Mancano le colonne {sorted(missing)} "
            f"in {par.FILE_DATI_MERCATO}"
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data.loc[
        ~data.index.duplicated(keep="first")
    ]


def log_returns(price_series):
    return np.log(
        price_series / price_series.shift(1)
    ).dropna()


def first_trading_date_on_or_after(index, target_date):
    position = index.searchsorted(
        pd.Timestamp(target_date),
        side="left",
    )

    if position >= len(index):
        return None

    return index[position]


def first_rate_date(market_data):
    column = f"RiskFree_{par.GMAB_MATURITY_YEARS}Y"
    rates = market_data[column].dropna()

    if rates.empty:
        raise RuntimeError(
            "Nessun tasso risk-free decennale disponibile."
        )

    return rates.index[0]


def get_initial_risk_free_rate(market_data, start_date):
    column = f"RiskFree_{par.GMAB_MATURITY_YEARS}Y"

    rates = market_data.loc[
        market_data.index <= start_date,
        column,
    ].dropna()

    if rates.empty:
        raise RuntimeError(
            f"Nessun tasso {par.GMAB_MATURITY_YEARS}Y "
            f"disponibile alla data {start_date.date()}."
        )

    return float(rates.iloc[-1])


def generate_backtest_windows(market_data):
    index_series = market_data["INDEX_EUR"].dropna()
    index = index_series.index
    returns = log_returns(index_series)

    max_lookback_days = (
        max(HISTORICAL_WINDOWS_YEARS.values())
        * par.TRADING_DAYS_PER_YEAR
    )

    if len(returns) < max_lookback_days:
        raise RuntimeError(
            "Dati insufficienti per stimare la volatilita' iniziale."
        )

    last_required_return_date = returns.index[
        max_lookback_days - 1
    ]

    first_volatility_position = index.searchsorted(
        last_required_return_date,
        side="right",
    )

    if first_volatility_position >= len(index):
        raise RuntimeError(
            "Impossibile determinare la prima finestra di backtest."
        )

    first_volatility_date = index[first_volatility_position]

    first_start = max(
        first_volatility_date,
        first_rate_date(market_data),
    )

    first_start = first_trading_date_on_or_after(
        index,
        first_start,
    )

    last_start = (
        index.max()
        - pd.DateOffset(years=par.GMAB_MATURITY_YEARS)
    )

    windows = []
    target_start = first_start

    while target_start <= last_start:
        start_date = first_trading_date_on_or_after(
            index,
            target_start,
        )

        if start_date is None:
            break

        target_end = (
            start_date
            + pd.DateOffset(years=par.GMAB_MATURITY_YEARS)
        )

        end_date = first_trading_date_on_or_after(
            index,
            target_end,
        )

        if end_date is not None and end_date > start_date:
            windows.append((start_date, end_date))

        target_start = (
            start_date
            + pd.DateOffset(years=par.BACKTEST_STEP_YEARS)
        )

    windows = list(dict.fromkeys(windows))

    if not windows:
        raise RuntimeError(
            "Dati insufficienti per costruire le finestre di backtest."
        )

    return windows


def estimate_volatility_scenarios(market_data, start_date):
    index_series = market_data["INDEX_EUR"].dropna()
    returns = log_returns(index_series)
    past_returns = returns.loc[returns.index < start_date]

    scenarios = {}

    for scenario, years in HISTORICAL_WINDOWS_YEARS.items():
        n_days = years * par.TRADING_DAYS_PER_YEAR

        if len(past_returns) < n_days:
            raise RuntimeError(
                f"Dati insufficienti per {scenario} alla data "
                f"{start_date.date()}: {len(past_returns)} osservazioni."
            )

        sample = past_returns.iloc[-n_days:]

        scenarios[scenario] = float(
            sample.std(ddof=1)
            * np.sqrt(par.TRADING_DAYS_PER_YEAR)
        )

    vix = market_data.loc[
        market_data.index <= start_date,
        "VIX",
    ].dropna()

    if not vix.empty:
        scenarios["proxy_vix_30d"] = float(vix.iloc[-1]) / 100.0

    return scenarios


def build_fund_paths(
    market_data,
    start_date,
    end_date,
    proxy_drag,
):
    index_series = market_data.loc[
        start_date:end_date,
        "INDEX_EUR",
    ].dropna()

    if len(index_series) < 2:
        raise RuntimeError(
            f"Percorso indice insufficiente tra "
            f"{start_date.date()} e {end_date.date()}."
        )

    hedge_asset = (
        par.INITIAL_FUND_VALUE
        * index_series
        / index_series.iloc[0]
    )

    elapsed_years = (
        index_series.index - index_series.index[0]
    ).days / 365.25

    effective_fund_drag = (
        get_effective_fund_drag(
            proxy_drag
        )
    )

    fund = (
        hedge_asset
        * np.exp(
            -effective_fund_drag
            * elapsed_years
        )
    )

    return pd.DataFrame(
        {
            "Hedge_EUR": hedge_asset,
            "Fund_EUR": fund,
        }
    )


def get_effective_fund_drag(proxy_drag):
    if np.isclose(proxy_drag, 0.0):
        return par.TOTAL_FEE_ANNUAL

    return (
        par.POLICY_FEE_ANNUAL
        + proxy_drag
    )


def realized_volatility(paths):
    returns = log_returns(paths["Hedge_EUR"])

    if len(returns) < 2:
        raise RuntimeError(
            "Osservazioni insufficienti per la volatilita' realizzata."
        )

    return float(
        returns.std(ddof=1)
        * np.sqrt(par.TRADING_DAYS_PER_YEAR)
    )


def run_single_gmab_hedge(
    paths,
    sigma,
    risk_free,
    rebalance_days,
    transaction_cost_rate,
    proxy_drag,
):
    dates = paths.index
    maturity_date = dates[-1]

    S0 = float(paths["Fund_EUR"].iloc[0])
    X0 = float(paths["Hedge_EUR"].iloc[0])
    T = float(par.GMAB_MATURITY_YEARS)

    effective_fund_drag = (
        get_effective_fund_drag(
            proxy_drag
        )
    )

    initial_price = bs_put_price(
        S0,
        par.G,
        risk_free,
        effective_fund_drag,
        sigma,
        T,
    )

    initial_delta = bs_put_delta(
        S0,
        par.G,
        risk_free,
        effective_fund_drag,
        sigma,
        T,
    )

    hedge_units = initial_delta * S0 / X0

    initial_transaction_cost = (
        abs(hedge_units)
        * X0
        * transaction_cost_rate
    )

    cash_account = (
        initial_price
        - hedge_units * X0
        - initial_transaction_cost
    )

    total_transaction_cost = initial_transaction_cost

    for i in range(1, len(paths)):
        previous_date = dates[i - 1]
        current_date = dates[i]

        elapsed = (
            current_date - previous_date
        ).days / 365.25

        cash_account *= np.exp(
            risk_free * elapsed
        )

        current_S = float(paths["Fund_EUR"].iloc[i])
        current_X = float(paths["Hedge_EUR"].iloc[i])

        portfolio_value = (
            hedge_units * current_X
            + cash_account
        )

        is_rebalance = i % rebalance_days == 0
        is_maturity = i == len(paths) - 1

        if is_rebalance and not is_maturity:
            remaining_time = max(
                (maturity_date - current_date).days / 365.25,
                0.0,
            )

            new_delta = bs_put_delta(
                current_S,
                par.G,
                risk_free,
                effective_fund_drag,
                sigma,
                remaining_time,
            )

            new_hedge_units = (
                new_delta
                * current_S
                / current_X
            )

            traded_units = (
                new_hedge_units
                - hedge_units
            )

            transaction_cost = (
                abs(traded_units)
                * current_X
                * transaction_cost_rate
            )

            total_transaction_cost += transaction_cost

            cash_account = (
                portfolio_value
                - new_hedge_units * current_X
                - transaction_cost
            )

            hedge_units = new_hedge_units

    terminal_fund_value = float(
        paths["Fund_EUR"].iloc[-1]
    )

    terminal_hedge_value = float(
        paths["Hedge_EUR"].iloc[-1]
    )

    portfolio_before_liquidation = (
        hedge_units * terminal_hedge_value
        + cash_account
    )

    liquidation_cost = (
        abs(hedge_units)
        * terminal_hedge_value
        * transaction_cost_rate
    )

    total_transaction_cost += liquidation_cost

    terminal_portfolio_value = (
        portfolio_before_liquidation
        - liquidation_cost
    )

    payoff = max(
        par.G - terminal_fund_value,
        0.0,
    )

    hedging_error = (
        terminal_portfolio_value
        - payoff
    )

    return {
        "data_scadenza": maturity_date,
        "prezzo_gmab_iniziale": initial_price,
        "delta_iniziale": initial_delta,
        "tasso_iniziale_fisso": risk_free,
        "drag_fondo_effettivo": effective_fund_drag,
        "valore_fondo_scadenza": terminal_fund_value,
        "valore_hedge_scadenza": terminal_hedge_value,
        "valore_copertura_scadenza": terminal_portfolio_value,
        "payoff_gmab": payoff,
        "costi_transazione_totali": total_transaction_cost,
        "hedging_error": hedging_error,
        "deficit": max(-hedging_error, 0.0),
        "surplus": max(hedging_error, 0.0),
    }


def aggregate_results(results):
    data = results.copy()

    data["errore_assoluto"] = data["hedging_error"].abs()
    data["errore_quadrato"] = data["hedging_error"] ** 2
    data["in_deficit"] = data["hedging_error"] < 0

    summary = (
        data.groupby(
            [
                "scenario_drag_proxy",
                "proxy_drag",
                "scenario",
                "ribilanciamento",
                "rebalance_days",
                "costo_transazione",
            ],
            as_index=False,
        )
        .agg(
            numero_finestre=("data_inizio", "count"),
            sigma_media=("sigma", "mean"),
            sigma_min=("sigma", "min"),
            sigma_max=("sigma", "max"),
            vol_realizzata_media=("vol_realizzata", "mean"),
            errore_stima_vol_medio=("errore_stima_vol", "mean"),
            prezzo_gmab_medio=("prezzo_gmab_iniziale", "mean"),
            errore_medio=("hedging_error", "mean"),
            mae=("errore_assoluto", "mean"),
            mse=("errore_quadrato", "mean"),
            deficit_medio=("deficit", "mean"),
            surplus_medio=("surplus", "mean"),
            frequenza_deficit=("in_deficit", "mean"),
            costo_transazione_medio=("costi_transazione_totali", "mean"),
            peggior_errore=("hedging_error", "min"),
            miglior_errore=("hedging_error", "max"),
        )
    )

    summary["rmse"] = np.sqrt(summary["mse"])
    summary["frequenza_deficit_pct"] = (
        summary["frequenza_deficit"] * 100
    )

    summary = summary.drop(
        columns=["mse", "frequenza_deficit"]
    )

    return (
        summary.sort_values(
            [
                "scenario_drag_proxy",
                "scenario",
                "costo_transazione",
                "rebalance_days",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_correlations(results):
    rows = []

    grouped = results.groupby(
        [
            "scenario_drag_proxy",
            "proxy_drag",
            "scenario",
            "ribilanciamento",
            "rebalance_days",
            "costo_transazione",
        ]
    )

    for keys, group in grouped:
        (
            drag_name,
            proxy_drag,
            scenario,
            rebalance_name,
            rebalance_days,
            transaction_cost_rate,
        ) = keys

        valid = group[
            ["errore_stima_vol", "hedging_error"]
        ].dropna()

        if (
            len(valid) >= 2
            and valid["errore_stima_vol"].std(ddof=1) > 0
            and valid["hedging_error"].std(ddof=1) > 0
        ):
            correlation = float(
                valid["errore_stima_vol"].corr(
                    valid["hedging_error"]
                )
            )
        else:
            correlation = np.nan

        rows.append(
            {
                "scenario_drag_proxy": drag_name,
                "proxy_drag": proxy_drag,
                "scenario": scenario,
                "ribilanciamento": rebalance_name,
                "rebalance_days": rebalance_days,
                "costo_transazione": transaction_cost_rate,
                "numero_finestre": len(valid),
                "correlazione_gap_vol_errore_hedging": correlation,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "scenario_drag_proxy",
                "scenario",
                "costo_transazione",
                "rebalance_days",
            ]
        )
        .reset_index(drop=True)
    )


def create_rmse_plot(summary):
    subset = summary.loc[
        (summary["scenario_drag_proxy"] == "senza_drag_proxy")
        & np.isclose(
            summary["costo_transazione"],
            par.TRANSACTION_COST_RATE,
        )
    ].copy()

    plt.figure(figsize=(11, 6))

    for scenario in subset["scenario"].unique():
        scenario_data = subset.loc[
            subset["scenario"] == scenario
        ].sort_values("rebalance_days")

        plt.plot(
            scenario_data["ribilanciamento"],
            scenario_data["rmse"],
            marker="o",
            label=scenario,
        )

    plt.xlabel("Frequenza di ribilanciamento")
    plt.ylabel("RMSE dell'errore di copertura")
    plt.title("GMAB: errore di copertura per frequenza")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.show()


def create_volatility_error_plot(results):
    subset = results.loc[
        (results["scenario_drag_proxy"] == "senza_drag_proxy")
        & (results["ribilanciamento"] == par.BASE_REBALANCE_FREQUENCY)
        & np.isclose(
            results["costo_transazione"],
            par.TRANSACTION_COST_RATE,
        )
    ].copy()

    plt.figure(figsize=(11, 6))

    for scenario in subset["scenario"].unique():
        scenario_data = subset.loc[
            subset["scenario"] == scenario
        ]

        correlation = scenario_data["errore_stima_vol"].corr(
            scenario_data["hedging_error"]
        )

        plt.scatter(
            scenario_data["errore_stima_vol"] * 100,
            scenario_data["hedging_error"],
            label=f"{scenario} (rho={correlation:.2f})",
        )

    plt.axhline(0.0, linestyle="--")
    plt.axvline(0.0, linestyle="--")
    plt.xlabel(
        "Volatilita' realizzata - volatilita' stimata "
        "(punti percentuali)"
    )
    plt.ylabel("Errore di copertura")
    plt.title("Errore di stima della volatilita' e hedging error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_VOLATILITY_ERROR, dpi=150)
    plt.show()


def build_proxy_drag_sensitivity(summary):
    sensitivity = summary.loc[
        (summary["ribilanciamento"] == par.BASE_REBALANCE_FREQUENCY)
        & np.isclose(
            summary["costo_transazione"],
            par.TRANSACTION_COST_RATE,
        )
    ][
        [
            "scenario_drag_proxy",
            "proxy_drag",
            "scenario",
            "prezzo_gmab_medio",
            "errore_medio",
            "mae",
            "rmse",
            "deficit_medio",
            "surplus_medio",
            "frequenza_deficit_pct",
        ]
    ].copy()

    return sensitivity.sort_values(
        ["scenario", "proxy_drag"]
    ).reset_index(drop=True)


def create_proxy_drag_plot(sensitivity):
    plt.figure(figsize=(11, 6))

    for drag_name in sensitivity["scenario_drag_proxy"].unique():
        scenario_data = sensitivity.loc[
            sensitivity["scenario_drag_proxy"] == drag_name
        ]

        plt.plot(
            scenario_data["scenario"],
            scenario_data["rmse"],
            marker="o",
            label=drag_name,
        )

    plt.xlabel("Scenario di volatilita'")
    plt.ylabel("RMSE dell'errore di copertura")
    plt.title("Sensitivita' al drag della proxy")
    plt.xticks(rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PROXY_DRAG, dpi=150)
    plt.show()


def main():
    par.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    market_data = load_market_data()
    windows = generate_backtest_windows(market_data)

    print("=== Configurazione GMAB ===")
    print(f"Proxy storica: {par.LONG_INDEX_TICKER}")
    print(f"Valuta: {par.CONTRACT_CURRENCY}")
    print(f"Valore iniziale fondo: {par.INITIAL_FUND_VALUE:.4f}")
    print(f"Capitale garantito: {par.G:.4f}")
    print(f"Scadenza: {par.GMAB_MATURITY_YEARS} anni")
    print(f"Fee totale modello: {par.TOTAL_FEE_ANNUAL:.4%}")
    print(
        "Tasso di pricing e cash: "
        f"RiskFree_{par.GMAB_MATURITY_YEARS}Y "
        "osservato all'inizio di ogni finestra e mantenuto fisso"
    )
    print(
        f"Drag proxy continuo di sensitivita': "
        f"{par.PROXY_DRAG_CONTINUOUS:.4%}"
    )
    print(
        "Scenario senza drag: "
        f"TER + fee polizza = {par.TOTAL_FEE_ANNUAL:.4%}"
    )
    print(
        "Scenario con drag: "
        f"fee polizza + drag empirico = "
        f"{get_effective_fund_drag(par.PROXY_DRAG_CONTINUOUS):.4%}"
    )
    print(f"Numero di finestre: {len(windows)}")

    results = []

    for start_date, end_date in windows:
        print(
            f"\n=== Finestra {start_date.date()} - "
            f"{end_date.date()} ==="
        )

        risk_free = get_initial_risk_free_rate(
            market_data,
            start_date,
        )

        scenarios = estimate_volatility_scenarios(
            market_data,
            start_date,
        )

        for drag_name, proxy_drag in par.PROXY_DRAG_SCENARIOS.items():
            paths = build_fund_paths(
                market_data,
                start_date,
                end_date,
                proxy_drag,
            )

            realized_sigma = realized_volatility(paths)

            for scenario, sigma in scenarios.items():
                volatility_error = realized_sigma - sigma

                for rebalance_name, rebalance_days in (
                    par.REBALANCE_FREQUENCIES.items()
                ):
                    for transaction_cost_rate in (
                        par.TRANSACTION_COST_SCENARIOS
                    ):
                        hedge = run_single_gmab_hedge(
                            paths,
                            sigma,
                            risk_free,
                            rebalance_days,
                            transaction_cost_rate,
                            proxy_drag,
                        )

                        results.append(
                            {
                                "data_inizio": start_date,
                                "data_fine": end_date,
                                "scenario_drag_proxy": drag_name,
                                "proxy_drag": proxy_drag,
                                "scenario": scenario,
                                "sigma": sigma,
                                "vol_realizzata": realized_sigma,
                                "errore_stima_vol": volatility_error,
                                "ribilanciamento": rebalance_name,
                                "rebalance_days": rebalance_days,
                                "costo_transazione": transaction_cost_rate,
                                **hedge,
                            }
                        )

    results_table = pd.DataFrame(results)
    summary = aggregate_results(results_table)
    correlations = calculate_correlations(results_table)
    proxy_drag_sensitivity = build_proxy_drag_sensitivity(summary)

    results_table.to_csv(
        OUTPUT_WINDOWS,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary.to_csv(
        OUTPUT_SCENARIOS,
        index=False,
    )

    correlations.to_csv(
        OUTPUT_CORRELATIONS,
        index=False,
    )

    proxy_drag_sensitivity.to_csv(
        OUTPUT_PROXY_DRAG,
        index=False,
    )

    print("\n=== Diagnostica delle finestre ===")

    for drag_name in par.PROXY_DRAG_SCENARIOS:
        window_payoffs = (
            results_table.loc[
                results_table["scenario_drag_proxy"] == drag_name,
                ["data_inizio", "data_fine", "payoff_gmab"],
            ]
            .drop_duplicates(
                subset=["data_inizio", "data_fine"]
            )
        )

        positive_payoffs = int(
            (window_payoffs["payoff_gmab"] > 0).sum()
        )

        print(f"\nScenario: {drag_name}")
        print(f"Finestre totali: {len(window_payoffs)}")
        print(
            f"Finestre con payoff GMAB > 0: "
            f"{positive_payoffs}"
        )
        print(
            f"Percentuale con evento: "
            f"{positive_payoffs / len(window_payoffs):.2%}"
        )

        if positive_payoffs == 0:
            print(
                "Il backtest storico non contiene eventi di "
                "attivazione della garanzia."
            )

    print("\n=== Sintesi sensitivita' drag proxy ===")
    print(
        proxy_drag_sensitivity.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(
        "\n=== Correlazione fra errore di stima "
        "della volatilita' e hedging error "
        "(caso base senza drag) ==="
    )

    base_correlations = correlations.loc[
        (correlations["scenario_drag_proxy"] == "senza_drag_proxy")
        & (
            correlations["ribilanciamento"]
            == par.BASE_REBALANCE_FREQUENCY
        )
        & np.isclose(
            correlations["costo_transazione"],
            par.TRANSACTION_COST_RATE,
        )
    ]

    print(
        base_correlations.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    create_rmse_plot(summary)
    create_volatility_error_plot(results_table)
    create_proxy_drag_plot(proxy_drag_sensitivity)

    print("\nFile salvati:")
    print(f"  - {OUTPUT_WINDOWS}")
    print(f"  - {OUTPUT_SCENARIOS}")
    print(f"  - {OUTPUT_CORRELATIONS}")
    print(f"  - {OUTPUT_PROXY_DRAG}")
    print(f"  - {OUTPUT_PLOT}")
    print(f"  - {OUTPUT_PLOT_VOLATILITY_ERROR}")
    print(f"  - {OUTPUT_PLOT_PROXY_DRAG}")


if __name__ == "__main__":
    main()
