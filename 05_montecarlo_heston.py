import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import parametri as par


N_PATHS = 10000
SEED = 12345

VARIANCE_PROXY_WINDOW_DAYS = 126
VARIANCE_SAMPLING_DAYS = (
    par.REBALANCE_FREQUENCIES[
        "mensile"
    ]
)

VARIANCE_RISK_PREMIUM_LAMBDA = 0.0

INTEGRATION_MAX = 100.0
INTEGRATION_POINTS = 250
INTEGRATION_BATCH_SIZE = 500

OUTPUT_PATHS = (
    par.OUTPUT_DIR
    / "heston_model_risk_per_path.csv"
)

OUTPUT_SUMMARY = (
    par.OUTPUT_DIR
    / "heston_model_risk_sintesi.csv"
)

OUTPUT_PARAMETERS = (
    par.OUTPUT_DIR
    / "heston_parametri_stimati.csv"
)

OUTPUT_VARIANCE_REGRESSION = (
    par.OUTPUT_DIR
    / "heston_variance_regression.csv"
)

OUTPUT_PLOT = (
    par.OUTPUT_DIR
    / "heston_model_risk_rmse.png"
)

OUTPUT_DISTRIBUTION_PLOT = (
    par.OUTPUT_DIR
    / "heston_hedging_error_distribution.png"
)

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
        raise RuntimeError(
            f"Dati insufficienti per "
            f"la volatilita' {years}Y."
        )

    return float(
        returns.iloc[-n_days:]
        .std(ddof=1)
        * np.sqrt(
            par.TRADING_DAYS_PER_YEAR
        )
    )


def load_market_data():
    data = pd.read_csv(
        par.FILE_DATI_MERCATO,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()

    required = {
        "INDEX_EUR",
        f"RiskFree_{par.GMAB_MATURITY_YEARS}Y",
    }

    missing = required.difference(
        data.columns
    )

    if missing:
        raise RuntimeError(
            f"Mancano le colonne "
            f"{sorted(missing)}."
        )

    data["INDEX_EUR"] = pd.to_numeric(
        data["INDEX_EUR"],
        errors="coerce",
    )

    return data


def estimate_heston_parameters(
    index_series,
):
    returns = log_returns(
        index_series
    )

    realized_variance = (
        returns
        .rolling(
            VARIANCE_PROXY_WINDOW_DAYS
        )
        .var(ddof=1)
        * par.TRADING_DAYS_PER_YEAR
    ).dropna()

    if len(realized_variance) < 100:
        raise RuntimeError(
            "Dati insufficienti per "
            "stimare i parametri Heston."
        )

    variance_monthly = (
        realized_variance.iloc[
            ::VARIANCE_SAMPLING_DAYS
        ]
    )

    v_t = (
        variance_monthly.iloc[:-1]
        .to_numpy()
    )

    v_next = (
        variance_monthly.iloc[1:]
        .to_numpy()
    )

    X = np.column_stack(
        [
            np.ones(len(v_t)),
            v_t,
        ]
    )

    intercept, persistence = (
        np.linalg.lstsq(
            X,
            v_next,
            rcond=None,
        )[0]
    )

    dt = (
        VARIANCE_SAMPLING_DAYS
        / par.TRADING_DAYS_PER_YEAR
    )

    if not (
        0.0
        < persistence
        < 1.0
    ):
        raise RuntimeError(
            "Persistenza della varianza "
            "non compatibile con la "
            "stima Heston."
        )

    kappa = (
        -np.log(
            persistence
        )
        / dt
    )

    theta = (
        intercept
        / (
            1.0
            - persistence
        )
    )

    residuals = (
        v_next
        - (
            intercept
            + persistence * v_t
        )
    )

    variance_scale = np.sqrt(
        np.maximum(
            v_t,
            1e-10,
        )
        * dt
    )

    xi = float(
        np.std(
            residuals
            / variance_scale,
            ddof=1,
        )
    )

    sampled_prices = (
        index_series.reindex(
            variance_monthly.index
        )
    )

    monthly_returns = np.log(
        sampled_prices.to_numpy()[1:]
        / sampled_prices.to_numpy()[:-1]
    )

    return_shocks = (
        monthly_returns
        - monthly_returns.mean()
    ) / variance_scale

    variance_shocks = (
        residuals
        / (
            xi
            * variance_scale
        )
    )

    rho = float(
        np.corrcoef(
            return_shocks,
            variance_shocks,
        )[0, 1]
    )

    rho = float(
        np.clip(
            rho,
            -0.95,
            0.95,
        )
    )

    v0 = float(
        realized_variance.iloc[-1]
    )

    if (
        theta <= 0.0
        or xi <= 0.0
        or v0 <= 0.0
    ):
        raise RuntimeError(
            "Parametri Heston non validi."
        )

    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "xi": float(xi),
        "rho": rho,
        "v0": v0,
    }


def build_risk_neutral_parameters(
    historical_parameters,
    lambda_variance,
):
    kappa_p = historical_parameters["kappa"]
    theta_p = historical_parameters["theta"]

    kappa_q = (
        kappa_p
        + lambda_variance
    )

    if kappa_q <= 0.0:
        raise RuntimeError(
            "kappa sotto Q deve essere positivo."
        )

    theta_q = (
        kappa_p
        * theta_p
        / kappa_q
    )

    return {
        **historical_parameters,
        "kappa": float(kappa_q),
        "theta": float(theta_q),
    }


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
        return np.where(
            np.asarray(S) < G,
            -1.0,
            0.0,
        )

    if sigma <= 0:
        future_S = (
            np.asarray(S)
            * np.exp(
                (r - fee) * T
            )
        )

        return np.where(
            future_S < G,
            -np.exp(-fee * T),
            0.0,
        )

    d1 = (
        np.log(
            np.asarray(S) / G
        )
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


def find_equivalent_bs_sigma(
    target_price,
    r,
):
    def objective(sigma):
        return (
            bs_put_price(
                par.INITIAL_FUND_VALUE,
                par.G,
                r,
                par.TOTAL_FEE_ANNUAL,
                sigma,
                par.GMAB_MATURITY_YEARS,
            )
            - target_price
        )

    lower = 1e-6
    upper = 3.0

    lower_value = objective(
        lower
    )

    upper_value = objective(
        upper
    )

    if (
        lower_value
        * upper_value
        > 0
    ):
        raise RuntimeError(
            "Impossibile trovare la "
            "volatilita' BS equivalente."
        )

    return float(
        brentq(
            objective,
            lower,
            upper,
        )
    )


def heston_cd(
    u,
    T,
    r,
    fee,
    kappa,
    theta,
    xi,
    rho,
):
    iu = 1j * u

    d = np.sqrt(
        (
            kappa
            - rho * xi * iu
        ) ** 2
        + xi ** 2
        * (
            iu
            + u ** 2
        )
    )

    d = np.where(
        np.real(d) < 0,
        -d,
        d,
    )

    g = (
        kappa
        - rho * xi * iu
        - d
    ) / (
        kappa
        - rho * xi * iu
        + d
    )

    exp_dt = np.exp(
        -d * T
    )

    C = (
        iu
        * (r - fee)
        * T
        + (
            kappa
            * theta
            / xi ** 2
        )
        * (
            (
                kappa
                - rho * xi * iu
                - d
            )
            * T
            - 2.0
            * np.log(
                (
                    1.0
                    - g * exp_dt
                )
                / (
                    1.0
                    - g
                )
            )
        )
    )

    D = (
        (
            kappa
            - rho * xi * iu
            - d
        )
        / xi ** 2
        * (
            1.0
            - exp_dt
        )
        / (
            1.0
            - g * exp_dt
        )
    )

    return C, D


def heston_put_price_delta(
    S,
    G,
    v,
    r,
    fee,
    T,
    parameters,
):
    if T <= 0:
        price = max(
            G - S,
            0.0,
        )

        delta = (
            -1.0
            if S < G
            else 0.0
        )

        return price, delta

    kappa = parameters["kappa"]
    theta = parameters["theta"]
    xi = parameters["xi"]
    rho = parameters["rho"]

    u = np.linspace(
        1e-5,
        INTEGRATION_MAX,
        INTEGRATION_POINTS,
    )

    C2, D2 = heston_cd(
        u,
        T,
        r,
        fee,
        kappa,
        theta,
        xi,
        rho,
    )

    phi2 = np.exp(
        1j
        * u
        * np.log(S)
        + C2
        + D2 * v
    )

    integrand2 = np.real(
        np.exp(
            -1j
            * u
            * np.log(G)
        )
        * phi2
        / (
            1j * u
        )
    )

    P2 = (
        0.5
        + np.trapezoid(
            integrand2,
            u,
        )
        / np.pi
    )

    shifted_u = (
        u - 1j
    )

    C1, D1 = heston_cd(
        shifted_u,
        T,
        r,
        fee,
        kappa,
        theta,
        xi,
        rho,
    )

    phi1 = np.exp(
        1j
        * shifted_u
        * np.log(S)
        + C1
        + D1 * v
    )

    denominator = (
        S
        * np.exp(
            (r - fee) * T
        )
    )

    integrand1 = np.real(
        np.exp(
            -1j
            * u
            * np.log(G)
        )
        * phi1
        / (
            1j
            * u
            * denominator
        )
    )

    P1 = (
        0.5
        + np.trapezoid(
            integrand1,
            u,
        )
        / np.pi
    )

    call = (
        S
        * np.exp(-fee * T)
        * P1
        - G
        * np.exp(-r * T)
        * P2
    )

    put = (
        call
        - S * np.exp(-fee * T)
        + G * np.exp(-r * T)
    )

    delta = (
        np.exp(-fee * T)
        * (
            P1 - 1.0
        )
    )

    return (
        float(put),
        float(delta),
    )


def heston_put_delta_array(
    S,
    v,
    G,
    r,
    fee,
    T,
    parameters,
):
    S = np.asarray(
        S,
        dtype=float,
    )

    v = np.asarray(
        v,
        dtype=float,
    )

    if T <= 0:
        return np.where(
            S < G,
            -1.0,
            0.0,
        )

    kappa = parameters["kappa"]
    theta = parameters["theta"]
    xi = parameters["xi"]
    rho = parameters["rho"]

    u = np.linspace(
        1e-5,
        INTEGRATION_MAX,
        INTEGRATION_POINTS,
    )

    shifted_u = (
        u - 1j
    )

    C, D = heston_cd(
        shifted_u,
        T,
        r,
        fee,
        kappa,
        theta,
        xi,
        rho,
    )

    deltas = np.empty(
        len(S)
    )

    for start in range(
        0,
        len(S),
        INTEGRATION_BATCH_SIZE,
    ):
        end = min(
            start
            + INTEGRATION_BATCH_SIZE,
            len(S),
        )

        S_batch = (
            S[start:end]
        )

        v_batch = (
            v[start:end]
        )

        phi = np.exp(
            1j
            * shifted_u[None, :]
            * np.log(
                S_batch[:, None]
            )
            + C[None, :]
            + D[None, :]
            * v_batch[:, None]
        )

        denominator = (
            S_batch[:, None]
            * np.exp(
                (r - fee) * T
            )
        )

        integrand = np.real(
            np.exp(
                -1j
                * u[None, :]
                * np.log(G)
            )
            * phi
            / (
                1j
                * u[None, :]
                * denominator
            )
        )

        P1 = (
            0.5
            + np.trapezoid(
                integrand,
                u,
                axis=1,
            )
            / np.pi
        )

        delta = (
            np.exp(-fee * T)
            * (
                P1 - 1.0
            )
        )

        deltas[start:end] = (
            np.clip(
                delta,
                -np.exp(
                    -fee * T
                ),
                0.0,
            )
        )

    return deltas


def simulate_heston_paths(
    parameters,
    r,
):
    n_steps = (
        par.GMAB_MATURITY_YEARS
        * par.TRADING_DAYS_PER_YEAR
    )

    rebalance_days = (
        par.REBALANCE_FREQUENCIES[
            par.BASE_REBALANCE_FREQUENCY
        ]
    )

    save_steps = list(
        range(
            0,
            n_steps + 1,
            rebalance_days,
        )
    )

    if save_steps[-1] != n_steps:
        save_steps.append(
            n_steps
        )

    n_saved = len(
        save_steps
    )

    rng = np.random.default_rng(
        SEED
    )

    hedge_asset = np.full(
        N_PATHS,
        par.INITIAL_FUND_VALUE,
        dtype=float,
    )

    variance = np.full(
        N_PATHS,
        parameters["v0"],
        dtype=float,
    )

    integrated_variance = np.zeros(
        N_PATHS,
        dtype=float,
    )

    hedge_paths = np.empty(
        (
            N_PATHS,
            n_saved,
        )
    )

    fund_paths = np.empty(
        (
            N_PATHS,
            n_saved,
        )
    )

    variance_paths = np.empty(
        (
            N_PATHS,
            n_saved,
        )
    )

    hedge_paths[:, 0] = (
        hedge_asset
    )

    fund_paths[:, 0] = (
        hedge_asset
    )

    variance_paths[:, 0] = (
        variance
    )

    kappa = parameters["kappa"]
    theta = parameters["theta"]
    xi = parameters["xi"]
    rho = parameters["rho"]

    dt = (
        1.0
        / par.TRADING_DAYS_PER_YEAR
    )

    save_position = 1

    for step in range(
        1,
        n_steps + 1,
    ):
        z1 = rng.standard_normal(
            N_PATHS
        )

        z2 = rng.standard_normal(
            N_PATHS
        )

        z_variance = (
            rho * z1
            + np.sqrt(
                1.0 - rho ** 2
            )
            * z2
        )

        positive_variance = np.maximum(
            variance,
            0.0,
        )

        integrated_variance += (
            positive_variance
            * dt
        )

        hedge_asset *= np.exp(
            (
                r
                - 0.5
                * positive_variance
            )
            * dt
            + np.sqrt(
                positive_variance
                * dt
            )
            * z1
        )

        variance = np.maximum(
            variance
            + kappa
            * (
                theta
                - positive_variance
            )
            * dt
            + xi
            * np.sqrt(
                positive_variance
                * dt
            )
            * z_variance,
            1e-10,
        )

        if (
            save_position
            < n_saved
            and step
            == save_steps[
                save_position
            ]
        ):
            elapsed_years = (
                step
                / par.TRADING_DAYS_PER_YEAR
            )

            hedge_paths[
                :,
                save_position,
            ] = hedge_asset

            fund_paths[
                :,
                save_position,
            ] = (
                hedge_asset
                * np.exp(
                    -par.TOTAL_FEE_ANNUAL
                    * elapsed_years
                )
            )

            variance_paths[
                :,
                save_position,
            ] = variance

            save_position += 1

    times = (
        np.asarray(
            save_steps,
            dtype=float,
        )
        / par.TRADING_DAYS_PER_YEAR
    )

    average_realized_variance = (
        integrated_variance
        / float(
            par.GMAB_MATURITY_YEARS
        )
    )

    return (
        hedge_paths,
        fund_paths,
        variance_paths,
        times,
        average_realized_variance,
    )


def run_bs_hedge(
    hedge_paths,
    fund_paths,
    times,
    sigma,
    r,
    initial_capital=None,
):
    n_paths = (
        hedge_paths.shape[0]
    )

    T = float(
        par.GMAB_MATURITY_YEARS
    )

    model_price = bs_put_price(
        par.INITIAL_FUND_VALUE,
        par.G,
        r,
        par.TOTAL_FEE_ANNUAL,
        sigma,
        T,
    )

    initial_delta = bs_put_delta(
        par.INITIAL_FUND_VALUE,
        par.G,
        r,
        par.TOTAL_FEE_ANNUAL,
        sigma,
        T,
    )

    hedge_units = np.full(
        n_paths,
        initial_delta,
    )

    if initial_capital is None:
        initial_capital = (
            model_price
        )

    cash = (
        initial_capital
        - hedge_units
        * hedge_paths[:, 0]
    )

    for j in range(
        1,
        len(times) - 1,
    ):
        dt = (
            times[j]
            - times[j - 1]
        )

        cash *= np.exp(
            r * dt
        )

        portfolio_value = (
            hedge_units
            * hedge_paths[:, j]
            + cash
        )

        remaining_time = (
            T - times[j]
        )

        delta = bs_put_delta(
            fund_paths[:, j],
            par.G,
            r,
            par.TOTAL_FEE_ANNUAL,
            sigma,
            remaining_time,
        )

        new_units = (
            delta
            * fund_paths[:, j]
            / hedge_paths[:, j]
        )

        cash = (
            portfolio_value
            - new_units
            * hedge_paths[:, j]
        )

        hedge_units = (
            new_units
        )

    final_dt = (
        times[-1]
        - times[-2]
    )

    cash *= np.exp(
        r * final_dt
    )

    terminal_value = (
        hedge_units
        * hedge_paths[:, -1]
        + cash
    )

    payoff = np.maximum(
        par.G
        - fund_paths[:, -1],
        0.0,
    )

    error = (
        terminal_value
        - payoff
    )

    return (
        model_price,
        error,
    )


def run_heston_hedge(
    hedge_paths,
    fund_paths,
    variance_paths,
    times,
    parameters,
    r,
):
    n_paths = (
        hedge_paths.shape[0]
    )

    T = float(
        par.GMAB_MATURITY_YEARS
    )

    (
        initial_price,
        initial_delta,
    ) = heston_put_price_delta(
        par.INITIAL_FUND_VALUE,
        par.G,
        parameters["v0"],
        r,
        par.TOTAL_FEE_ANNUAL,
        T,
        parameters,
    )

    hedge_units = np.full(
        n_paths,
        initial_delta,
    )

    cash = (
        initial_price
        - hedge_units
        * hedge_paths[:, 0]
    )

    for j in range(
        1,
        len(times) - 1,
    ):
        dt = (
            times[j]
            - times[j - 1]
        )

        cash *= np.exp(
            r * dt
        )

        portfolio_value = (
            hedge_units
            * hedge_paths[:, j]
            + cash
        )

        remaining_time = (
            T - times[j]
        )

        delta = (
            heston_put_delta_array(
                fund_paths[:, j],
                variance_paths[:, j],
                par.G,
                r,
                par.TOTAL_FEE_ANNUAL,
                remaining_time,
                parameters,
            )
        )

        new_units = (
            delta
            * fund_paths[:, j]
            / hedge_paths[:, j]
        )

        cash = (
            portfolio_value
            - new_units
            * hedge_paths[:, j]
        )

        hedge_units = (
            new_units
        )

    final_dt = (
        times[-1]
        - times[-2]
    )

    cash *= np.exp(
        r * final_dt
    )

    terminal_value = (
        hedge_units
        * hedge_paths[:, -1]
        + cash
    )

    payoff = np.maximum(
        par.G
        - fund_paths[:, -1],
        0.0,
    )

    error = (
        terminal_value
        - payoff
    )

    return (
        initial_price,
        error,
    )

def heston_variance_regression(
    realized_variance,
    heston_error,
):
    x = np.asarray(
        realized_variance,
        dtype=float,
    )

    y = np.asarray(
        heston_error,
        dtype=float,
    )

    if len(x) != len(y):
        raise ValueError(
            "Varianza realizzata ed hedging error "
            "devono avere la stessa lunghezza."
        )

    if len(x) < 2:
        raise ValueError(
            "Osservazioni insufficienti "
            "per la regressione."
        )

    design_matrix = np.column_stack(
        (
            np.ones(
                len(x),
                dtype=float,
            ),
            x,
        )
    )

    coefficients, _, _, _ = (
        np.linalg.lstsq(
            design_matrix,
            y,
            rcond=None,
        )
    )

    alpha = float(
        coefficients[0]
    )

    beta = float(
        coefficients[1]
    )

    fitted = (
        design_matrix
        @ coefficients
    )

    residuals = (
        y - fitted
    )

    ss_residual = float(
        np.sum(
            residuals ** 2
        )
    )

    ss_total = float(
        np.sum(
            (
                y - y.mean()
            ) ** 2
        )
    )

    if ss_total <= 0:
        raise ValueError(
            "La varianza dell'hedging error "
            "e' nulla: R^2 non definito."
        )

    r_squared = (
        1.0
        - ss_residual
        / ss_total
    )

    correlation = float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )

    regression_table = pd.DataFrame(
        [
            {
                "n_percorsi": len(x),
                "varianza_media_realizzata_media": (
                    x.mean()
                ),
                "varianza_media_realizzata_std": (
                    x.std(
                        ddof=1
                    )
                ),
                "hedging_error_medio": (
                    y.mean()
                ),
                "alpha": alpha,
                "beta": beta,
                "r_squared": r_squared,
                "correlazione": correlation,
            }
        ]
    )

    return regression_table

def summarize_results(
    results,
):
    data = pd.DataFrame(
        results
    )

    data[
        "errore_totale_assoluto"
    ] = (
        data[
            "hedging_error_totale"
        ].abs()
    )

    data[
        "errore_totale_quadrato"
    ] = (
        data[
            "hedging_error_totale"
        ] ** 2
    )

    data[
        "errore_hedging_assoluto"
    ] = (
        data[
            "hedging_error_capitale_equo"
        ].abs()
    )

    data[
        "errore_hedging_quadrato"
    ] = (
        data[
            "hedging_error_capitale_equo"
        ] ** 2
    )

    data[
        "deficit_totale"
    ] = (
        data[
            "hedging_error_totale"
        ] < 0
    )

    data[
        "deficit_hedging"
    ] = (
        data[
            "hedging_error_capitale_equo"
        ] < 0
    )

    summary = (
        data.groupby(
            "strategia",
            as_index=False,
        )
        .agg(
            sigma_bs=(
                "sigma_bs",
                "first",
            ),
            prezzo_iniziale=(
                "prezzo_iniziale",
                "first",
            ),
            pricing_error_t0=(
                "pricing_error_t0",
                "first",
            ),
            pricing_error_scadenza=(
                "pricing_error_scadenza",
                "first",
            ),
            errore_totale_medio=(
                "hedging_error_totale",
                "mean",
            ),
            mae_totale=(
                "errore_totale_assoluto",
                "mean",
            ),
            mse_totale=(
                "errore_totale_quadrato",
                "mean",
            ),
            errore_hedging_medio=(
                "hedging_error_capitale_equo",
                "mean",
            ),
            mae_hedging=(
                "errore_hedging_assoluto",
                "mean",
            ),
            mse_hedging=(
                "errore_hedging_quadrato",
                "mean",
            ),
            frequenza_deficit_totale=(
                "deficit_totale",
                "mean",
            ),
            frequenza_deficit_hedging=(
                "deficit_hedging",
                "mean",
            ),
            peggior_errore_totale=(
                "hedging_error_totale",
                "min",
            ),
            miglior_errore_totale=(
                "hedging_error_totale",
                "max",
            ),
        )
    )

    summary[
        "rmse_totale"
    ] = np.sqrt(
        summary[
            "mse_totale"
        ]
    )

    summary[
        "rmse_hedging"
    ] = np.sqrt(
        summary[
            "mse_hedging"
        ]
    )

    summary[
        "frequenza_deficit_totale_pct"
    ] = (
        summary[
            "frequenza_deficit_totale"
        ]
        * 100
    )

    summary[
        "frequenza_deficit_hedging_pct"
    ] = (
        summary[
            "frequenza_deficit_hedging"
        ]
        * 100
    )

    summary = summary.drop(
        columns=[
            "mse_totale",
            "mse_hedging",
            "frequenza_deficit_totale",
            "frequenza_deficit_hedging",
        ]
    )

    order = {
        "Heston": 0,
        "BS_sigma_equivalente": 1,
        "BS_storica_1y": 2,
        "BS_storica_3y": 3,
        "BS_storica_5y": 4,
    }

    summary[
        "ordine"
    ] = (
        summary[
            "strategia"
        ]
        .map(order)
        .fillna(99)
    )

    return (
        summary
        .sort_values(
            "ordine"
        )
        .drop(
            columns="ordine"
        )
        .reset_index(
            drop=True
        )
    )


def main():
    data = load_market_data()

    index_series = (
        data[
            "INDEX_EUR"
        ].dropna()
    )

    returns = log_returns(
        index_series
    )

    parameters_historical = (
        estimate_heston_parameters(
            index_series
        )
    )

    parameters_q = (
        build_risk_neutral_parameters(
            parameters_historical,
            VARIANCE_RISK_PREMIUM_LAMBDA,
        )
    )

    risk_free_column = (
        f"RiskFree_"
        f"{par.GMAB_MATURITY_YEARS}Y"
    )

    risk_free_series = (
        pd.to_numeric(
            data[
                risk_free_column
            ],
            errors="coerce",
        )
        .dropna()
    )

    r = float(
        risk_free_series.iloc[-1]
    )

    (
        heston_analytic_price,
        _,
    ) = heston_put_price_delta(
        par.INITIAL_FUND_VALUE,
        par.G,
        parameters_q["v0"],
        r,
        par.TOTAL_FEE_ANNUAL,
        par.GMAB_MATURITY_YEARS,
        parameters_q,
    )

    equivalent_sigma = (
        find_equivalent_bs_sigma(
            heston_analytic_price,
            r,
        )
    )

    bs_scenarios = {
        "BS_sigma_equivalente": (
            equivalent_sigma
        ),
        "BS_storica_1y": (
            historical_volatility(
                returns,
                1,
            )
        ),
        "BS_storica_3y": (
            historical_volatility(
                returns,
                3,
            )
        ),
        "BS_storica_5y": (
            historical_volatility(
                returns,
                5,
            )
        ),
    }

    feller_left = (
        2.0
        * parameters_q["kappa"]
        * parameters_q["theta"]
    )

    feller_right = (
        parameters_q["xi"] ** 2
    )

    parameter_table = pd.DataFrame(
        [
            {
                "kappa_P": parameters_historical["kappa"],
                "theta_P": parameters_historical["theta"],
                "xi_P": parameters_historical["xi"],
                "rho_P": parameters_historical["rho"],
                "v0": parameters_historical["v0"],
                "lambda_varianza": VARIANCE_RISK_PREMIUM_LAMBDA,
                "kappa_Q": parameters_q["kappa"],
                "theta_Q": parameters_q["theta"],
                "xi_Q": parameters_q["xi"],
                "rho_Q": parameters_q["rho"],
                "risk_free": r,
                "fee": par.TOTAL_FEE_ANNUAL,
                "sigma_bs_equivalente": equivalent_sigma,
                "prezzo_heston": heston_analytic_price,
                "feller_left_Q": feller_left,
                "feller_right_Q": feller_right,
                "feller_satisfied_Q": (
                    feller_left
                    >= feller_right
                ),
                "schema_simulazione": "Euler full truncation",
                "finestra_varianza_giorni": VARIANCE_PROXY_WINDOW_DAYS,
                "passo_campionamento_varianza_giorni": VARIANCE_SAMPLING_DAYS,
                "finestre_varianza_sovrapposte": (
                    VARIANCE_PROXY_WINDOW_DAYS
                    > VARIANCE_SAMPLING_DAYS
                ),
                "stima_parametri": "dati storici - misura P",
                "pricing_simulazione": "misura Q con lambda varianza = 0",
                "calibrazione_superficie_opzioni": False,
            }
        ]
    )

    parameter_table.to_csv(
        OUTPUT_PARAMETERS,
        index=False,
    )

    print(
        "=== PARAMETRIZZAZIONE HESTON ==="
    )

    print(
        "Parametri stimati su dati storici "
        "dell'indice proxy in EUR (misura P)."
    )

    print(
        "Non si tratta di una calibrazione "
        "a una superficie di opzioni."
    )

    print(
        f"kappa_P: "
        f"{parameters_historical['kappa']:.6f}"
    )

    print(
        f"theta_P: "
        f"{parameters_historical['theta']:.6f}"
    )

    print(
        f"xi_P: "
        f"{parameters_historical['xi']:.6f}"
    )

    print(
        f"rho_P: "
        f"{parameters_historical['rho']:.6f}"
    )

    print(
        f"v0: "
        f"{parameters_historical['v0']:.6f}"
    )

    print(
        "\n=== PASSAGGIO P -> Q ==="
    )

    print(
        f"Premio al rischio di varianza lambda: "
        f"{VARIANCE_RISK_PREMIUM_LAMBDA:.6f}"
    )

    print(
        "Assunzione di base: lambda = 0, quindi "
        "kappa e theta stimati sotto P sono "
        "riutilizzati sotto Q."
    )

    print(
        f"kappa_Q: "
        f"{parameters_q['kappa']:.6f}"
    )

    print(
        f"theta_Q: "
        f"{parameters_q['theta']:.6f}"
    )

    print(
        "\n=== DIAGNOSTICA HESTON ==="
    )

    print(
        f"2*kappa_Q*theta_Q: "
        f"{feller_left:.6f}"
    )

    print(
        f"xi_Q^2: "
        f"{feller_right:.6f}"
    )

    print(
        f"Feller soddisfatta: "
        f"{feller_left >= feller_right}"
    )

    print(
        "Schema di simulazione: Euler full truncation."
    )

    print(
        f"Varianza realizzata: finestra rolling "
        f"di {VARIANCE_PROXY_WINDOW_DAYS} giorni, "
        f"campionata ogni {VARIANCE_SAMPLING_DAYS} giorni."
    )

    if (
        VARIANCE_PROXY_WINDOW_DAYS
        > VARIANCE_SAMPLING_DAYS
    ):
        print(
            "Nota: le finestre della varianza realizzata "
            "sono sovrapposte; questo puo' aumentare la "
            "persistenza stimata e ridurre kappa."
        )

    print(
        "\n=== VOLATILITA' BS EQUIVALENTE ==="
    )

    print(
        f"Prezzo Heston: "
        f"{heston_analytic_price:.6f}"
    )

    print(
        f"Sigma BS equivalente: "
        f"{equivalent_sigma:.4%}"
    )

    print(
        f"Prezzo BS equivalente: "
        f"{bs_put_price(
            par.INITIAL_FUND_VALUE,
            par.G,
            r,
            par.TOTAL_FEE_ANNUAL,
            equivalent_sigma,
            par.GMAB_MATURITY_YEARS,
        ):.6f}"
    )

    print(
        "\n=== SIMULAZIONE HESTON ==="
    )

    print(
        f"Percorsi: {N_PATHS}"
    )

    print(
        f"Scadenza: "
        f"{par.GMAB_MATURITY_YEARS} anni"
    )

    print(
        f"Ribilanciamento: "
        f"{par.BASE_REBALANCE_FREQUENCY}"
    )

    print(
        f"Tasso risk-free: "
        f"{r:.4%}"
    )

    print(
        f"Fee: "
        f"{par.TOTAL_FEE_ANNUAL:.4%}"
    )

    (
        hedge_paths,
        fund_paths,
        variance_paths,
        times,
        average_realized_variance,
    ) = simulate_heston_paths(
        parameters_q,
        r,
    )

    payoff = np.maximum(
        par.G
        - fund_paths[:, -1],
        0.0,
    )

    heston_mc_price = float(
        np.exp(
            -r
            * par.GMAB_MATURITY_YEARS
        )
        * payoff.mean()
    )

    print(
        "\n=== CONTROLLO PREZZO HESTON ==="
    )

    print(
        f"Prezzo analitico: "
        f"{heston_analytic_price:.6f}"
    )

    print(
        f"Prezzo Monte Carlo: "
        f"{heston_mc_price:.6f}"
    )

    print(
        f"Differenza: "
        f"{heston_mc_price - heston_analytic_price:.6f}"
    )

    results = []

    (
        heston_price,
        heston_error,
    ) = run_heston_hedge(
        hedge_paths,
        fund_paths,
        variance_paths,
        times,
        parameters_q,
        r,
    )

    variance_regression = (
        heston_variance_regression(
            average_realized_variance,
            heston_error,
        )
    )

    variance_regression.to_csv(
        OUTPUT_VARIANCE_REGRESSION,
        index=False,
    )

    print(
        "\n=== HESTON: HEDGING ERROR "
        "E VARIANZA REALIZZATA ==="
    )

    print(
        variance_regression.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    for path_number, error in enumerate(
        heston_error,
        start=1,
    ):
        results.append(
            {
                "percorso": path_number,
                "strategia": "Heston",
                "sigma_bs": np.nan,
                "prezzo_iniziale": heston_price,
                "pricing_error_t0": 0.0,
                "pricing_error_scadenza": 0.0,
                "hedging_error_totale": error,
                "hedging_error_capitale_equo": error,
            }
        )

    for (
        scenario,
        sigma,
    ) in bs_scenarios.items():

        (
            bs_price,
            bs_error_total,
        ) = run_bs_hedge(
            hedge_paths,
            fund_paths,
            times,
            sigma,
            r,
            initial_capital=None,
        )

        (
            _,
            bs_error_fair_capital,
        ) = run_bs_hedge(
            hedge_paths,
            fund_paths,
            times,
            sigma,
            r,
            initial_capital=heston_analytic_price,
        )

        pricing_error_t0 = (
            bs_price
            - heston_analytic_price
        )

        pricing_error_terminal = (
            pricing_error_t0
            * np.exp(
                r
                * par.GMAB_MATURITY_YEARS
            )
        )

        for path_number in range(
            1,
            N_PATHS + 1,
        ):
            results.append(
                {
                    "percorso": path_number,
                    "strategia": scenario,
                    "sigma_bs": sigma,
                    "prezzo_iniziale": bs_price,
                    "pricing_error_t0": pricing_error_t0,
                    "pricing_error_scadenza": pricing_error_terminal,
                    "hedging_error_totale": bs_error_total[
                        path_number - 1
                    ],
                    "hedging_error_capitale_equo": bs_error_fair_capital[
                        path_number - 1
                    ],
                }
            )

    results_table = pd.DataFrame(
        results
    )

    decomposition_check = (
        results_table.loc[
            results_table[
                "strategia"
            ] != "Heston"
        ]
        .assign(
            differenza=lambda x: (
                x[
                    "hedging_error_totale"
                ]
                - x[
                    "hedging_error_capitale_equo"
                ]
                - x[
                    "pricing_error_scadenza"
                ]
            )
        )[
            "differenza"
        ]
        .abs()
        .max()
    )

    summary = summarize_results(
        results
    )

    results_table.to_csv(
        OUTPUT_PATHS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print(
        "\n=== CONTROLLO DECOMPOSIZIONE ==="
    )

    print(
        f"Errore massimo identita': "
        f"{decomposition_check:.10f}"
    )

    print(
        "\nIdentita' verificata:"
    )

    print(
        "errore totale = "
        "errore hedging a capitale equo "
        "+ errore di pricing capitalizzato"
    )

    print(
        "\n=== CONFRONTO MODEL RISK ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    plot_data = (
        summary.copy()
    )

    x = np.arange(
        len(plot_data)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x - width / 2,
        plot_data[
            "rmse_totale"
        ],
        width,
        label="RMSE totale",
    )

    plt.bar(
        x + width / 2,
        plot_data[
            "rmse_hedging"
        ],
        width,
        label=(
            "RMSE con capitale iniziale equo"
        ),
    )

    plt.xticks(
        x,
        plot_data[
            "strategia"
        ],
        rotation=20,
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "Heston vs Black-Scholes: "
        "pricing e hedging"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PLOT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Distribuzione degli hedging error a capitale iniziale equo
    # --------------------------------------------------------

    strategy_order = [
        "Heston",
        "BS_sigma_equivalente",
        "BS_storica_1y",
        "BS_storica_3y",
        "BS_storica_5y",
    ]

    strategy_labels = [
        "Heston",
        r"BS $\sigma_{\mathrm{eq}}$",
        "BS storica 1 anno",
        "BS storica 3 anni",
        "BS storica 5 anni",
    ]

    distribution_data = [
        results_table.loc[
            results_table["strategia"] == strategy,
            "hedging_error_capitale_equo",
        ].to_numpy()
        for strategy in strategy_order
    ]

    plt.figure(
        figsize=(10, 5.5)
    )

    plt.boxplot(
        distribution_data,
        vert=False,
        tick_labels=strategy_labels,
        whis=(1, 99),
        showfliers=False,
    )

    # Matplotlib dispone il primo boxplot in basso:
    # invertiamo l'asse per mantenere l'ordine logico
    # Heston -> BS equivalente -> BS storiche.
    plt.gca().invert_yaxis()

    # Lo zero separa deficit (a sinistra)
    # e surplus (a destra).
    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Hedging error a capitale iniziale equo"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DISTRIBUTION_PLOT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nFile salvati:"
    )

    print(
        f"  - {OUTPUT_PARAMETERS}"
    )

    print(
        f"  - {OUTPUT_PATHS}"
    )

    print(
        f"  - {OUTPUT_SUMMARY}"
    )

    print(
        f"  - {OUTPUT_VARIANCE_REGRESSION}"
    )

    print(
        f"  - {OUTPUT_PLOT}"
    )

    print(
        f"  - {OUTPUT_DISTRIBUTION_PLOT}"
    )

if __name__ == "__main__":
    main()
