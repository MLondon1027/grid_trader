from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GridBacktestResult:
    equity: pd.DataFrame
    drawdowns: pd.DataFrame
    metrics: pd.DataFrame
    annual_returns: pd.DataFrame
    trade_log: pd.DataFrame
    diagnostics: dict[str, object]


def _prepare_bars(ohlc: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close"]
    missing = set(required).difference(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC data are missing columns: {sorted(missing)}")

    bars = ohlc.loc[:, required].copy().sort_index()
    bars.index = pd.to_datetime(bars.index).tz_localize(None)
    bars = bars.loc[~bars.index.duplicated(keep="last")]
    bars = bars.apply(pd.to_numeric, errors="coerce").dropna()
    if len(bars) < 2:
        raise ValueError("At least two complete daily price bars are required.")
    return bars


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity.div(equity.cummax()).sub(1.0)


def _metrics(equity: pd.Series, starting_capital: float) -> dict[str, float]:
    normalized = equity / starting_capital
    returns = normalized.pct_change(fill_method=None)
    returns.iloc[0] = normalized.iloc[0] - 1.0
    elapsed_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = normalized.iloc[-1] - 1.0
    cagr = normalized.iloc[-1] ** (1.0 / elapsed_years) - 1.0
    weekend_share = float((equity.index.dayofweek >= 5).mean())
    annualization = 365 if weekend_share > 0.10 else 252
    volatility = returns.std(ddof=1) * np.sqrt(annualization)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(annualization)
        if returns.std(ddof=1) > 0 else np.nan
    )
    annual = returns.resample("YE").apply(lambda values: (1.0 + values).prod() - 1.0)
    return {
        "Ending Value": float(equity.iloc[-1]),
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Annualized Volatility": float(volatility),
        "Sharpe (0% risk-free)": float(sharpe),
        "Maximum Drawdown": float(_drawdown(equity).min()),
        "Worst Calendar Year": float(annual.min()) if not annual.empty else np.nan,
    }


def run_grid_backtest(
    ohlc: pd.DataFrame,
    starting_capital: float = 10_000.0,
    profit_target_pct: float = 2.0,
    reentry_drop_pct: float = 2.0,
    transaction_cost_bps: float = 5.0,
    stop_loss_pct: float | None = None,
    benchmark_name: str = "Buy & Hold",
) -> GridBacktestResult:
    """Simulate one fully invested position with target-sale and dip-rebuy orders.

    The initial purchase occurs at the first close. Subsequent orders are tested
    against daily adjusted OHLC bars. Same-bar sale and rebuy are prohibited.
    If a stop and target are both touched during one bar, the stop is assumed to
    execute first because daily data do not reveal the intraday event order.
    """
    if starting_capital <= 0:
        raise ValueError("Starting capital must be positive.")
    if profit_target_pct <= 0:
        raise ValueError("Profit target must be positive.")
    if reentry_drop_pct < 0:
        raise ValueError("Re-entry decline cannot be negative.")
    if not 0 <= transaction_cost_bps < 10_000:
        raise ValueError("Transaction cost must be between 0 and 10,000 basis points.")
    if stop_loss_pct is not None and not 0 < stop_loss_pct < 100:
        raise ValueError("Stop loss must be between 0% and 100% when enabled.")

    bars = _prepare_bars(ohlc)
    target_fraction = profit_target_pct / 100.0
    reentry_fraction = reentry_drop_pct / 100.0
    stop_fraction = None if stop_loss_pct is None else stop_loss_pct / 100.0
    fee_fraction = transaction_cost_bps / 10_000.0

    cash = float(starting_capital)
    entry_capital = cash
    entry_price: float | None = float(bars.iloc[0]["Close"])
    entry_date: pd.Timestamp | None = bars.index[0]
    units = cash * (1.0 - fee_fraction) / entry_price
    cash = 0.0
    next_entry_price: float | None = None
    last_exit_date: pd.Timestamp | None = None
    trade_events = 1
    trades: list[dict[str, object]] = []
    strategy_values: list[float] = []
    exposure: list[float] = []

    for date, row in bars.iterrows():
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        if units > 0 and date != entry_date:
            assert entry_price is not None and entry_date is not None
            target_price = entry_price * (1.0 + target_fraction)
            stop_price = (
                entry_price * (1.0 - stop_fraction)
                if stop_fraction is not None else None
            )
            exit_price: float | None = None
            exit_reason: str | None = None

            if stop_price is not None and open_price <= stop_price:
                exit_price, exit_reason = open_price, "Stop"
            elif open_price >= target_price:
                exit_price, exit_reason = open_price, "Target"
            elif stop_price is not None and low <= stop_price:
                exit_price, exit_reason = stop_price, "Stop"
            elif high >= target_price:
                exit_price, exit_reason = target_price, "Target"

            if exit_price is not None:
                cash = units * exit_price * (1.0 - fee_fraction)
                trades.append(
                    {
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Exit Date": date,
                        "Exit Price": exit_price,
                        "Exit Reason": exit_reason,
                        "Net Return": cash / entry_capital - 1.0,
                        "Days Held": (date - entry_date).days,
                        "Capital After Exit": cash,
                    }
                )
                units = 0.0
                next_entry_price = exit_price * (1.0 - reentry_fraction)
                entry_price = None
                entry_date = None
                last_exit_date = date
                trade_events += 1

        if (
            units == 0
            and next_entry_price is not None
            and date != last_exit_date
            and low <= next_entry_price
        ):
            fill_price = min(open_price, next_entry_price)
            entry_capital = cash
            units = cash * (1.0 - fee_fraction) / fill_price
            cash = 0.0
            entry_price = fill_price
            entry_date = date
            next_entry_price = None
            trade_events += 1

        strategy_values.append(cash + units * close)
        exposure.append(1.0 if units > 0 else 0.0)

    strategy_equity = pd.Series(strategy_values, index=bars.index, name="Grid Strategy")
    benchmark_units = starting_capital * (1.0 - fee_fraction) / float(bars.iloc[0]["Close"])
    benchmark_equity = (benchmark_units * bars["Close"]).rename(benchmark_name)
    equity = pd.concat([strategy_equity, benchmark_equity], axis=1)
    drawdowns = equity.apply(_drawdown)

    returns = equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.iloc[0] / starting_capital - 1.0
    annual_returns = returns.resample("YE").apply(
        lambda values: (1.0 + values).prod() - 1.0
    )
    annual_returns.index = annual_returns.index.year
    annual_returns.index.name = "Year"

    metrics = pd.DataFrame(
        {
            "Grid Strategy": _metrics(strategy_equity, starting_capital),
            benchmark_name: _metrics(benchmark_equity, starting_capital),
        }
    ).T

    trade_log = pd.DataFrame(trades)
    target_exits = sum(trade["Exit Reason"] == "Target" for trade in trades)
    stop_exits = sum(trade["Exit Reason"] == "Stop" for trade in trades)
    elapsed_years = max((bars.index[-1] - bars.index[0]).days / 365.25, 1 / 365.25)
    diagnostics: dict[str, object] = {
        "Completed round trips": len(trades),
        "Target exits": target_exits,
        "Stop exits": stop_exits,
        "Round trips per year": len(trades) / elapsed_years,
        "Profitable cycles per year": target_exits / elapsed_years,
        "Time invested": float(np.mean(exposure)),
        "Trade events": trade_events,
        "Open position": bool(units > 0),
        "Open entry date": entry_date,
        "Open entry price": entry_price,
        "Open price return": (
            float(bars.iloc[-1]["Close"]) / entry_price - 1.0
            if units > 0 and entry_price is not None else np.nan
        ),
        "Waiting re-entry price": next_entry_price,
        "First data date": bars.index[0],
        "Last data date": bars.index[-1],
    }

    return GridBacktestResult(
        equity=equity,
        drawdowns=drawdowns,
        metrics=metrics,
        annual_returns=annual_returns,
        trade_log=trade_log,
        diagnostics=diagnostics,
    )
