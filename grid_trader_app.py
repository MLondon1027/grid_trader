from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from grid_strategy_engine import (
    find_parameters_for_trade_frequency,
    run_grid_backtest,
)


st.set_page_config(page_title="Grid Trading Backtest", page_icon="↕", layout="wide")


@st.cache_data(ttl=21_600, show_spinner=False)
def download_ohlc(ticker: str, start: date, end: date) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=30,
    )
    if data.empty:
        raise ValueError(f"No market data were returned for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(data.columns):
        raise ValueError("Downloaded data did not contain complete OHLC prices.")
    return data[required].sort_index().dropna()


def money(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float, decimals: int = 1) -> str:
    return "—" if pd.isna(value) else f"{value:.{decimals}%}"


st.title("Grid Trading Backtest")
st.write(
    "Buy an asset, place a limit sale above the entry price, then wait for a "
    "specified decline before buying again. Compare the complete result with "
    "buying and holding the same asset."
)

with st.sidebar:
    st.header("Strategy settings")
    ticker = st.text_input(
        "Yahoo Finance ticker",
        value="BTC-USD",
        help="Examples: BTC-USD, ETH-USD, TSLA, NVDA.",
    ).strip().upper()
    start_date = st.date_input(
        "Start date", value=date(2020, 1, 1), min_value=date(2003, 1, 1)
    )
    end_date = st.date_input("End date", value=date.today(), max_value=date.today())
    starting_capital = st.number_input(
        "Starting investment", min_value=100.0, value=10_000.0, step=1_000.0
    )
    search_mode = st.radio(
        "Parameter mode",
        ["Find about 100 cycles/year", "Choose parameters manually"],
    )
    if search_mode.startswith("Find"):
        target_cycles_per_year = st.number_input(
            "Desired profitable cycles per year",
            min_value=1,
            max_value=180,
            value=100,
            step=5,
        )
        st.caption(
            "Tests profit targets from 2% to 3% and re-entry pullbacks from 0% to 5%."
        )
    else:
        profit_target_pct = st.number_input(
            "Sell target above entry (%)",
            min_value=0.1,
            max_value=25.0,
            value=2.0,
            step=0.1,
        )
        reentry_drop_pct = st.number_input(
            "Pullback from highest price after sale (%)",
            min_value=0.0,
            max_value=25.0,
            value=0.5,
            step=0.1,
            help=(
                "After selling, the app tracks new highs and buys when price pulls "
                "back by this percentage. It does not require a return below the sale price."
            ),
        )
    transaction_cost_bps = st.number_input(
        "Cost per one-way order (basis points)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        help="Five basis points equals 0.05%. A round trip normally contains two orders.",
    )
    use_stop = st.checkbox("Use a stop-loss", value=False)
    stop_loss_pct = st.number_input(
        "Stop below entry (%)",
        min_value=0.1,
        max_value=50.0,
        value=8.0,
        step=0.5,
        disabled=not use_stop,
    )
    run = st.button("Run backtest", type="primary", use_container_width=True)

st.info(
    "Research backtest only. Taxes, market impact, exchange outages, and spread "
    "beyond the entered transaction cost are excluded."
)

if run:
    if not ticker:
        st.error("Enter a ticker.")
        st.stop()
    if start_date >= end_date:
        st.error("The start date must be before the end date.")
        st.stop()

    try:
        with st.spinner("Downloading prices and simulating orders…"):
            bars = download_ohlc(ticker, start_date, end_date)
            if search_mode.startswith("Find"):
                result, parameter_search = find_parameters_for_trade_frequency(
                    bars,
                    starting_capital=starting_capital,
                    target_cycles_per_year=target_cycles_per_year,
                    transaction_cost_bps=transaction_cost_bps,
                    stop_loss_pct=stop_loss_pct if use_stop else None,
                    benchmark_name=f"Buy & Hold {ticker}",
                )
                selected_parameters = parameter_search.iloc[0]
            else:
                result = run_grid_backtest(
                    bars,
                    starting_capital=starting_capital,
                    profit_target_pct=profit_target_pct,
                    reentry_drop_pct=reentry_drop_pct,
                    transaction_cost_bps=transaction_cost_bps,
                    stop_loss_pct=stop_loss_pct if use_stop else None,
                    reentry_reference="post_exit_high",
                    benchmark_name=f"Buy & Hold {ticker}",
                )

        grid = result.metrics.loc["Grid Strategy"]
        benchmark_name = next(
            name for name in result.metrics.index if name != "Grid Strategy"
        )
        benchmark = result.metrics.loc[benchmark_name]
        diagnostics = result.diagnostics

        if search_mode.startswith("Find"):
            achieved = float(diagnostics["Profitable cycles per year"])
            selected_text = (
                f"Best frequency match: {selected_parameters['Profit Target %']:.2f}% "
                f"profit target and {selected_parameters['Re-entry Pullback %']:.2f}% "
                f"pullback, producing {achieved:.1f} profitable cycles per year."
            )
            if abs(achieved - target_cycles_per_year) <= 10:
                st.success(selected_text)
            else:
                st.warning(
                    selected_text
                    + f" The tested rules could not get close to {target_cycles_per_year} "
                    "without lowering the 2% minimum profit target."
                )

            with st.expander("View the closest parameter combinations"):
                search_table = parameter_search.head(15).copy()
                for column in ["Total Return", "Maximum Drawdown"]:
                    search_table[column] = search_table[column].map(percent)
                st.dataframe(search_table, hide_index=True, use_container_width=True)

        summary = st.columns(5)
        summary[0].metric(
            "Grid ending value",
            money(grid["Ending Value"]),
            money(grid["Ending Value"] - benchmark["Ending Value"]) + " vs. hold",
        )
        summary[1].metric("Buy-and-hold value", money(benchmark["Ending Value"]))
        summary[2].metric("Target exits", f"{diagnostics['Target exits']:,}")
        summary[3].metric(
            "Profitable cycles/year", f"{diagnostics['Profitable cycles per year']:.1f}"
        )
        summary[4].metric("Time invested", percent(diagnostics["Time invested"]))

        if diagnostics["Open position"]:
            st.warning(
                "The test ends with an unfinished position entered at "
                f"${diagnostics['Open entry price']:,.2f} on "
                f"{pd.Timestamp(diagnostics['Open entry date']):%Y-%m-%d}. Its price "
                f"return is {diagnostics['Open price return']:.1%}; the position is "
                "included at the final market price."
            )
        elif diagnostics["Waiting re-entry price"] is not None:
            st.info(
                "The strategy ends in cash waiting to rebuy at "
                f"${diagnostics['Waiting re-entry price']:,.2f}."
            )

        st.subheader("Growth of the starting investment")
        growth = result.equity.rename_axis("Date").reset_index().melt(
            "Date", var_name="Portfolio", value_name="Value"
        )
        growth_chart = px.line(growth, x="Date", y="Value", color="Portfolio")
        growth_chart.update_layout(
            yaxis_tickprefix="$", hovermode="x unified", legend_title_text=""
        )
        st.plotly_chart(growth_chart, use_container_width=True)

        st.subheader("Drawdowns")
        drawdowns = result.drawdowns.rename_axis("Date").reset_index().melt(
            "Date", var_name="Portfolio", value_name="Drawdown"
        )
        drawdown_chart = px.area(
            drawdowns, x="Date", y="Drawdown", color="Portfolio"
        )
        drawdown_chart.update_layout(
            yaxis_tickformat=".0%", hovermode="x unified", legend_title_text=""
        )
        st.plotly_chart(drawdown_chart, use_container_width=True)

        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Performance statistics")
            metrics = result.metrics.copy()
            metrics["Ending Value"] = metrics["Ending Value"].map(money)
            for column in [
                "Total Return",
                "CAGR",
                "Annualized Volatility",
                "Maximum Drawdown",
                "Worst Calendar Year",
            ]:
                metrics[column] = metrics[column].map(percent)
            metrics["Sharpe (0% risk-free)"] = metrics["Sharpe (0% risk-free)"].map(
                lambda value: "—" if pd.isna(value) else f"{value:.2f}"
            )
            st.dataframe(metrics.T, use_container_width=True)

        with right:
            st.subheader("Execution summary")
            execution = pd.DataFrame(
                {
                    "Measure": [
                        "Completed round trips",
                        "Target exits",
                        "Stop exits",
                        "Round trips per year",
                        "Profitable cycles per year",
                        "Time invested",
                    ],
                    "Value": [
                        f"{diagnostics['Completed round trips']:,}",
                        f"{diagnostics['Target exits']:,}",
                        f"{diagnostics['Stop exits']:,}",
                        f"{diagnostics['Round trips per year']:.1f}",
                        f"{diagnostics['Profitable cycles per year']:.1f}",
                        percent(diagnostics["Time invested"]),
                    ],
                }
            )
            st.dataframe(execution, hide_index=True, use_container_width=True)

            st.subheader("Calendar-year returns")
            st.dataframe(
                result.annual_returns.style.format("{:.1%}"),
                use_container_width=True,
            )

        st.subheader("Completed trades")
        if result.trade_log.empty:
            st.info("No round trip completed during the selected period.")
        else:
            ledger = result.trade_log.copy()
            ledger["Entry Date"] = pd.to_datetime(ledger["Entry Date"]).dt.strftime(
                "%Y-%m-%d"
            )
            ledger["Exit Date"] = pd.to_datetime(ledger["Exit Date"]).dt.strftime(
                "%Y-%m-%d"
            )
            st.dataframe(
                ledger.style.format(
                    {
                        "Entry Price": "${:,.2f}",
                        "Exit Price": "${:,.2f}",
                        "Net Return": "{:.2%}",
                        "Capital After Exit": "${:,.2f}",
                    }
                ),
                use_container_width=True,
                height=480,
            )
            st.download_button(
                "Download trade ledger",
                result.trade_log.to_csv(index=False).encode("utf-8"),
                "grid_trade_ledger.csv",
                "text/csv",
            )

        st.download_button(
            "Download daily portfolio values",
            result.equity.rename_axis("Date").reset_index().to_csv(index=False).encode(
                "utf-8"
            ),
            "grid_backtest_portfolio_values.csv",
            "text/csv",
        )

        st.caption(
            "Daily adjusted OHLC bars are used. Same-day sale and rebuy are "
            "prohibited. If a stop and target are both touched on one bar, the "
            "stop is assumed to execute first. A gap through an order fills at the open. "
            "Automatic parameter selection is in-sample and targets trade frequency, not return."
        )
    except Exception as exc:
        st.error(f"Backtest could not run: {exc}")
        st.caption("Yahoo Finance can throttle downloads temporarily. Retry later if the ticker is valid.")
else:
    st.subheader("What this test answers")
    st.markdown(
        """
        - How many profitable target exits actually occurred.
        - Whether the strategy approached 100 completed cycles per year.
        - How often the capital remained invested or waited in cash.
        - Whether small realized gains compensated for missed upside and unfinished losses.
        - Whether the complete strategy beat simply holding the same asset.
        """
    )
