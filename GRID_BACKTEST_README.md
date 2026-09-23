# Grid backtest

This is a standalone Streamlit application. It does not import or modify the
existing Market Strategy Lab.

## Run it

```bash
python -m pip install -r requirements.txt
streamlit run grid_trader_app.py
```

Select the grid strategy, enter a Yahoo Finance ticker such as `BTC-USD`,
choose the test dates, and set:

- sell target above entry;
- required decline after a sale before rebuying;
- one-way transaction cost; and
- optional stop-loss.

The output includes the strategy-versus-buy-and-hold equity curve, drawdowns,
annual returns, completed cycles per year, time invested, any unfinished
position, and a downloadable trade ledger.

## Execution assumptions

- The first purchase fills at the first available closing price.
- Later target, stop, and rebuy orders use adjusted daily OHLC prices.
- A sale and rebuy cannot occur on the same daily bar.
- If both a stop and target are touched on the same bar, the stop fills first.
- A gap through an order fills at the opening price.
- Transaction costs apply on every purchase and sale.
- An unfinished final position is marked to the final closing price.
- Taxes, spread beyond the entered cost, market impact, and exchange outages
  are excluded.

Daily bars cannot establish the exact sequence of intraday highs and lows.
For serious evaluation, rerun a promising rule on hourly or minute data before
using it with money.
