"""
Clean Real-time Order Book Simulator + Smooth Candles
Compatible with Dash 3.x and Python 3.12+
No deprecated calls, no missing callbacks.
"""

from __future__ import annotations
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


# ============================================================
# Helpers / Models
# ============================================================

def new_id(prefix=""):
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def round_price(price, tick):
    return float(round(price / tick) * tick)


@dataclass
class Order:
    order_id: str
    side: str                  # BUY or SELL
    type: str                  # LIMIT / MARKET / IOC / FOK / STOP / STOP_LIMIT
    price: Optional[float]
    qty: float
    remaining: float
    timestamp: datetime
    seq: int
    tif: Optional[str] = None
    stop_price: Optional[float] = None
    status: str = "OPEN"


@dataclass
class TradeEvent:
    trade_id: str
    price: float
    qty: float
    aggressor: str
    timestamp: datetime
    maker_order_id: str
    taker_order_id: str
    seq: int


# ============================================================
# Order Book (Matching Engine)
# ============================================================

class OrderBook:
    def __init__(self, tick_size=0.25):
        self.tick = tick_size

        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}
        self.bid_prices: List[float] = []
        self.ask_prices: List[float] = []

        self.order_map: Dict[str, Order] = {}
        self.stop_orders: List[Order] = []
        self.trade_log: List[TradeEvent] = []

        self.seq = 0
        self.trade_seq = 0
        self.last_trade_price: Optional[float] = None
        self.last_trade_time: Optional[datetime] = None

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _next_seq(self):
        self.seq += 1
        return self.seq

    def _next_trade_seq(self):
        self.trade_seq += 1
        return self.trade_seq

    def best_bid(self):
        if not self.bid_prices:
            return None
        p = self.bid_prices[0]
        total = sum(o.remaining for o in self.bids[p])
        return p, total

    def best_ask(self):
        if not self.ask_prices:
            return None
        p = self.ask_prices[0]
        total = sum(o.remaining for o in self.asks[p])
        return p, total

    def mid(self):
        bb = self.best_bid()
        ba = self.best_ask()
        if bb and ba:
            return (bb[0] + ba[0]) / 2
        return self.last_trade_price or None

    # ----------------------------------------------------------
    # Managing Price Levels
    # ----------------------------------------------------------
    def _insert_level(self, price, side):
        if side == "BUY":
            if price not in self.bids:
                self.bids[price] = deque()
                i = 0
                while i < len(self.bid_prices) and self.bid_prices[i] > price:
                    i += 1
                self.bid_prices.insert(i, price)
        else:
            if price not in self.asks:
                self.asks[price] = deque()
                i = 0
                while i < len(self.ask_prices) and self.ask_prices[i] < price:
                    i += 1
                self.ask_prices.insert(i, price)

    def _remove_if_empty(self, price, side):
        if side == "BUY":
            dq = self.bids.get(price)
            if not dq:
                self.bids.pop(price, None)
                if price in self.bid_prices:
                    self.bid_prices.remove(price)
        else:
            dq = self.asks.get(price)
            if not dq:
                self.asks.pop(price, None)
                if price in self.ask_prices:
                    self.ask_prices.remove(price)

    # ----------------------------------------------------------
    # Submitting Orders
    # ----------------------------------------------------------
    def submit(self, side, type_, qty, price, now, tif=None, stop_price=None):
        oid = new_id("o")
        seq = self._next_seq()

        order = Order(
            order_id=oid, side=side, type=type_, price=price,
            qty=qty, remaining=qty, timestamp=now, seq=seq,
            tif=tif, stop_price=stop_price
        )

        # Handle STOP orders (stored but not traded yet)
        if type_ in ("STOP", "STOP_LIMIT"):
            self.stop_orders.append(order)
            self.order_map[oid] = order
            return order

        # Normal order → matching engine
        remainder, trades = self._match(order, now)

        for t in trades:
            self.trade_log.append(t)
            self.last_trade_price = t.price
            self.last_trade_time = t.timestamp

        # If remainder left → add passive
        if remainder.remaining > 0:
            if remainder.tif == "IOC":
                remainder.status = "CANCELLED"
            else:
                self._add_passive(remainder)
        else:
            remainder.status = "FILLED"

        self.order_map[oid] = remainder

        # Trigger stops if needed
        self._check_stops(now)
        return remainder

    # ----------------------------------------------------------
    # Matching Logic
    # ----------------------------------------------------------
    def _match(self, incoming, now):
        trades = []
        limit_price = None if incoming.type == "MARKET" else incoming.price

        while incoming.remaining > 0:
            opposite_prices = self.ask_prices if incoming.side == "BUY" else self.bid_prices
            if not opposite_prices:
                break

            best_price = opposite_prices[0]

            # Price constraint check
            if limit_price is not None:
                if incoming.side == "BUY" and best_price > limit_price:
                    break
                if incoming.side == "SELL" and best_price < limit_price:
                    break

            dq = self.asks[best_price] if incoming.side == "BUY" else self.bids[best_price]

            while dq and incoming.remaining > 0:
                maker = dq[0]
                q = min(incoming.remaining, maker.remaining)

                t = TradeEvent(
                    trade_id=new_id("t"),
                    price=best_price,
                    qty=q,
                    aggressor=incoming.side,
                    timestamp=now,
                    maker_order_id=maker.order_id,
                    taker_order_id=incoming.order_id,
                    seq=self._next_trade_seq()
                )
                trades.append(t)

                maker.remaining -= q
                incoming.remaining -= q

                if maker.remaining <= 0:
                    dq.popleft()
                self.order_map[maker.order_id] = maker

            self._remove_if_empty(best_price, "SELL" if incoming.side == "BUY" else "BUY")

        return incoming, trades

    # ----------------------------------------------------------
    # Passive Insert
    # ----------------------------------------------------------
    def _add_passive(self, order):
        p = round_price(order.price, self.tick)
        self._insert_level(p, order.side)

        dq = self.bids[p] if order.side == "BUY" else self.asks[p]
        dq.append(order)
        order.price = p

    # ----------------------------------------------------------
    # Stop Orders
    # ----------------------------------------------------------
    def _check_stops(self, now):
        if not self.stop_orders or self.last_trade_price is None:
            return

        triggered = []

        for o in list(self.stop_orders):
            if o.side == "BUY" and self.last_trade_price >= o.stop_price:
                triggered.append(o)
            elif o.side == "SELL" and self.last_trade_price <= o.stop_price:
                triggered.append(o)

        for o in triggered:
            self.stop_orders.remove(o)
            o.timestamp = now
            o.seq = self._next_seq()

            if o.type == "STOP":
                o.type = "MARKET"
            elif o.type == "STOP_LIMIT":
                o.type = "LIMIT"

            self.submit(o.side, o.type, o.remaining, o.price, now, o.tif, None)

    # ----------------------------------------------------------
    # Exports
    # ----------------------------------------------------------
    def snapshot_l2(self, depth=12):
        bids = [(p, sum(o.remaining for o in self.bids[p])) for p in self.bid_prices[:depth]]
        asks = [(p, sum(o.remaining for o in self.asks[p])) for p in self.ask_prices[:depth]]
        return {"bids": bids, "asks": asks}

    def trades_df(self):
        rows = []
        for t in self.trade_log:
            rows.append({
                "timestamp": t.timestamp,
                "price": t.price,
                "qty": t.qty,
                "aggressor": t.aggressor
            })
        if not rows:
            return pd.DataFrame(columns=["timestamp", "price", "qty", "aggressor"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")


# ============================================================
# Real-time Order Flow Simulator
# ============================================================

class Simulator:
    def __init__(self, book, base_price=100.0, tick=0.25):
        self.book = book
        self.base_price = base_price
        self.tick = tick
        np.random.seed(42)

    def _rand_side(self):
        return "BUY" if np.random.rand() < 0.5 else "SELL"

    def _rand_qty(self):
        if np.random.rand() < 0.9:
            return round(np.random.exponential(3) + 0.1, 2)
        return round(np.random.exponential(15) + 1, 2)

    def _rand_price(self):
        mid = self.book.mid() or self.base_price
        ticks = int(np.random.randn() * 2)
        return round_price(mid + ticks * self.tick, self.tick)

    def step(self, now, secs=1.0):
        n_orders = np.random.poisson(2)
        for _ in range(n_orders):
            ts = now + timedelta(seconds=np.random.rand() * secs)
            self._emit(ts)

    def _emit(self, now):
        side = self._rand_side()
        qty = self._rand_qty()
        type_ = "LIMIT" if np.random.rand() < 0.75 else "MARKET"

        price = None if type_ == "MARKET" else self._rand_price()

        self.book.submit(side, type_, qty, price, now)


# ============================================================
# Candle Construction
# ============================================================

def make_candles(df, start, end):
    if df.empty:
        idx = pd.date_range(start, end, freq="60S", inclusive="left")
        return pd.DataFrame(index=idx, columns=["open", "high", "low", "close", "volume"]).fillna(0)

    df = df.set_index("timestamp")

    ohlc = df["price"].resample("60S").ohlc()
    vol = df["qty"].resample("60S").sum().rename("volume")

    candles = ohlc.join(vol).fillna(method="ffill").fillna(0)
    return candles


# ============================================================
# Plot Builders
# ============================================================

def build_candles_fig(candles, trades):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=candles.index,
        open=candles["open"],
        high=candles["high"],
        low=candles["low"],
        close=candles["close"],
        name="Candles"
    ))

    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades["timestamp"], y=trades["price"],
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            name="Trades"
        ))

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
    return fig


def build_l2_fig(book):
    snap = book.snapshot_l2(12)
    fig = go.Figure()

    # Bids
    if snap["bids"]:
        prices = [str(p) for p, q in snap["bids"]]
        qtys = [q for p, q in snap["bids"]]
        fig.add_trace(go.Bar(x=qtys, y=prices, orientation='h', name="Bids", marker_color="green"))

    # Asks
    if snap["asks"]:
        prices = [str(p) for p, q in snap["asks"]]
        qtys = [q for p, q in snap["asks"]]
        fig.add_trace(go.Bar(x=qtys, y=prices, orientation='h', name="Asks", marker_color="red"))

    fig.update_layout(template="plotly_dark", height=500)
    return fig


# ============================================================
# Dash App
# ============================================================

def create_app(book, sim):
    app = Dash(__name__)

    app.layout = html.Div([
        html.Div(id="stats", style={"color": "white", "marginBottom": "10px"}),

        dcc.Graph(id="candles"),
        dcc.Graph(id="l2"),

        dcc.Interval(id="timer", interval=1000, n_intervals=0)
    ], style={"backgroundColor": "#111", "padding": "20px"})

    # MAIN UPDATE LOOP
    @app.callback(
        Output("candles", "figure"),
        Output("l2", "figure"),
        Output("stats", "children"),
        Input("timer", "n_intervals")
    )
    def update_graph(n):
        now = datetime.now(UTC).replace(microsecond=0)

        # Simulation steps
        sim.step(now)

        trades = book.trades_df()

        start = now - timedelta(minutes=30)
        end = now + timedelta(seconds=1)

        candles = make_candles(trades, start, end)

        fig1 = build_candles_fig(candles, trades)
        fig2 = build_l2_fig(book)

        bb = book.best_bid()
        ba = book.best_ask()

        return fig1, fig2, [
            html.Div(f"Best Bid: {bb}"),
            html.Div(f"Best Ask: {ba}"),
            html.Div(f"Mid: {book.mid()}"),
            html.Div(f"Last Trade: {book.last_trade_price}")
        ]

    return app


# ============================================================
# Entrypoint
# ============================================================

def main():
    book = OrderBook(tick_size=0.25)
    sim = Simulator(book)

    # Seed initial depth
    now = datetime.now(UTC).replace(microsecond=0)
    for i in range(5):
        book.submit("BUY", "LIMIT", 3, round_price(100 - (i+1)*0.25, 0.25), now)
    for i in range(5):
        book.submit("SELL", "LIMIT", 3, round_price(100 + (i+1)*0.25, 0.25), now)

    app = create_app(book, sim)
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
