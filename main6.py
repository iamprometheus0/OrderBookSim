"""
Realistic Market Order-Book Simulator (Dash) — Minimal Refactor (Option A)
- Single-file, cleaned & organized.
- Adds OFI tracking + OFI time-series chart under candles.
- Maintains all previous features and visualizations.
Run:
    python3 main_realistic.py
"""

from __future__ import annotations
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

# -----------------------
# Helpers & Data Models
# -----------------------

def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"

def round_price(p: float, tick: float) -> float:
    return float(round(round(p / tick) * tick, 8))

@dataclass
class Order:
    order_id: str
    side: str                # "BUY" / "SELL"
    type: str                # "LIMIT" / "MARKET" / "STOP" / "STOP_LIMIT"
    price: Optional[float]
    qty: float
    remaining: float
    timestamp: datetime
    seq: int
    tif: Optional[str] = None
    stop_price: Optional[float] = None
    status: str = "OPEN"     # OPEN / PARTIAL / FILLED / CANCELLED
    iceberg_total: Optional[float] = None
    iceberg_display: Optional[float] = None

@dataclass
class TradeEvent:
    trade_id: str
    price: float
    qty: float
    aggressor: str           # side of taker
    timestamp: datetime
    maker_order_id: str
    taker_order_id: str
    seq: int

# -----------------------
# Matching Engine: OrderBook
# -----------------------

class OrderBook:
    def __init__(self, tick_size: float = 0.25):
        self.tick = tick_size
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}
        self.bid_prices: List[float] = []  # descending
        self.ask_prices: List[float] = []  # ascending
        self.order_map: Dict[str, Order] = {}
        self.stop_orders: List[Order] = []
        self.trade_log: List[TradeEvent] = []

        self._seq = 0
        self._trade_seq = 0
        self.last_trade_price: Optional[float] = None
        self.last_trade_time: Optional[datetime] = None

        # OFI tracking
        self.prev_snapshot: Optional[Tuple[float, float]] = None  # (bid_vol, ask_vol)
        self.ofi: float = 0.0

    # Sequence helpers
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _next_trade_seq(self) -> int:
        self._trade_seq += 1
        return self._trade_seq

    # Price-level helpers
    def _insert_level(self, price: float, side: str):
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

    def _remove_level_if_empty(self, price: float, side: str):
        if side == "BUY":
            dq = self.bids.get(price)
            if not dq or len(dq) == 0:
                self.bids.pop(price, None)
                if price in self.bid_prices:
                    self.bid_prices.remove(price)
        else:
            dq = self.asks.get(price)
            if not dq or len(dq) == 0:
                self.asks.pop(price, None)
                if price in self.ask_prices:
                    self.ask_prices.remove(price)

    # Top-of-book / mid
    def best_bid(self) -> Optional[Tuple[float, float]]:
        if not self.bid_prices:
            return None
        p = self.bid_prices[0]
        qty = sum(o.remaining for o in self.bids[p])
        return p, qty

    def best_ask(self) -> Optional[Tuple[float, float]]:
        if not self.ask_prices:
            return None
        p = self.ask_prices[0]
        qty = sum(o.remaining for o in self.asks[p])
        return p, qty

    def mid_price(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb and ba:
            return (bb[0] + ba[0]) / 2
        if self.last_trade_price is not None:
            return self.last_trade_price
        return None

    # Submit / cancel
    def submit_order(self, side: str, order_type: str, qty: float,
                     price: Optional[float], now: datetime,
                     time_in_force: Optional[str] = None,
                     stop_price: Optional[float] = None,
                     iceberg_total: Optional[float] = None,
                     iceberg_display: Optional[float] = None) -> Order:

        seq = self._next_seq()
        oid = new_id("o")
        o = Order(order_id=oid, side=side, type=order_type, price=price,
                  qty=qty, remaining=qty, timestamp=now, seq=seq,
                  tif=time_in_force, stop_price=stop_price,
                  iceberg_total=iceberg_total, iceberg_display=iceberg_display)

        # Stop orders kept separately
        if order_type in ("STOP", "STOP_LIMIT"):
            self.stop_orders.append(o)
            self.order_map[oid] = o
            # OFI: book state changed by adding a stop order
            self.compute_ofi()
            return o

        # Route to matching core
        remainder, trades = self._match(o, now)
        for t in trades:
            self.trade_log.append(t)
            self.last_trade_price = t.price
            self.last_trade_time = t.timestamp

        # Insert remainder as passive if allowed
        if remainder.remaining > 0:
            if remainder.tif == "IOC":
                remainder.status = "CANCELLED"
            else:
                self._insert_passive(remainder)
        else:
            remainder.status = "FILLED"
        self.order_map[oid] = remainder

        # Check stops after trades
        self._check_stop_triggers(now)

        # OFI after all state changes
        self.compute_ofi()
        return remainder

    def cancel_order(self, order_id: str) -> bool:
        o = self.order_map.get(order_id)
        if not o or o.status != "OPEN":
            return False
        price = o.price
        side = o.side
        dq = self.bids.get(price) if side == "BUY" else self.asks.get(price)
        if dq:
            for el in list(dq):
                if el.order_id == order_id:
                    dq.remove(el)
                    o.status = "CANCELLED"
                    self._remove_level_if_empty(price, side)
                    # OFI: book volume removed
                    self.compute_ofi()
                    return True
        return False

    # Matching core
    def _match(self, incoming: Order, now: datetime) -> Tuple[Order, List[TradeEvent]]:
        trades: List[TradeEvent] = []
        price_limit = None if incoming.type == "MARKET" else incoming.price

        while incoming.remaining > 0:
            opposite_prices = self.ask_prices if incoming.side == "BUY" else self.bid_prices
            if not opposite_prices:
                break
            best_price = opposite_prices[0]
            # price constraint
            if price_limit is not None:
                if incoming.side == "BUY" and best_price > price_limit:
                    break
                if incoming.side == "SELL" and best_price < price_limit:
                    break
            level_dq = self.asks[best_price] if incoming.side == "BUY" else self.bids[best_price]
            # FIFO at level
            while level_dq and incoming.remaining > 0:
                maker = level_dq[0]
                trade_qty = min(incoming.remaining, maker.remaining)
                trade_price = best_price
                t = TradeEvent(
                    trade_id=new_id("t"),
                    price=trade_price,
                    qty=trade_qty,
                    aggressor=incoming.side,
                    timestamp=now,
                    maker_order_id=maker.order_id,
                    taker_order_id=incoming.order_id,
                    seq=self._next_trade_seq()
                )
                trades.append(t)
                # update sizes/statuses
                incoming.remaining -= trade_qty
                maker.remaining -= trade_qty
                if maker.remaining <= 0:
                    maker.status = "FILLED"
                    level_dq.popleft()
                else:
                    maker.status = "PARTIAL"
                self.order_map.setdefault(maker.order_id, maker)
            # remove empty level
            self._remove_level_if_empty(best_price, "SELL" if incoming.side == "BUY" else "BUY")
        return incoming, trades

    def _insert_passive(self, order: Order):
        if order.price is None:
            order.status = "CANCELLED"
            return
        p = round_price(order.price, self.tick)
        order.price = p
        self._insert_level(p, order.side)
        dq = self.bids[p] if order.side == "BUY" else self.asks[p]
        dq.append(order)
        order.status = "OPEN"

    # Stop triggers
    def _check_stop_triggers(self, now: datetime):
        if not self.stop_orders:
            return
        triggered = []
        last = self.last_trade_price
        if last is None:
            return
        for o in list(self.stop_orders):
            if o.side == "BUY" and o.stop_price is not None and last >= o.stop_price:
                triggered.append(o)
            elif o.side == "SELL" and o.stop_price is not None and last <= o.stop_price:
                triggered.append(o)
        for o in triggered:
            self.stop_orders.remove(o)
            if o.type == "STOP":
                o.type = "MARKET"
                o.price = None
            elif o.type == "STOP_LIMIT":
                o.type = "LIMIT"
            o.seq = self._next_seq()
            o.timestamp = datetime.now(UTC)
            remainder, trades = self._match(o, o.timestamp)
            for t in trades:
                self.trade_log.append(t)
                self.last_trade_price = t.price
            if remainder.remaining > 0 and remainder.tif != "IOC":
                self._insert_passive(remainder)
                self.order_map[remainder.order_id] = remainder
            else:
                remainder.status = "FILLED" if remainder.remaining == 0 else "CANCELLED"
                self.order_map[remainder.order_id] = remainder

        # OFI after stop-driven changes
        self.compute_ofi()

    # Snapshot & export
    def snapshot_l2(self, depth: int = 12) -> Dict[str, List[Tuple[float, float]]]:
        bids = []
        asks = []
        for p in self.bid_prices[:depth]:
            total = sum(o.remaining for o in self.bids[p])
            bids.append((p, total))
        for p in self.ask_prices[:depth]:
            total = sum(o.remaining for o in self.asks[p])
            asks.append((p, total))
        return {"bids": bids, "asks": asks}

    def trades_to_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trade_log:
            rows.append({
                "trade_id": t.trade_id,
                "price": t.price,
                "qty": t.qty,
                "aggressor": t.aggressor,
                "timestamp": t.timestamp,
                "maker": t.maker_order_id,
                "taker": t.taker_order_id,
                "seq": t.seq
            })
        if not rows:
            return pd.DataFrame(columns=["trade_id","price","qty","aggressor","timestamp"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")

    # OFI computation (top-of-book by default)
    def compute_ofi(self, depth: int = 1) -> float:
        """
        OFI = (Bid added - Bid removed) - (Ask added - Ask removed)
        Uses top depth levels (default 1). Updates self.ofi and prev_snapshot.
        """
        snap = self.snapshot_l2(depth=depth)
        bids = snap.get("bids", [])
        asks = snap.get("asks", [])

        bid_vol = bids[0][1] if bids else 0.0
        ask_vol = asks[0][1] if asks else 0.0

        if self.prev_snapshot is None:
            self.prev_snapshot = (bid_vol, ask_vol)
            return self.ofi

        prev_bid, prev_ask = self.prev_snapshot
        bid_added = max(0.0, bid_vol - prev_bid)
        bid_removed = max(0.0, prev_bid - bid_vol)
        ask_added = max(0.0, ask_vol - prev_ask)
        ask_removed = max(0.0, prev_ask - ask_vol)

        self.ofi = (bid_added - bid_removed) - (ask_added - ask_removed)
        self.prev_snapshot = (bid_vol, ask_vol)
        return self.ofi

# -----------------------
# Market Simulator
# -----------------------

class MarketSimulator:
    def __init__(self, book: OrderBook,
                 base_price: float = 100.0,
                 tick: float = 0.25,
                 seed: Optional[int] = 42):
        self.book = book
        self.base_price = base_price
        self.tick = tick
        self.rng = np.random.default_rng(seed)
        # Hawkes-like intensities
        self.mu = 0.9
        self.alpha = 0.6
        self.decay = 1.5
        self.intensity_buy = 0.5
        self.intensity_sell = 0.5
        # momentum memory
        self.momentum = 0.0
        self.mom_decay = 0.9
        self.mom_alpha = 0.25
        # vol tracking
        self.price_window = deque(maxlen=200)
        # replenishment / iceberg
        self.replenish_prob = 0.6
        self.iceberg_prob = 0.08
        # impact scaling
        self.base_impact = 0.2
        self.stress_scale = 2.0

    def update_after_trade(self, trade: TradeEvent):
        sign = 1 if trade.aggressor == "BUY" else -1
        self.momentum = self.mom_alpha * sign + self.mom_decay * self.momentum
        if trade.aggressor == "BUY":
            self.intensity_buy += self.alpha
        else:
            self.intensity_sell += self.alpha
        self.price_window.append(trade.price)
        self.intensity_buy = max(0.01, self.intensity_buy * np.exp(-1.0 / self.decay))
        self.intensity_sell = max(0.01, self.intensity_sell * np.exp(-1.0 / self.decay))

    def realized_volatility(self, lookback: int = 60) -> float:
        prices = list(self.price_window)[-lookback:]
        if len(prices) < 2:
            return 0.0
        logrets = np.diff(np.log(np.array(prices) + 1e-12))
        return float(np.std(logrets, ddof=0))

    def stress_metric(self) -> float:
        vol = self.realized_volatility(60)
        mom = abs(self.momentum)
        return vol * self.stress_scale + mom

    def choose_side(self) -> str:
        base_buy_prob = 0.5 + 0.3 * np.tanh(self.momentum)
        total_intensity = self.intensity_buy + self.intensity_sell + 1e-12
        hawkes_buy = self.intensity_buy / total_intensity
        p_buy = 0.6 * base_buy_prob + 0.4 * hawkes_buy
        return "BUY" if self.rng.random() < p_buy else "SELL"

    def sample_order_size(self, side: str, is_market: bool = False) -> float:
        vol = self.realized_volatility(60)
        if self.rng.random() < 0.9:
            size = self.rng.exponential(2.5) + 0.1
        else:
            size = self.rng.exponential(12) + 1.0
        if vol > 0.0005:
            if is_market:
                size *= (1.0 + min(6.0, vol * 300))
            else:
                size *= max(0.3, 1.0 - min(0.9, vol * 200))
        return round(float(size), 2)

    def sample_limit_price(self, side: str) -> float:
        mid = self.book.mid_price() or self.base_price
        stress = self.stress_metric()
        std_ticks = max(1, int(1 + stress * 4))
        ticks = int(round(self.rng.normal(0, std_ticks)))
        return round_price(mid + (ticks * self.tick), self.tick)

    def compute_walk_depth(self, base_qty: float) -> int:
        stress = self.stress_metric()
        base_depth = 1 if stress < 0.5 else 1 + int(stress * 3)
        extra = int(min(10, base_qty / 2.0))
        return base_depth + extra

    def emit_order(self, now: datetime):
        side = self.choose_side()
        stress = self.stress_metric()
        prob_market = 0.12 + 0.25 * min(1.0, stress)
        is_market = self.rng.random() < prob_market
        size = self.sample_order_size(side, is_market=is_market)

        if is_market:
            depth = self.compute_walk_depth(size)
            # Use market order submit; the matching core will walk as needed
            self.book.submit_order(side=side, order_type="MARKET", qty=size, price=None, now=now)
        else:
            # sometimes cross on best to simulate taker-on-limit
            if self.rng.random() < (0.18 + 0.15 * min(1.0, stress)):
                ba = self.book.best_ask()
                bb = self.book.best_bid()
                if side == "BUY" and ba:
                    price = ba[0]
                    self.book.submit_order(side=side, order_type="LIMIT", qty=size, price=price, now=now)
                elif side == "SELL" and bb:
                    price = bb[0]
                    self.book.submit_order(side=side, order_type="LIMIT", qty=size, price=price, now=now)
                else:
                    price = self.sample_limit_price(side)
                    self.book.submit_order(side=side, order_type="LIMIT", qty=size, price=price, now=now)
            else:
                price = self.sample_limit_price(side)
                is_ice = self.rng.random() < self.iceberg_prob
                if is_ice:
                    total = round(size * (2 + self.rng.integers(0, 6)), 2)
                    display = round(max(0.1, size), 2)
                    self.book.submit_order(side=side, order_type="LIMIT", qty=display, price=price, now=now,
                                           iceberg_total=total, iceberg_display=display)
                else:
                    self.book.submit_order(side=side, order_type="LIMIT", qty=size, price=price, now=now)

    def replenish_after_trades(self):
        recent_trades = self.book.trades_to_df().tail(20)
        for _, row in recent_trades.iterrows():
            if self.rng.random() < self.replenish_prob:
                price = row['price']
                side = "BUY" if self.rng.random() < 0.5 else "SELL"
                size = round(1.0 + abs(self.rng.normal(0, 2.0)), 2)
                p = round_price(price + (self.rng.integers(-2, 3) * self.tick), self.tick)
                self.book.submit_order(side=side, order_type="LIMIT", qty=size, price=p, now=datetime.now(UTC))

        # OFI after replenishments
        self.book.compute_ofi()

    def seed_depth_cshape(self, depth_levels: int = 20, base_qty: float = 50.0):
        mid = self.base_price
        decay = 0.12
        for lvl in range(1, depth_levels + 1):
            price_up = round_price(mid + lvl * self.tick, self.tick)
            price_dn = round_price(mid - lvl * self.tick, self.tick)
            qty_up = max(0.5, base_qty * np.exp(-decay * lvl) * (1.0 + 0.1 * self.rng.normal()))
            qty_dn = max(0.5, base_qty * np.exp(-decay * lvl) * (1.0 + 0.1 * self.rng.normal()))
            for _ in range(self.rng.integers(1, 4)):
                self.book.submit_order("SELL", "LIMIT", qty=round(float(qty_up / self.rng.integers(1,4)), 2),
                                       price=price_up, now=datetime.now(UTC))
            for _ in range(self.rng.integers(1, 4)):
                self.book.submit_order("BUY", "LIMIT", qty=round(float(qty_dn / self.rng.integers(1,4)), 2),
                                       price=price_dn, now=datetime.now(UTC))

    def step(self, now: datetime, seconds: float = 1.0):
        self.momentum = getattr(self, "momentum", 0.0) * 0.995 if hasattr(self, "momentum") else 0.0
        base_rate = self.mu
        effective_rate = base_rate + 0.1 * (self.intensity_buy + self.intensity_sell)
        n = self.rng.poisson(lam=max(0.5, effective_rate) * seconds)
        for _ in range(n):
            self.emit_order(now)
            if self.book.trade_log:
                last_trade = self.book.trade_log[-1]
                self.update_after_trade(last_trade)
        if self.rng.random() < 0.3:
            self.replenish_after_trades()

# -----------------------
# Aggregation & Plotting
# -----------------------

def aggregate_trades_to_candles(trades_df: pd.DataFrame, start: datetime, end: datetime, freq: str = '60S') -> pd.DataFrame:
    if trades_df.empty:
        idx = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
        empty = pd.DataFrame(index=idx, columns=["open","high","low","close","volume"]).fillna(0)
        return empty
    df = trades_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    ohlc = df['price'].resample(freq).ohlc()
    vol = df['qty'].resample(freq).sum().rename('volume')
    out = ohlc.join(vol).fillna(method='ffill').fillna(0)
    return out

def heikin_ashi(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        return candles
    ha = candles[['open','high','low','close']].copy()
    ha['ha_close'] = (ha['open'] + ha['high'] + ha['low'] + ha['close']) / 4
    ha['ha_open'] = 0.0
    ha.iloc[0, ha.columns.get_loc('ha_open')] = (ha.iloc[0]['open'] + ha.iloc[0]['close']) / 2.0
    for i in range(1, len(ha)):
        ha.iloc[i, ha.columns.get_loc('ha_open')] = (ha.iloc[i-1]['ha_open'] + ha.iloc[i-1]['ha_close']) / 2.0
    ha['ha_high'] = ha[['high','ha_open','ha_close']].max(axis=1)
    ha['ha_low'] = ha[['low','ha_open','ha_close']].min(axis=1)
    res = pd.DataFrame({
        'open': ha['ha_open'],
        'high': ha['ha_high'],
        'low': ha['ha_low'],
        'close': ha['ha_close'],
        'volume': candles['volume']
    }, index=candles.index)
    return res

def build_candlestick_figure(candles: pd.DataFrame, trades_df: pd.DataFrame, use_heikin: bool = True):
    fig = go.Figure()
    if candles.empty:
        return fig
    display = heikin_ashi(candles) if use_heikin else candles
    fig.add_trace(go.Candlestick(x=display.index,
                                 open=display['open'], high=display['high'],
                                 low=display['low'], close=display['close'],
                                 name='Candles (1m)'))
    if not trades_df.empty:
        df = trades_df.sort_values('timestamp').copy()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], mode='markers', name='Trades',
                                 marker={'size':5, 'opacity':0.6}))
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600)
    return fig

def build_l2_figure(book: OrderBook, depth: int = 12):
    snap = book.snapshot_l2(depth=depth)
    bids = snap['bids']
    asks = snap['asks']
    fig = go.Figure()
    if bids:
        fig.add_trace(go.Bar(x=[q for (_, q) in bids], y=[str(p) for (p, _) in bids], orientation='h',
                             name='Bids', marker_color='green'))
    if asks:
        fig.add_trace(go.Bar(x=[q for (_, q) in asks], y=[str(p) for (p, _) in asks], orientation='h',
                             name='Asks', marker_color='red'))
    fig.update_layout(template='plotly_dark', height=500)
    return fig

def build_ofi_figure(times_iso: List[str], values: List[float]):
    fig = go.Figure()
    if len(times_iso) > 0:
        # convert ISO strings to datetimes for plotting
        times = pd.to_datetime(times_iso)
        fig.add_trace(go.Scatter(x=times, y=values, mode='lines', line=dict(width=2), name='OFI'))
    # zero-line
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'))
    fig.update_layout(template='plotly_dark', height=180, margin=dict(l=40, r=20, t=10, b=40))
    return fig

# -----------------------
# Dash App & Callbacks
# -----------------------

def create_app(book: OrderBook, sim: MarketSimulator):
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.H4("Realistic Market Order Book Simulator", style={'color':'#ddd'}),
            html.Div(id='stats', style={'color':'#ddd'})
        ]),

        # Left column: candles + OFI
        html.Div([
            dcc.Graph(id='candles-graph'),
            dcc.Graph(id='ofi-graph')
        ], style={'width':'70%', 'display':'inline-block', 'verticalAlign':'top'}),

        # Right column: L2 + controls
        html.Div([
            dcc.Graph(id='l2-graph'),
            html.Div([html.Button("Pause/Resume", id='pause', n_clicks=0)], style={'paddingTop':'8px'})
        ], style={'width':'28%', 'display':'inline-block', 'paddingLeft':'8px'}),

        dcc.Interval(id='interval', interval=1000, n_intervals=0),

        # Store OFI history (ISO string timestamps) and paused state
        dcc.Store(id='ofi-store', data={'times': [], 'values': []}),
        dcc.Store(id='paused', data=False)
    ], style={'backgroundColor':'#111', 'padding':10})

    @app.callback(Output('paused', 'data'), Input('pause', 'n_clicks'), State('paused', 'data'))
    def toggle_pause(n_clicks, state):
        if n_clicks is None:
            return False
        return not state

    @app.callback(
        Output('candles-graph', 'figure'),
        Output('ofi-graph', 'figure'),
        Output('l2-graph', 'figure'),
        Output('stats', 'children'),
        Output('ofi-store', 'data'),
        Input('interval', 'n_intervals'),
        Input('paused', 'data'),
        State('ofi-store', 'data')
    )
    def update(n_intervals, paused, ofi_store):
        now_dt = datetime.now(UTC).replace(microsecond=0)
        if not paused:
            sim.step(now_dt, seconds=1.0)

        trades_df = book.trades_to_df()

        # candles over last 60 minutes
        end = now_dt + timedelta(seconds=1)
        start = now_dt - timedelta(minutes=60)
        candles = aggregate_trades_to_candles(trades_df, start, end)
        candle_fig = build_candlestick_figure(candles, trades_df, use_heikin=True)

        # L2 snapshot
        l2_fig = build_l2_figure(book, depth=12)

        # Stats
        bb = book.best_bid()
        ba = book.best_ask()
        spread = None
        mid = book.mid_price()
        if bb and ba:
            spread = round(ba[0] - bb[0], 8)

        # Update OFI history store (use ISO timestamps for safe JSON storage)
        ofi_val = round(book.ofi, 6)
        ts_list = ofi_store.get('times', []) if ofi_store else []
        vals_list = ofi_store.get('values', []) if ofi_store else []
        ts_list.append(now_dt.isoformat())
        vals_list.append(ofi_val)

        # Keep recent history (e.g., last 1200 points)
        ts_list = ts_list[-1200:]
        vals_list = vals_list[-1200:]

        ofi_fig = build_ofi_figure(ts_list, vals_list)

        stats = [
            html.Div(f"Best Bid: {bb}", style={'color':'#ccc'}),
            html.Div(f"Best Ask: {ba}", style={'color':'#ccc'}),
            html.Div(f"Spread: {spread}", style={'color':'#ccc'}),
            html.Div(f"Mid: {mid}", style={'color':'#ccc'}),
            html.Div(f"Last Trade: {book.last_trade_price}", style={'color':'#ccc'}),
            html.Div(f"Stress Metric: {round(sim.stress_metric(),6)}", style={'color':'#ccc'}),
            html.Div(f"OFI: {ofi_val}", style={'color':'#ccc'})
        ]

        return candle_fig, ofi_fig, l2_fig, stats, {'times': ts_list, 'values': vals_list}

    return app

# -----------------------
# Entrypoint
# -----------------------

def main():
    tick = 0.25
    book = OrderBook(tick_size=tick)
    sim = MarketSimulator(book=book, base_price=100.0, tick=tick, seed=42)

    # Seed book
    sim.seed_depth_cshape(depth_levels=18, base_qty=80.0)

    # Run app
    app = create_app(book, sim)
    app.run(debug=False, port=8050)

if __name__ == "__main__":
    main()
