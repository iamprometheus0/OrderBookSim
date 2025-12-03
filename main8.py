"""
Realistic Market Order-Book Simulator (Dash) — with OFI vs Stress Scatter Plot
- Single-file, minimal refactor.
- Features: OFI, OFI timeseries, OFI vs Stress scatter, Heikin-Ashi candles,
  L2 depth, alpha & decay sliders, replenishment, iceberg orders, stop orders.
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
# Helpers & Models
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
    status: str = "OPEN"
    iceberg_total: Optional[float] = None
    iceberg_display: Optional[float] = None


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


# -----------------------
# Order Book
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

        # OFI tracking: previous top-of-book snapshot and current OFI
        self.prev_snapshot: Optional[Tuple[float, float]] = None
        self.ofi: float = 0.0

    # ---- sequence helpers ----
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _next_trade_seq(self) -> int:
        self._trade_seq += 1
        return self._trade_seq

    # ---- price-level helpers ----
    def _insert_level(self, price: float, side: str):
        if side == "BUY":
            if price not in self.bids:
                self.bids[price] = deque()
                # keep bid_prices descending
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

    # ---- top-of-book / mid ----
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

    # ---- submit / cancel ----
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
        # Stop orders sit in stop_orders
        if order_type in ("STOP", "STOP_LIMIT"):
            self.stop_orders.append(o)
            self.order_map[oid] = o
            self.compute_ofi()
            return o

        # Route to matching core
        remainder, trades = self._match(o, now)
        for t in trades:
            self.trade_log.append(t)
            self.last_trade_price = t.price
            self.last_trade_time = t.timestamp

        # If remainder exists, insert passive if allowed
        if remainder.remaining > 0:
            if remainder.tif == "IOC":
                remainder.status = "CANCELLED"
            else:
                self._insert_passive(remainder)
        else:
            remainder.status = "FILLED"
        self.order_map[oid] = remainder

        # After trades, check stops & triggers
        self._check_stop_triggers(now)

        # compute OFI after the state changes caused by this submission
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
                    # OFI changes because book volume removed
                    self.compute_ofi()
                    return True
        return False

    # ---- matching core (FIFO / price-time) ----
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
                # update
                incoming.remaining -= trade_qty
                maker.remaining -= trade_qty
                if maker.remaining <= 0:
                    maker.status = "FILLED"
                    level_dq.popleft()
                else:
                    maker.status = "PARTIAL"
                self.order_map.setdefault(maker.order_id, maker)
            # after level exhausted
            self._remove_level_if_empty(best_price, "SELL" if incoming.side == "BUY" else "BUY")
        return incoming, trades

    def _insert_passive(self, order: Order):
        # Insert limit order at level with FIFO
        if order.price is None:
            order.status = "CANCELLED"
            return
        p = round_price(order.price, self.tick)
        order.price = p
        self._insert_level(p, order.side)
        dq = self.bids[p] if order.side == "BUY" else self.asks[p]
        dq.append(order)
        order.status = "OPEN"

    # ---- stop triggers ----
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
            # convert stop to market or to limit
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

        # OFI may have changed due to triggered orders being converted/executed
        self.compute_ofi()

    # ---- snapshots / export ----
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

    # ---- OFI computation ----
    def compute_ofi(self, depth: int = 1) -> float:
        """
        Compute Order Flow Imbalance (OFI) using top-of-book changes.
        OFI = (Bid added - Bid removed) - (Ask added - Ask removed)
        Only uses top depth levels (default 1).
        Updates self.ofi and self.prev_snapshot.
        """
        snap = self.snapshot_l2(depth=depth)
        bids = snap.get("bids", [])
        asks = snap.get("asks", [])

        bid_vol = bids[0][1] if bids else 0.0
        ask_vol = asks[0][1] if asks else 0.0

        if self.prev_snapshot is None:
            # initialize snapshot, keep OFI as-is (likely 0)
            self.prev_snapshot = (bid_vol, ask_vol)
            return self.ofi

        prev_bid, prev_ask = self.prev_snapshot

        bid_added = max(0.0, bid_vol - prev_bid)
        bid_removed = max(0.0, prev_bid - bid_vol)
        ask_added = max(0.0, ask_vol - prev_ask)
        ask_removed = max(0.0, prev_ask - ask_vol)

        self.ofi = (bid_added - bid_removed) - (ask_added - ask_removed)

        # update snapshot
        self.prev_snapshot = (bid_vol, ask_vol)
        return self.ofi


# -----------------------
# Market-realistic Simulator
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
        self.mu = 0.9                 # baseline rate (per second) total
        self.alpha = 0.6              # self-excitation
        self.decay = 1.5              # decay rate per second
        self.intensity_buy = 0.5
        self.intensity_sell = 0.5
        # momentum memory (for AR(1)-like autocorrelation)
        self.momentum = 0.0
        # sliders will control these
        self.mom_decay = 0.9
        self.mom_alpha = 0.25
        # volatility tracking (realized vol of last N trades' log returns)
        self.price_window = deque(maxlen=200)
        # replenishment parameters
        self.replenish_prob = 0.6     # probability to refill after a fill
        self.iceberg_prob = 0.08      # chance an added passive is iceberg
        # price impact scaling
        self.base_impact = 0.2
        self.stress_scale = 2.0

    # ---- utility metrics ----
    def update_after_trade(self, trade: TradeEvent):
        # update momentum: buy = +1, sell = -1
        sign = 1 if trade.aggressor == "BUY" else -1
        self.momentum = self.mom_alpha * sign + self.mom_decay * self.momentum
        # update intensities (Hawkes-like): bump the aggressive side
        if trade.aggressor == "BUY":
            self.intensity_buy += self.alpha
        else:
            self.intensity_sell += self.alpha
        # store price for vol calc
        self.price_window.append(trade.price)
        # small decay every update
        self.intensity_buy = max(0.01, self.intensity_buy * np.exp(-1.0 / self.decay))
        self.intensity_sell = max(0.01, self.intensity_sell * np.exp(-1.0 / self.decay))

    def realized_volatility(self, lookback: int = 60) -> float:
        # compute realized volatility (std of log returns) on last lookback trades
        prices = list(self.price_window)[-lookback:]
        if len(prices) < 2:
            return 0.0
        logrets = np.diff(np.log(np.array(prices) + 1e-12))
        return float(np.std(logrets, ddof=0))

    def stress_metric(self) -> float:
        # combine vol and momentum to produce a stress value >=0
        vol = self.realized_volatility(60)
        mom = abs(self.momentum)
        return vol * self.stress_scale + mom

    # ---- choose side with autocorrelation ----
    def choose_side(self) -> str:
        # base buy prob adjusted by momentum (AR1-like)
        base_buy_prob = 0.5 + 0.3 * np.tanh(self.momentum)  # momentum pushes p
        # also modify by Hawkes intensities
        total_intensity = self.intensity_buy + self.intensity_sell + 1e-12
        hawkes_buy = self.intensity_buy / total_intensity
        p_buy = 0.6 * base_buy_prob + 0.4 * hawkes_buy
        return "BUY" if self.rng.random() < p_buy else "SELL"

    # ---- produce order sizes responsive to volatility ----
    def sample_order_size(self, side: str, is_market: bool = False) -> float:
        vol = self.realized_volatility(60)
        # baseline size distribution
        if self.rng.random() < 0.9:
            size = self.rng.exponential(2.5) + 0.1
        else:
            size = self.rng.exponential(12) + 1.0
        # vol sensitive: higher vol increases market order sizes and reduces passive sizes
        if vol > 0.0005:
            # scale market orders more aggressively
            if is_market:
                size *= (1.0 + min(6.0, vol * 300))
            else:
                # passive limit sizes shrink under stress
                size *= max(0.3, 1.0 - min(0.9, vol * 200))
        return round(float(size), 2)

    # ---- price selection for limit orders (near mid) ----
    def sample_limit_price(self, side: str) -> float:
        mid = self.book.mid_price() or self.base_price
        # jitter in ticks, narrower when calm, wider when stressed
        stress = self.stress_metric()
        std_ticks = max(1, int(1 + stress * 4))
        ticks = int(round(self.rng.normal(0, std_ticks)))
        return round_price(mid + (ticks * self.tick), self.tick)

    # ---- price-impact: how deep to walk the book ----
    def compute_walk_depth(self, base_qty: float) -> int:
        # deeper walk when stress is high or order is large relative to top size
        stress = self.stress_metric()
        # base depth (levels)
        base_depth = 1 if stress < 0.5 else 1 + int(stress * 3)
        # scale with size
        extra = int(min(10, base_qty / 2.0))
        return base_depth + extra

    # ---- emit an order with price impact behaviour ----
    def emit_order(self, now: datetime):
        side = self.choose_side()
        # decide market vs limit with stress sensitivity
        stress = self.stress_metric()
        prob_market = 0.12 + 0.25 * min(1.0, stress)  # more market orders during stress
        is_market = self.rng.random() < prob_market
        size = self.sample_order_size(side, is_market=is_market)
        if is_market:
            # market order should walk the book deeper depending on stress & size
            depth = self.compute_walk_depth(size)
            # submit a MARKET order with qty == size (matching core will walk)
            self.book.submit_order(side=side, order_type="MARKET", qty=size, price=None, now=now)
        else:
            # limit order: choose price near mid, but sometimes cross (simulate taker-on-limit)
            if self.rng.random() < (0.18 + 0.15 * min(1.0, stress)):
                # cross: submit at opposite best to take liquidity
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
                # post passive near mid; sometimes make it iceberg
                price = self.sample_limit_price(side)
                is_ice = self.rng.random() < self.iceberg_prob
                if is_ice:
                    total = round(size * (2 + self.rng.integers(0, 6)), 2)
                    display = round(max(0.1, size), 2)
                    # create displayed slice now, while total is larger
                    self.book.submit_order(side=side, order_type="LIMIT", qty=display, price=price, now=now,
                                           iceberg_total=total, iceberg_display=display)
                else:
                    self.book.submit_order(side=side, order_type="LIMIT", qty=size, price=price, now=now)

    # ---- replenishment after fills (refill behaviour) ----
    def replenish_after_trades(self):
        # examine recent trades and probabilistically refill levels that were hit
        recent_trades = self.book.trades_to_df().tail(20)
        for _, row in recent_trades.iterrows():
            if self.rng.random() < self.replenish_prob:
                price = row['price']
                side = "BUY" if self.rng.random() < 0.5 else "SELL"
                size = round(1.0 + abs(self.rng.normal(0, 2.0)), 2)
                p = round_price(price + (self.rng.integers(-2, 3) * self.tick), self.tick)
                self.book.submit_order(side=side, order_type="LIMIT", qty=size, price=p, now=datetime.now(UTC))

        # ensure OFI updated after replenishment-based submissions
        self.book.compute_ofi()

    # ---- seed book with C-shaped depth curve ----
    def seed_depth_cshape(self, depth_levels: int = 20, base_qty: float = 50.0):
        """Create a C-shaped depth curve: large near best, decays exponentially away from mid."""
        mid = self.base_price
        decay = 0.12  # controls how fast qty decays with ticks
        now = datetime.now(UTC)
        # create symmetric levels around mid
        for lvl in range(1, depth_levels + 1):
            price_up = round_price(mid + lvl * self.tick, self.tick)
            price_dn = round_price(mid - lvl * self.tick, self.tick)
            # quantity follows C-shape: heavy near best (lvl small), decays exponentially
            qty_up = max(0.5, base_qty * np.exp(-decay * lvl) * (1.0 + 0.1 * self.rng.normal()))
            qty_dn = max(0.5, base_qty * np.exp(-decay * lvl) * (1.0 + 0.1 * self.rng.normal()))
            # post multiple small orders at each level to create FIFO queue
            for _ in range(self.rng.integers(1, 4)):
                self.book.submit_order("SELL", "LIMIT", qty=round(float(qty_up / self.rng.integers(1,4)), 2),
                                       price=price_up, now=now)
            for _ in range(self.rng.integers(1, 4)):
                self.book.submit_order("BUY", "LIMIT", qty=round(float(qty_dn / self.rng.integers(1,4)), 2),
                                       price=price_dn, now=now)

    # ---- step function to be called each second ----
    def step(self, now: datetime, seconds: float = 1.0):
        # decay momentum slightly each second (already handled in update_after_trade but ensure bounded)
        self.momentum = getattr(self, "momentum", 0.0) * 0.995 if hasattr(self, "momentum") else 0.0

        # sample number of events this second from Poisson with baseline + intensities
        base_rate = self.mu
        # combine intensities roughly
        effective_rate = base_rate + 0.1 * (self.intensity_buy + self.intensity_sell)
        n = self.rng.poisson(lam=max(0.5, effective_rate) * seconds)
        for _ in range(n):
            self.emit_order(now)
            # after emitting, incorporate any trades into momentum/intensity updates
            # read recent trades generated by book and update simulator state
            if self.book.trade_log:
                last_trade = self.book.trade_log[-1]
                self.update_after_trade(last_trade)
        # occasionally replenish
        if self.rng.random() < 0.3:
            self.replenish_after_trades()


# -----------------------
# Candle aggregation & plotting (unchanged)
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

    # Binance-like smooth Heikin-Ashi styling
    up_color = 'rgba(0,200,120,0.75)'   # greenish
    down_color = 'rgba(255,80,80,0.80)' # reddish

    fig.add_trace(go.Candlestick(
        x=display.index,
        open=display['open'], high=display['high'],
        low=display['low'], close=display['close'],
        name='Candle',
        increasing=dict(line=dict(color=up_color)),
        decreasing=dict(line=dict(color=down_color)),
        showlegend=True
    ))

    # Trade markers: subtle, small, semi-transparent
    if not trades_df.empty:
        df = trades_df.sort_values('timestamp').copy()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['price'],
            mode='markers',
            name='Trades',
            marker=dict(size=4, opacity=0.45, color='rgba(255,165,0,0.6)'),
            showlegend=True
        ))

    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600,
                      legend=dict(orientation='h', yanchor='top', y=0.99, xanchor='left', x=0.01))
    return fig

def build_l2_figure(book: OrderBook, depth: int = 12):
    snap = book.snapshot_l2(depth=depth)
    bids = snap['bids']
    asks = snap['asks']
    fig = go.Figure()
    if bids:
        fig.add_trace(go.Bar(x=[q for (_, q) in bids], y=[str(p) for (p, _) in bids],
                             orientation='h', name='Bids', marker_color='green'))
    if asks:
        fig.add_trace(go.Bar(x=[q for (_, q) in asks], y=[str(p) for (p, _) in asks],
                             orientation='h', name='Asks', marker_color='red'))
    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=60, r=20, t=20, b=20))
    return fig

def build_ofi_figure(times_iso: List[str], values: List[float]):
    fig = go.Figure()
    if len(times_iso) > 0:
        times = pd.to_datetime(times_iso)
        fig.add_trace(go.Scatter(x=times, y=values, mode='lines', line=dict(width=2), name='OFI'))
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'))
    fig.update_layout(template='plotly_dark', height=180, margin=dict(l=40, r=20, t=10, b=40))
    return fig

def build_regime_figure(times_iso: List[str], ofi_vals: List[float], stress_vals: List[float]):
    """
    OFI vs Stress scatter: x=OFI, y=Stress.
    Visual: faded older points + highlighted latest point.
    """
    fig = go.Figure()
    if ofi_vals and stress_vals and len(ofi_vals) == len(stress_vals):
        n = len(ofi_vals)
        if n > 1:
            # historical points (all except last) - subtle color
            fig.add_trace(go.Scatter(
                x=ofi_vals[:-1],
                y=stress_vals[:-1],
                mode='markers',
                marker=dict(size=6, color='rgba(180,180,180,0.25)'),
                hoverinfo='skip',
                name='History'
            ))
        # latest point
        fig.add_trace(go.Scatter(
            x=[ofi_vals[-1]],
            y=[stress_vals[-1]],
            mode='markers',
            marker=dict(size=9, color='rgba(0,200,120,0.95)' if ofi_vals[-1] >= 0 else 'rgba(255,80,80,0.95)', line=dict(width=1, color='white')),
            hovertemplate="time=%{text}<br>OFI=%{x:.4f}<br>Stress=%{y:.4f}",
            text=[times_iso[-1]],
            name='Latest'
        ))
    fig.update_layout(template='plotly_dark', height=300, xaxis_title='OFI', yaxis_title='Stress',
                      margin=dict(l=40, r=20, t=20, b=30))
    return fig


# -----------------------
# Dash App wiring (Interval-driven)
# -----------------------

def create_app(book: OrderBook, sim: MarketSimulator):
    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.H4("Realistic Market Order Book Simulator", style={'color':'#ddd'}),
            html.Div(id='stats', style={'color':'#ddd'})
        ]),

        html.Div([
            dcc.Graph(id='candles-graph'),
            dcc.Graph(id='ofi-graph'),
            dcc.Graph(id='regime-graph')   # NEW: OFI vs Stress scatter
        ], style={'width':'70%', 'display':'inline-block', 'verticalAlign':'top'}),

        html.Div([
            dcc.Graph(id='l2-graph'),
            html.Div([
                html.Button("Pause/Resume", id='pause', n_clicks=0),
            ], style={'paddingTop':'8px'}),
            html.Div(style={'paddingTop':'12px'}),
            html.Label("Alpha (Momentum Strength)", style={'color':'#ccc'}),
            dcc.Slider(id='alpha-slider', min=0.0, max=1.0, step=0.01, value=sim.mom_alpha),
            html.Br(),
            html.Label("Decay (Momentum Forgetting Rate)", style={'color':'#ccc'}),
            dcc.Slider(id='decay-slider', min=0.5, max=0.999, step=0.001, value=sim.mom_decay),
        ], style={'width':'28%', 'display':'inline-block', 'paddingLeft':'8px', 'verticalAlign':'top'}),

        dcc.Interval(id='interval', interval=1000, n_intervals=0),
        dcc.Store(id='ofi-store', data={'times': [], 'values': []}),
        dcc.Store(id='regime-store', data={'times': [], 'ofi': [], 'stress': []}),
        dcc.Store(id='paused', data=False)
    ], style={'backgroundColor':'#111', 'padding':10})

    @app.callback(Output('paused', 'data'), Input('pause', 'n_clicks'), State('paused', 'data'))
    def toggle_pause(n_clicks, state):
        # toggle on every click
        if n_clicks is None:
            return False
        return not state

    @app.callback(
        Output('candles-graph', 'figure'),
        Output('ofi-graph', 'figure'),
        Output('regime-graph', 'figure'),
        Output('l2-graph', 'figure'),
        Output('stats', 'children'),
        Output('ofi-store', 'data'),
        Output('regime-store', 'data'),
        Input('interval', 'n_intervals'),
        Input('paused', 'data'),
        Input('alpha-slider', 'value'),
        Input('decay-slider', 'value'),
        State('ofi-store', 'data'),
        State('regime-store', 'data')
    )
    def update(n_intervals, paused, alpha_val, decay_val, ofi_store, regime_store):
        # Ensure stores are valid
        if ofi_store is None:
            ofi_store = {'times': [], 'values': []}
        if regime_store is None:
            regime_store = {'times': [], 'ofi': [], 'stress': []}

        # Update simulator momentum params from sliders
        sim.mom_alpha = float(alpha_val)
        sim.mom_decay = float(decay_val)

        now = datetime.now(UTC).replace(microsecond=0)
        if not paused:
            # run simulation step(s)
            sim.step(now, seconds=1.0)

        # build visuals
        trades_df = book.trades_to_df()
        # 60-minute sliding window for candles
        end = now + timedelta(seconds=1)
        start = now - timedelta(minutes=60)
        candles = aggregate_trades_to_candles(trades_df, start, end)
        candle_fig = build_candlestick_figure(candles, trades_df, use_heikin=True)
        l2_fig = build_l2_figure(book, depth=12)

        # Update OFI store safely (times in ISO for JSON)
        ts = ofi_store.get('times', [])
        vals = ofi_store.get('values', [])
        ts.append(now.isoformat())
        vals.append(float(book.ofi))
        ts = ts[-1200:]
        vals = vals[-1200:]
        ofi_fig = build_ofi_figure(ts, vals)

        # Update regime store: OFI vs Stress
        r_ts = regime_store.get('times', [])
        r_ofi = regime_store.get('ofi', [])
        r_stress = regime_store.get('stress', [])
        current_stress = float(sim.stress_metric())
        r_ts.append(now.isoformat())
        r_ofi.append(float(book.ofi))
        r_stress.append(current_stress)
        r_ts = r_ts[-1200:]
        r_ofi = r_ofi[-1200:]
        r_stress = r_stress[-1200:]
        regime_fig = build_regime_figure(r_ts, r_ofi, r_stress)

        bb = book.best_bid()
        ba = book.best_ask()
        spread = None
        mid = book.mid_price()
        if bb and ba:
            spread = round(ba[0] - bb[0], 8)
        stats = [
            html.Div(f"Best Bid: {bb}", style={'color':'#ccc'}),
            html.Div(f"Best Ask: {ba}", style={'color':'#ccc'}),
            html.Div(f"Spread: {spread}", style={'color':'#ccc'}),
            html.Div(f"Mid: {mid}", style={'color':'#ccc'}),
            html.Div(f"Last Trade: {book.last_trade_price}", style={'color':'#ccc'}),
            html.Div(f"Stress Metric: {round(sim.stress_metric(),6)}", style={'color':'#ccc'}),
            html.Div(f"OFI: {round(book.ofi,4)}", style={'color':'#ccc'}),
            html.Div(f"Alpha: {round(sim.mom_alpha,3)}", style={'color':'#ccc'}),
            html.Div(f"Decay: {round(sim.mom_decay,3)}", style={'color':'#ccc'})
        ]

        return candle_fig, ofi_fig, regime_fig, l2_fig, stats, {'times': ts, 'values': vals}, {'times': r_ts, 'ofi': r_ofi, 'stress': r_stress}

    return app


# -----------------------
# Entrypoint
# -----------------------

def main():
    tick = 0.25
    book = OrderBook(tick_size=tick)
    sim = MarketSimulator(book=book, base_price=100.0, tick=tick, seed=42)

    # seed depth with C-shaped curve
    sim.seed_depth_cshape(depth_levels=18, base_qty=80.0)

    app = create_app(book, sim)
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()
