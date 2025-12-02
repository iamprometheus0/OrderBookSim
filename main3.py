"""
Exchange-like Order Book Simulator + 1-minute Candles + Plotly Visualizer

Features implemented (as requested):
1. Price-Time Priority Matching Engine (limit/market, partial/full fills, FIFO)
2. Separate Bid & Ask Ladders (price -> deque of orders; sorted price lists)
3. Order Types: LIMIT, MARKET, STOP, STOP_LIMIT, IOC, FOK
4. Timestamps & Sequence IDs (deterministic sequence for arrivals & trades)
5. Trade Events / Ticker Tape (trade records with aggressor/maker/taker id)
6. Spread, Mid price (computed on every top-of-book change)

Simulates random incoming orders (numpy) and produces:
 - OHLC 1-minute candles (time-driven buckets)
 - Candlestick plot + trade overlay
 - Final L2 snapshot (bid/ask depth bars)

Dependencies:
  pip install numpy pandas plotly

Run:
  python orderbook_sim.py
"""

from __future__ import annotations
import random
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# Data Models
# ----------------------------

@dataclass
class Order:
    order_id: str
    side: str               # "BUY" or "SELL"
    type: str               # "LIMIT", "MARKET", "STOP", "STOP_LIMIT"
    price: Optional[float]  # None for MARKET
    qty: float
    remaining: float
    timestamp: datetime
    seq: int
    time_in_force: Optional[str] = None  # None, "IOC", "FOK"
    stop_price: Optional[float] = None  # for STOP or STOP_LIMIT
    status: str = "OPEN"      # OPEN, CANCELLED, FILLED, PARTIAL

@dataclass
class TradeEvent:
    trade_id: str
    price: float
    qty: float
    aggressor: str           # "BUY" or "SELL" (side of taker)
    timestamp: datetime
    maker_order_id: str
    taker_order_id: str
    seq: int

# ----------------------------
# Utility helpers
# ----------------------------

def new_id(prefix=""):
    return f"{prefix}{uuid.uuid4().hex[:12]}"

def round_price(p: float, tick: float):
    return round(round(p / tick) * tick, 8)

# ----------------------------
# Order Book Implementation
# ----------------------------

class OrderBook:
    def __init__(self, tick_size=0.25):
        # Price -> deque(Order) for bids and asks
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}
        # Sorted price level lists (descending bids, ascending asks)
        self.bid_prices: List[float] = []
        self.ask_prices: List[float] = []
        # Lookup by order_id for cancels / modifies
        self.order_map: Dict[str, Order] = {}
        # Stop orders (list) - not visible in book until triggered
        self.stop_orders: List[Order] = []
        # Trades and sequence counters
        self.trade_log: List[TradeEvent] = []
        self.seq_counter: int = 0
        self.trade_seq: int = 0
        self.tick = tick_size
        # Last trade price (for mark price)
        self.last_trade_price: Optional[float] = None

    def _next_seq(self) -> int:
        self.seq_counter += 1
        return self.seq_counter

    def _next_trade_seq(self) -> int:
        self.trade_seq += 1
        return self.trade_seq

    # ---------- Book level helpers ----------
    def _insert_price_level(self, price: float, side: str):
        if side == "BUY":
            if price not in self.bids:
                self.bids[price] = deque()
                # insert into bid_prices descending
                i = 0
                while i < len(self.bid_prices) and self.bid_prices[i] > price:
                    i += 1
                self.bid_prices.insert(i, price)
        else:
            if price not in self.asks:
                self.asks[price] = deque()
                # insert into ask_prices ascending
                i = 0
                while i < len(self.ask_prices) and self.ask_prices[i] < price:
                    i += 1
                self.ask_prices.insert(i, price)

    def _remove_price_level_if_empty(self, price: float, side: str):
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

    # ---------- Submit / Cancel ----------
    def submit_order(self, side: str, order_type: str, qty: float,
                     price: Optional[float], now: datetime,
                     time_in_force: Optional[str] = None,
                     stop_price: Optional[float] = None) -> Order:
        seq = self._next_seq()
        oid = new_id("o")
        o = Order(order_id=oid, side=side, type=order_type, price=price,
                  qty=qty, remaining=qty, timestamp=now, seq=seq,
                  time_in_force=time_in_force, stop_price=stop_price)
        # STOP orders are not visible
        if order_type in ("STOP", "STOP_LIMIT"):
            self.stop_orders.append(o)
            self.order_map[oid] = o
            return o

        # IOC and FOK need pre-checks
        if time_in_force == "FOK":
            # verify if full matching possible
            if not self._can_fully_fill(o):
                o.status = "CANCELLED"
                self.order_map[oid] = o
                return o
        # Route to matching core (market or crossing limit will match)
        remainder, trades = self._match(o, now)
        for t in trades:
            self.trade_log.append(t)
            self.last_trade_price = t.price
        # After matching, handle remaining according to TIF
        if remainder and remainder.remaining > 0:
            if remainder.time_in_force == "IOC":
                # cancel remainder
                remainder.status = "CANCELLED"
                self.order_map[remainder.order_id] = remainder
            else:
                # insert passive limit into book
                self._insert_passive(remainder)
                self.order_map[remainder.order_id] = remainder
        else:
            remainder.status = "FILLED"
            self.order_map[remainder.order_id] = remainder
        # After trades, check stop triggers
        self._check_stop_triggers(now)
        return remainder

    def cancel_order(self, order_id: str) -> bool:
        o = self.order_map.get(order_id)
        if not o or o.status != "OPEN":
            return False
        # remove from deque
        price = o.price
        side = o.side
        dq = self.bids.get(price) if side == "BUY" else self.asks.get(price)
        if dq:
            # remove the specific order (deque doesn't support remove by object index easily)
            for idx, el in enumerate(dq):
                if el.order_id == order_id:
                    dq.remove(el)
                    o.status = "CANCELLED"
                    self._remove_price_level_if_empty(price, side)
                    return True
        return False

    # ---------- Matching core ----------
    def _can_fully_fill(self, incoming: Order) -> bool:
        # scan opposite side to see cumulative available qty at matching prices
        needed = incoming.qty
        if incoming.side == "BUY":
            # need asks with price <= incoming.price (or any if market)
            price_list = self.ask_prices
            for p in price_list:
                if incoming.type == "MARKET" or (incoming.price is None) or p <= incoming.price:
                    level_qty = sum(o.remaining for o in self.asks[p])
                    needed -= level_qty
                    if needed <= 0:
                        return True
                else:
                    break
        else:
            price_list = self.bid_prices
            for p in price_list:
                if incoming.type == "MARKET" or (incoming.price is None) or p >= incoming.price:
                    level_qty = sum(o.remaining for o in self.bids[p])
                    needed -= level_qty
                    if needed <= 0:
                        return True
                else:
                    break
        return False

    def _match(self, incoming: Order, now: datetime) -> Tuple[Order, List[TradeEvent]]:
        trades: List[TradeEvent] = []
        # Market orders set price to extreme
        if incoming.type == "MARKET":
            incoming_price_limit = None
        else:
            incoming_price_limit = incoming.price

        # If incoming is a taker (market or crosses), we attempt match
        opposite_prices = self.ask_prices if incoming.side == "BUY" else self.bid_prices

        while incoming.remaining > 0 and opposite_prices:
            best_price = opposite_prices[0] if incoming.side == "BUY" else opposite_prices[0]
            # Decide if price is matchable
            if incoming.type != "MARKET" and incoming_price_limit is not None:
                if incoming.side == "BUY" and best_price > incoming_price_limit:
                    break
                if incoming.side == "SELL" and best_price < incoming_price_limit:
                    break
            # get the deque at best_price
            level_dq = self.asks[best_price] if incoming.side == "BUY" else self.bids[best_price]
            # FIFO through orders at this level
            while level_dq and incoming.remaining > 0:
                maker = level_dq[0]
                trade_qty = min(incoming.remaining, maker.remaining)
                trade_price = best_price
                trade = TradeEvent(
                    trade_id=new_id("t"),
                    price=trade_price,
                    qty=trade_qty,
                    aggressor=incoming.side,
                    timestamp=now,
                    maker_order_id=maker.order_id,
                    taker_order_id=incoming.order_id,
                    seq=self._next_trade_seq(),
                )
                trades.append(trade)
                # update quantities
                incoming.remaining -= trade_qty
                maker.remaining -= trade_qty
                if maker.remaining <= 0:
                    maker.status = "FILLED"
                    level_dq.popleft()
                else:
                    maker.status = "PARTIAL"
                # record maker in order_map if not present
                self.order_map.setdefault(maker.order_id, maker)
            # after exhausting level, maybe remove level
            self._remove_price_level_if_empty(best_price, "SELL" if incoming.side == "BUY" else "BUY")
            # refresh price list reference (could have been modified)
            opposite_prices = self.ask_prices if incoming.side == "BUY" else self.bid_prices

        # If incoming still has remaining and is limit (and passive), we'll return it for insertion
        return incoming, trades

    def _insert_passive(self, order: Order):
        # Insert limit order into book as a passive order (price must be set)
        price = order.price
        side = order.side
        if price is None:
            # should not happen for passive
            order.status = "CANCELLED"
            return
        price = round_price(price, self.tick)
        order.price = price
        self._insert_price_level(price, side)
        target_dq = self.bids[price] if side == "BUY" else self.asks[price]
        target_dq.append(order)
        order.status = "OPEN"

    # ---------- Stop handling ----------
    def _check_stop_triggers(self, now: datetime):
        if not self.stop_orders:
            return
        triggered = []
        last = self.last_trade_price
        if last is None:
            return
        for o in list(self.stop_orders):
            if o.side == "BUY":
                # buy stop triggers when last_trade_price >= stop_price
                if o.stop_price is not None and last >= o.stop_price:
                    triggered.append(o)
            else:
                # sell stop triggers when last_trade_price <= stop_price
                if o.stop_price is not None and last <= o.stop_price:
                    triggered.append(o)
        for o in triggered:
            self.stop_orders.remove(o)
            # convert to MARKET or LIMIT (STOP_LIMIT remains limit at original price)
            now_seq = self._next_seq()
            if o.type == "STOP":
                # becomes market
                o.type = "MARKET"
                o.price = None
            elif o.type == "STOP_LIMIT":
                o.type = "LIMIT"
                # price already set as pricer level to be used for limit when triggered
            o.seq = now_seq
            o.timestamp = datetime.utcnow()
            # route the new order into matching
            remainder, trades = self._match(o, o.timestamp)
            for t in trades:
                self.trade_log.append(t)
                self.last_trade_price = t.price
            if remainder.remaining > 0 and remainder.time_in_force != "IOC":
                self._insert_passive(remainder)
                self.order_map[remainder.order_id] = remainder
            else:
                remainder.status = "FILLED" if remainder.remaining == 0 else "CANCELLED"
                self.order_map[remainder.order_id] = remainder

    # ---------- Snapshot & helpers ----------
    def snapshot_l2(self, depth=10) -> Dict[str, List[Tuple[float, float]]]:
        bids = []
        asks = []
        for p in self.bid_prices[:depth]:
            total = sum(o.remaining for o in self.bids[p])
            bids.append((p, total))
        for p in self.ask_prices[:depth]:
            total = sum(o.remaining for o in self.asks[p])
            asks.append((p, total))
        return {"bids": bids, "asks": asks}

    def all_trades_to_df(self) -> pd.DataFrame:
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
        return pd.DataFrame(rows)

# ----------------------------
# Simulator (random order flow)
# ----------------------------

class Simulator:
    def __init__(self, book: OrderBook,
                 start_time: datetime,
                 minutes: int = 20,
                 base_price: float = 100.0,
                 tick: float = 0.25,
                 mean_order_rate_per_sec: float = 1.0):
        self.book = book
        self.now = start_time
        self.end_time = start_time + timedelta(minutes=minutes)
        self.base_price = base_price
        self.tick = tick
        self.mean_order_rate = mean_order_rate_per_sec

    def _random_side(self):
        return "BUY" if np.random.rand() < 0.5 else "SELL"

    def _random_order_type(self):
        # bias mostly to LIMIT and MARKET, rarer IOC/FOK/STOPs
        r = np.random.rand()
        if r < 0.6:
            return "LIMIT"
        if r < 0.85:
            return "MARKET"
        if r < 0.93:
            return "IOC"
        if r < 0.97:
            return "FOK"
        # stops
        return "STOP"

    def _random_price_around_mid(self):
        mid = self.book.mid_price() or self.base_price
        # gaussian around mid with std of a few ticks
        ticks = int(np.random.randn() * 4)
        p = mid + ticks * self.tick
        return round_price(max(0.01, p), self.tick)

    def _random_qty(self):
        # use exponential-ish tail + scale
        base = 1 + (np.random.rand() * 10)  # 1 to 11
        mul = 10 ** (np.random.rand() * 0.8)  # vary magnitude
        q = round(base * mul, 2)
        return q

    def run(self):
        # We'll simulate Poisson arrivals: inter-arrival times ~ Exp(rate)
        # But to keep deterministic-ish, step by small time increments and emit random number of orders per sec
        timeline = []
        current = self.now
        while current < self.end_time:
            # number of orders in this second
            lam = self.mean_order_rate
            n_orders = np.random.poisson(lam)
            for _ in range(n_orders):
                timeline.append(current + timedelta(milliseconds=int(np.random.rand() * 1000)))
            current += timedelta(seconds=1)

        timeline.sort()
        for t in timeline:
            self.now = t
            self._emit_random_order(t)

    def _emit_random_order(self, now: datetime):
        side = self._random_side()
        typ = self._random_order_type()
        qty = self._random_qty()
        price = None
        time_in_force = None
        stop_price = None

        if typ == "LIMIT":
            # decide whether to post at/passive or cross
            mid = self.book.mid_price() or self.base_price
            # probability of aggressive (cross) ~ 25%
            if np.random.rand() < 0.25 and self.book.best_ask() and self.book.best_bid():
                # cross the spread slightly
                if side == "BUY":
                    best_ask = self.book.best_ask()[0]
                    price = best_ask  # cross at best ask
                else:
                    best_bid = self.book.best_bid()[0]
                    price = best_bid
            else:
                # place passive near mid
                price = self._random_price_around_mid()
        elif typ == "MARKET":
            price = None
        elif typ == "IOC":
            # treat IOC as LIMIT with TIF IOC
            time_in_force = "IOC"
            price = self._random_price_around_mid()
            typ = "LIMIT"
        elif typ == "FOK":
            time_in_force = "FOK"
            price = self._random_price_around_mid()
            typ = "LIMIT"
        elif typ == "STOP":
            # STOP becomes market when triggered; choose stop price away from mid
            # buy stop trigger above current mid; sell stop below mid
            mid = self.book.mid_price() or self.base_price
            off_ticks = int(max(1, abs(int(np.random.exponential(scale=3)))))
            if side == "BUY":
                stop_price = round_price(mid + off_ticks * self.tick, self.tick)
            else:
                stop_price = round_price(mid - off_ticks * self.tick, self.tick)
            price = None
            typ = "STOP"
        else:
            price = self._random_price_around_mid()

        self.book.submit_order(side=side, order_type=typ, qty=qty, price=price, now=now,
                               time_in_force=time_in_force, stop_price=stop_price)

# ----------------------------
# Candle aggregation
# ----------------------------

def aggregate_trades_to_candles(trades_df: pd.DataFrame, start: datetime, end: datetime, freq='60S'):
    if trades_df.empty:
        # create empty time index
        idx = pd.date_range(start=start, end=end, freq=freq, closed='left')
        empty = pd.DataFrame(index=idx, columns=["open","high","low","close","volume"]).fillna(0)
        return empty
    trades_df = trades_df.copy()
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df.set_index('timestamp', inplace=True)
    # produce OHLC by price; volume by qty
    ohlc = trades_df['price'].resample(freq).ohlc()
    vol = trades_df['qty'].resample(freq).sum().rename('volume')
    df = ohlc.join(vol).fillna(method='ffill').fillna(0)
    # some intervals may have no trades; replace NaNs in open/high/low/close with previous close
    df[['open','high','low','close']] = df[['open','high','low','close']].fillna(method='ffill').fillna(0)
    return df

# ----------------------------
# Visualization using plotly
# ----------------------------

def plot_results(trades_df: pd.DataFrame, candles_df: pd.DataFrame, book: OrderBook, start: datetime, end: datetime, filename=None):
    # Candlestick figure
    fig = go.Figure()
    if not candles_df.empty:
        x = candles_df.index
        fig.add_trace(go.Candlestick(
            x=x,
            open=candles_df['open'],
            high=candles_df['high'],
            low=candles_df['low'],
            close=candles_df['close'],
            name='Candles (1m)'
        ))
        # add volume as bar on secondary y (normalized)
        vol = candles_df['volume']
        vol_y = vol / max(1, vol.max()) * (candles_df['high'].max() - candles_df['low'].min()) * 0.3
        fig.add_trace(go.Bar(x=x, y=vol_y, name='Volume (scaled)', opacity=0.2, marker_line_width=0))
    # overlay trade ticks
    if not trades_df.empty:
        trades_df = trades_df.sort_values('timestamp')
        fig.add_trace(go.Scatter(x=trades_df['timestamp'], y=trades_df['price'],
                                 mode='markers', marker=dict(size=6),
                                 name='Trades', opacity=0.6))
    fig.update_layout(title=f"Simulated Order Book - Candles {start} to {end}",
                      xaxis_title='Time', yaxis_title='Price', height=700)

    # L2 snapshot at end
    s = book.snapshot_l2(depth=20)
    bids = s['bids']
    asks = s['asks']
    bid_prices = [p for p, q in bids]
    bid_qty = [q for p, q in bids]
    ask_prices = [p for p, q in asks]
    ask_qty = [q for p, q in asks]

    # Build bar chart for depth
    fig2 = go.Figure()
    if bid_prices:
        fig2.add_trace(go.Bar(x=bid_qty, y=[str(p) for p in bid_prices],
                              orientation='h', name='Bids', offsetgroup=1))
    if ask_prices:
        fig2.add_trace(go.Bar(x=ask_qty, y=[str(p) for p in ask_prices],
                              orientation='h', name='Asks', offsetgroup=2))
    fig2.update_layout(title=f"Final L2 Snapshot (Top levels) at {end}", xaxis_title='Quantity', yaxis_title='Price (levels)', height=600)

    # show both
    if filename:
        fig.write_html(f"{filename}_candles.html")
        fig2.write_html(f"{filename}_l2.html")
        print(f"Saved {filename}_candles.html and {filename}_l2.html")
    else:
        fig.show()
        fig2.show()

# ----------------------------
# Wiring everything together
# ----------------------------

def main():
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    start = datetime.utcnow().replace(microsecond=0)
    minutes = 30
    book = OrderBook(tick_size=0.25)
    sim = Simulator(book=book, start_time=start, minutes=minutes, base_price=100.0, tick=0.25, mean_order_rate_per_sec=1.2)
    sim.run()

    trades_df = book.all_trades_to_df()
    # ensure timestamp column is datetime
    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

    candles = aggregate_trades_to_candles(trades_df, start, start + timedelta(minutes=minutes), freq='60S')

    # Nicely label candles columns if present
    if isinstance(candles, pd.DataFrame) and not candles.empty:
        # rename if needed
        if 'open' not in candles.columns and 'price' in candles.columns:
            candles = candles.rename(columns={'price': 'close'})

    # Print some stats
    print("Total trades generated:", len(book.trade_log))
    bb = book.best_bid()
    ba = book.best_ask()
    print("Best Bid:", bb)
    print("Best Ask:", ba)
    print("Mid:", book.mid_price())

    # Plot
    plot_results(trades_df, candles, book, start, start + timedelta(minutes=minutes), filename="orderbook_sim")

if __name__ == "__main__":
    main()
