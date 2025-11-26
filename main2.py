"""
Order Book + Time-based Candles Live Simulator & Visualizer

Run:
    python3 main.py

Dependencies:
    pip install matplotlib mplfinance numpy

By default the simulation is accelerated so 1 real second == 1 simulated minute.
Set TIME_ACCEL = 1.0 for real-time (1 real second == 1 simulated second).
"""

import random
import time
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# ---------------------- CONFIG ----------------------
TICK_SIZE = 0.05                # minimum price increment
INITIAL_MID = 100.0             # starting mid price
SPREAD_START = 0.2              # initial spread in price units
MAX_PRICE_MOVE = 0.5            # when generating raw price
TICKS_PER_EVENT = (1, 3)        # when an order placed, how many match attempts (small random)
NUM_PRICE_LEVELS = 10           # how many levels to show on orderbook depth (per side)

CANDLE_INTERVAL_SECONDS = 60.0  # candle length in simulated seconds (1 minute)
TIME_ACCEL = 60.0               # simulation speed: 1 real second == TIME_ACCEL simulated seconds
                                 # -> default 60 makes 1 real second equal 1 simulated minute (fast demo)

EVENTS_PER_REAL_SECOND = 200    # how many order events the simulator attempts per real second (controls activity)
SIM_DURATION_REAL_SECONDS = 30  # how long the demo runs (real seconds). Set None to run indefinitely.

# Visualization refresh
PLOT_REFRESH_INTERVAL = 0.3     # seconds between plot updates (real seconds)

# ---------------------- ORDER BOOK DATA STRUCTURES ----------------------
# We'll store orders aggregated by price level (simpler & faster for visualization).
# buy_book and sell_book map price -> total_qty
buy_book = defaultdict(int)
sell_book = defaultdict(int)

# For deterministic ordering of levels used to display top N
def sorted_buy_levels():
    return sorted(buy_book.items(), key=lambda x: -x[0])

def sorted_sell_levels():
    return sorted(sell_book.items(), key=lambda x: x[0])

# Trade ticks: each entry is (sim_timestamp, price, qty)
trade_ticks = []

# Candle storage: deque of (timestamp, open, high, low, close, volume)
candles = deque()

# Maintain current candle being built
current_candle = None  # dict with keys: start_ts, open, high, low, close, volume

# Simulation clock (simulated seconds since epoch-like base)
sim_start_real = time.time()
sim_base = datetime.now()   # use this as origin for simulated timestamps

def now_sim_seconds():
    """Return current simulated time in seconds (float) since sim_base."""
    real_elapsed = time.time() - sim_start_real
    return real_elapsed * TIME_ACCEL

def now_sim_datetime():
    """Return simulated datetime (sim_base + simulated elapsed seconds)."""
    return sim_base + timedelta(seconds=now_sim_seconds())

# ---------------------- UTILITIES ----------------------
def snap_price(raw):
    """Round raw price to nearest tick size."""
    return round(raw / TICK_SIZE) * TICK_SIZE

def top_of_book():
    """Return (best_bid_price, best_ask_price) or (None, None) if absent."""
    bids = sorted_buy_levels()
    asks = sorted_sell_levels()
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    return best_bid, best_ask

# ---------------------- ORDER / MATCH LOGIC ----------------------
def place_random_order():
    """Place a random buy or sell order (aggregated per price)."""
    side = random.choice(["BUY", "SELL"])
    # generate around the current mid with some spread
    mid = compute_mid_price() or INITIAL_MID
    raw = mid + random.uniform(-MAX_PRICE_MOVE, MAX_PRICE_MOVE)
    price = snap_price(raw)
    qty = random.randint(1, 20)

    if side == "BUY":
        buy_book[price] += qty
    else:
        sell_book[price] += qty

    # try some immediate matching attempts after inserting (makes activity)
    for _ in range(random.randint(*TICKS_PER_EVENT)):
        match_once()

def match_once():
    """Attempt to match top-of-book until no match at top level (one loop)."""
    bids = sorted_buy_levels()
    asks = sorted_sell_levels()
    if not bids or not asks:
        return

    best_bid_price, best_bid_qty = bids[0]
    best_ask_price, best_ask_qty = asks[0]

    if best_bid_price >= best_ask_price:
        # execute at ask price
        trade_price = best_ask_price
        trade_qty = min(best_bid_qty, best_ask_qty)
        ts = now_sim_datetime()
        trade_ticks.append((ts, trade_price, trade_qty))

        # reduce aggregated quantities
        buy_book[best_bid_price] -= trade_qty
        sell_book[best_ask_price] -= trade_qty
        if buy_book[best_bid_price] <= 0:
            del buy_book[best_bid_price]
        if sell_book[best_ask_price] <= 0:
            del sell_book[best_ask_price]

        # update candle with this tick
        add_tick_to_candle(ts, trade_price, trade_qty)

def compute_mid_price():
    """Return mid price if both sides exist, else None."""
    b, a = top_of_book()
    if b is None or a is None:
        return None
    return (b + a) / 2.0

# ---------------------- CANDLE MANAGEMENT ----------------------
def start_new_candle(start_ts):
    """Initialize a new candle dict with given start_ts (datetime)."""
    return {
        "start": start_ts,
        "open": None,
        "high": -math.inf,
        "low": math.inf,
        "close": None,
        "volume": 0
    }

def add_tick_to_candle(ts, price, qty):
    """Add a trade tick into the current candle, and close/push candle on interval."""
    global current_candle
    if current_candle is None:
        # start a candle aligned to the ts
        current_candle = start_new_candle(ts)

    # If candle interval expired, close it and start a new one.
    elapsed = (ts - current_candle["start"]).total_seconds()
    if elapsed >= CANDLE_INTERVAL_SECONDS:
        # push existing candle only if it has data (open is not None)
        if current_candle["open"] is not None:
            push_candle(current_candle)
        # start new candle with ts as start
        current_candle = start_new_candle(ts)

    # Write the tick
    if current_candle["open"] is None:
        current_candle["open"] = price
    current_candle["high"] = max(current_candle["high"], price)
    current_candle["low"] = min(current_candle["low"], price)
    current_candle["close"] = price
    current_candle["volume"] += qty

def push_candle(c):
    """Convert candle dict into a tuple and append to deque."""
    ts = c["start"]
    # If candle had no trades, do nothing
    if c["open"] is None:
        return
    candles.append((
        mdates.date2num(ts),  # x as float date for plotting
        c["open"],
        c["high"],
        c["low"],
        c["close"],
        c["volume"]
    ))
    # keep only last N candles to avoid slowdowns
    while len(candles) > 200:
        candles.popleft()

# ---------------------- VISUALIZATION ----------------------
def prepare_orderbook_bars(n_levels=NUM_PRICE_LEVELS):
    """Return x (qty), y (price labels), colors for stacked horizontal bar chart."""
    bids = sorted_buy_levels()[:n_levels]
    asks = sorted_sell_levels()[:n_levels]

    # For visual symmetry ensure same number of levels
    max_levels = max(len(bids), len(asks))
    bids = bids + [(None, 0)] * (max_levels - len(bids))
    asks = asks + [(None, 0)] * (max_levels - len(asks))

    # y positions (descending prices)
    y_labels = []
    bid_qtys = []
    ask_qtys = []
    for (bprice, bqty), (aprice, aqty) in zip(bids, asks[::-1]):
        # using reversed asks so orders have roughly aligned mid on chart
        y_labels.append((bprice if bprice is not None else aprice))
        bid_qtys.append(bqty if bprice is not None else 0)
        ask_qtys.append(aqty if aprice is not None else 0)

    return bid_qtys, ask_qtys, y_labels

def draw(ax_left, ax_right):
    """Draw orderbook depth on left axis and candles on right axis."""
    ax_left.clear()
    ax_right.clear()

    # ------- Order book depth (left) -------
    bids = sorted_buy_levels()[:NUM_PRICE_LEVELS]
    asks = sorted_sell_levels()[:NUM_PRICE_LEVELS]

    # Prepare aggregated lists for plotting
    # For bids: descending price order; for asks: ascending
    bid_prices = [p for p, q in bids]
    bid_qtys = [q for p, q in bids]
    ask_prices = [p for p, q in asks]
    ask_qtys = [q for p, q in asks]

    # plot bids and asks as horizontal bars around center
    # choose a central vertical axis for price; bars extend left (bids) and right (asks)
    # We'll plot price on y-axis, qty on x-axis
    if bid_prices:
        ax_left.barh([str(p) for p in bid_prices], bid_qtys, align='center', label='bids')
    if ask_prices:
        ax_left.barh([str(p) for p in ask_prices], [-q for q in ask_qtys], align='center', label='asks')

    ax_left.set_title("Order Book (top levels)")
    ax_left.set_xlabel("Quantity (bids -> positive, asks -> negative)")
    ax_left.legend(loc='upper right')

    # ------- Candles (right) -------
    ax_right.set_title("Time-based Candles (simulated minutes)")
    ax_right.set_xlabel("Time")
    ax_right.set_ylabel("Price")

    # draw candlesticks using candlestick_ohlc which expects list of [x, o, h, l, c]
    ohlc = []
    for (x, o, h, l, c, v) in candles:
        ohlc.append([x, o, h, l, c])

    if ohlc:
        # autoscale view to the last X candles
        visible = ohlc[-120:]
        candlestick_ohlc(ax_right, visible, width=0.0008 * CANDLE_INTERVAL_SECONDS, colorup='g', colordown='r')

        # format x-axis as dates
        ax_right.xaxis_date()
        ax_right.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax_right.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # display a small text summary
    mid = compute_mid_price()
    best_bid, best_ask = top_of_book()
    summary = f"Sim time: {now_sim_datetime().strftime('%H:%M:%S')}  |  Mid: {mid:.2f}  |  BestBid: {best_bid}  BestAsk: {best_ask}  Trades: {len(trade_ticks)}"
    ax_right.text(0.01, 0.98, summary, transform=ax_right.transAxes, fontsize=9, va='top')

    plt.tight_layout()

# ---------------------- MAIN SIMULATION LOOP ----------------------
def run_simulation():
    """Run simulation while updating plot periodically."""
    fig, (ax_left, ax_right) = plt.subplots(ncols=2, figsize=(12, 6))
    plt.ion()
    plt.show(block=False)

    last_plot_time = 0.0
    start_real = time.time()

    # initialize candle start to current sim time
    global current_candle
    current_candle = start_new_candle(now_sim_datetime())

    try:
        while True:
            # generate a burst of events proportional to refresh and EVENTS_PER_REAL_SECOND
            burst_count = max(1, int(EVENTS_PER_REAL_SECOND * PLOT_REFRESH_INTERVAL))
            for _ in range(burst_count):
                place_random_order()

            # occasionally try to close candle based on sim time (in case no trades happen)
            sim_now = now_sim_datetime()
            if (sim_now - current_candle["start"]).total_seconds() >= CANDLE_INTERVAL_SECONDS:
                # push existing candle (if had trades) and start new one
                if current_candle["open"] is not None:
                    push_candle(current_candle)
                current_candle = start_new_candle(sim_now)

            # update plot if enough real time passed
            if time.time() - last_plot_time >= PLOT_REFRESH_INTERVAL:
                draw(ax_left, ax_right)
                plt.pause(0.001)  # allow GUI event loop to update
                last_plot_time = time.time()

            # stop after duration if specified
            if SIM_DURATION_REAL_SECONDS is not None and (time.time() - start_real) >= SIM_DURATION_REAL_SECONDS:
                break

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # finalize last candle
        if current_candle and current_candle["open"] is not None:
            push_candle(current_candle)
        plt.ioff()
        draw(ax_left, ax_right)
        plt.show(block=True)
        print("Simulation finished. Total trades:", len(trade_ticks))


if __name__ == "__main__":
    run_simulation()
