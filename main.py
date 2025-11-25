import random
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc


TICK_SIZE = 0.05              # price increment step
TICKS_PER_CANDLE = 40         # tick-based candle size
NUM_EVENTS = 3000             # number of random order events

buy_orders = []               # list of [price, qty]
sell_orders = []              # list of [price, qty]
trade_ticks = []              # captured trade prices


def random_price():
    """Generate a random tick-aligned price."""
    raw = random.uniform(95, 105)
    return round(raw / TICK_SIZE) * TICK_SIZE


def place_order():
    """Create a random buy or sell order and add it to the book."""
    side = random.choice(["BUY", "SELL"])
    price = random_price()
    qty = random.randint(1, 100)

    if side == "BUY":
        buy_orders.append([price, qty])
    else:
        sell_orders.append([price, qty])


def sort_books():
    """Sort BUY (high→low) and SELL (low→high)."""
    buy_orders.sort(key=lambda x: -x[0])
    sell_orders.sort(key=lambda x: x[0])


def match_orders():
    """Match trades when the best bid >= best ask. Each trade generates a tick."""
    while buy_orders and sell_orders and buy_orders[0][0] >= sell_orders[0][0]:
        trade_price = sell_orders[0][0]            # trade always at ask
        trade_qty = min(buy_orders[0][1], sell_orders[0][1])

        trade_ticks.append(trade_price)

        buy_orders[0][1] -= trade_qty
        sell_orders[0][1] -= trade_qty

        if buy_orders[0][1] == 0:
            buy_orders.pop(0)
        if sell_orders[0][1] == 0:
            sell_orders.pop(0)


for _ in range(NUM_EVENTS):
    place_order()
    sort_books()
    match_orders()



candles = []
for i in range(0, len(trade_ticks), TICKS_PER_CANDLE):
    chunk = trade_ticks[i:i + TICKS_PER_CANDLE]

    if len(chunk) < TICKS_PER_CANDLE:   # ignore incomplete candle
        break

    open_price = chunk[0]
    high_price = max(chunk)
    low_price = min(chunk)
    close_price = chunk[-1]

    candles.append([i, open_price, high_price, low_price, close_price])


fig, ax = plt.subplots()
ax.set_title("Tick-Based Candle Chart from Order Book Simulation")
ax.set_xlabel("Candle Index")
ax.set_ylabel("Price")

candlestick_ohlc(ax, candles, width=0.6)
plt.show()

