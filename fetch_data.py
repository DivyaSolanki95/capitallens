"""
CapitalLens - Step 1: download real daily market data and build a clean price panel.

Sources (open data on GitHub, mirrored from the US Federal Reserve FRED and EIA):
  - FX rates:   datasets/exchange-rates   (quoted as foreign currency per 1 USD)
  - Brent crude: datasets/oil-prices      (USD per barrel)
  - Natural gas: datasets/natural-gas     (Henry Hub, USD per MMBtu)

Output: data/prices.csv (USD value of each risk factor), data/data_quality.csv
"""
from pathlib import Path
import pandas as pd

CUTOFF = "2026-11-13"   # freeze the as-of date so results are reproducible
START = "2007-07-02"  # natural gas is reliably daily from mid-2007; keeps the 2008 crisis in scope
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/datasets"
SOURCES = {
    "fx.csv": f"{BASE}/exchange-rates/main/data/daily.csv",
    "brent.csv": f"{BASE}/oil-prices/main/data/brent-daily.csv",
    "natgas.csv": f"{BASE}/natural-gas/main/data/daily.csv",
}
FX = {"Australia": "AUD", "Euro": "EUR", "Japan": "JPY", "United Kingdom": "GBP",
      "India": "INR", "New Zealand": "NZD"}


def load(name):
    """Use the saved snapshot if present, otherwise download it."""
    path = RAW / name
    if not path.exists():
        print(f"downloading {name} ...")
        pd.read_csv(SOURCES[name]).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["Date"])


fx = load("fx.csv")
fx = fx[fx.Country.isin(FX)].pivot_table(index="Date", columns="Country", values="Exchange rate")
fx = fx.rename(columns=FX)
usd_value = 1 / fx  # USD value of one unit of each foreign currency (long AUD = long this series)

brent = load("brent.csv").set_index("Date").Price.rename("BRENT")
gas = load("natgas.csv").set_index("Date").Price.rename("NATGAS")

raw = pd.concat([usd_value, brent, gas], axis=1, sort=True).sort_index().loc[START:CUTOFF]

# ---- data quality: business days, gaps, bad values ----
bdays = pd.bdate_range(raw.index.min(), raw.index.max())
panel = raw.reindex(bdays)
dq = pd.DataFrame({
    "first_date": raw.apply(lambda s: s.first_valid_index().date()),
    "missing_business_days": panel.isna().sum(),
    "non_positive_values": (raw <= 0).sum(),
})
# holidays differ across markets: carry the last price forward at most 3 days, then drop leftover gaps
panel = panel.ffill(limit=3)
dq["still_missing_after_ffill"] = panel.isna().sum()
panel = panel.dropna()
dq["rows_used"] = len(panel)

panel.index.name = "Date"
panel.to_csv("data/prices.csv")
dq.to_csv("data/data_quality.csv")
print(f"prices: {panel.shape[0]} days x {panel.shape[1]} risk factors, {panel.index.min().date()} to {panel.index.max().date()}")
print(dq)
