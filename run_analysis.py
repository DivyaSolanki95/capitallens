"""
CapitalLens - Step 2: market-risk analysis of a multi-asset FX and energy trading book.

1. Daily hypothetical P&L for the current book on real market data (2007-2026)
2. 99% 1-day VaR with four models + 97.5% Expected Shortfall
3. Backtesting: exceptions, Kupiec, Christoffersen, Basel traffic light
4. Stressed VaR (worst 250-day window in history) and Basel 2.5 capital
5. Component VaR by position, historical and hypothetical stress tests
6. Capital impact of three proposed new trades
Outputs go to output/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import risk_engine as re_

OUT = Path("output"); OUT.mkdir(exist_ok=True)
prices = pd.read_csv("data/prices.csv", parse_dates=["Date"], index_col="Date")
rets = prices.pct_change().dropna()

# Current trading book: USD exposure to each risk factor (long = gains if the factor rises vs USD)
BOOK = pd.Series({"AUD": 60e6, "NZD": 15e6, "EUR": -40e6, "GBP": 25e6, "JPY": 30e6,
                  "INR": -20e6, "BRENT": 8e6, "NATGAS": -1.5e6}, name="position_usd")
WINDOW = 250
BT_START = "2009-07-01"  # first date every model (incl. 500-day windows) has full history: same test period for all


def pnl_of(book):
    return (rets[book.index] * book).sum(axis=1).rename("pnl")


CHAMPION = "Historical simulation"            # current production model (250-day window)
CHALLENGER = "Filtered HS, 500-day (challenger)"  # proposed model


def models(pnl):
    return pd.DataFrame({
        "Historical simulation": re_.hs_var(pnl, WINDOW),
        "Historical simulation, 500-day": re_.hs_var(pnl, 500),
        CHALLENGER: re_.filtered_hs_var(pnl, 500, 0.94),
        "Parametric normal": re_.normal_var(pnl, WINDOW),
        "EWMA (RiskMetrics)": re_.ewma_var(pnl),
        "Filtered HS, 250-day": re_.filtered_hs_var(pnl, WINDOW),
    })


def stressed_window(pnl):
    """The 250-day window with the largest historical-simulation VaR for this book."""
    roll = -pnl.rolling(WINDOW).quantile(0.01)
    end = roll.idxmax()
    start = pnl.index[pnl.index.get_loc(end) - WINDOW + 1]
    return start, end


def svar_on(pnl, start, end):
    w = pnl.loc[start:end]
    q = w.quantile(0.01)
    q25 = w.quantile(0.025)
    return -q, -w[w <= q25].mean()


def capital_for(book, s_start, s_end, model="champion"):
    pnl = pnl_of(book)
    var_hs = (re_.hs_var(pnl, WINDOW) if model == "champion" else re_.filtered_hs_var(pnl, 500, 0.94)).loc[BT_START:]
    exc = re_.exceptions(pnl.loc[BT_START:], var_hs)
    svar, ses = svar_on(pnl, s_start, s_end)
    cap = re_.basel_capital(var_hs.dropna(), svar, int(exc.iloc[-WINDOW:].sum()))
    es_today = re_.hs_es(pnl.iloc[-(WINDOW + 1):], WINDOW).iloc[-1]
    return pnl, var_hs, exc, svar, ses, cap, es_today


# ================= base book =================
pnl = pnl_of(BOOK)
var = models(pnl).loc[BT_START:]
actual = pnl.loc[BT_START:]

# ---- backtesting ----
rows, exc_frames = [], {}
for m in var:
    e = re_.exceptions(actual, var[m])
    exc_frames[m] = e
    n, x = len(e), int(e.sum())
    lr_k, p_k = re_.kupiec_pof(x, n)
    lr_c, p_c = re_.christoffersen_ind(e)
    roll = e.rolling(WINDOW).sum().dropna()
    last = int(e.iloc[-WINDOW:].sum())
    rows.append({"model": m, "days_tested": n, "exceptions": x, "expected_at_99%": round(0.01 * n, 1),
                 "exception_rate": round(x / n, 4), "kupiec_p_value": round(p_k, 4),
                 "kupiec_result": "Pass" if p_k > 0.05 else "Reject",
                 "christoffersen_p_value": round(p_c, 4),
                 "clustering_result": "Pass" if p_c > 0.05 else "Clustered",
                 "exceptions_last_250d": last, "current_zone": re_.traffic_light(last)[0],
                 "pct_days_green": round((roll <= 4).mean(), 3),
                 "pct_days_yellow": round(((roll >= 5) & (roll <= 9)).mean(), 3),
                 "pct_days_red": round((roll >= 10).mean(), 3)})
backtest = pd.DataFrame(rows)

by_year = pd.DataFrame({m: e.groupby(e.index.year).sum() for m, e in exc_frames.items()})
by_year.index.name = "year"

# ---- stressed VaR, capital, ES ----
s_start, s_end = stressed_window(pnl)
pnl, var_hs, exc_hs, svar, ses, cap, es_today = capital_for(BOOK, s_start, s_end)
cap_tbl = pd.DataFrame({"item": list(cap.keys()), "value": list(cap.values())})
_, var_ch, exc_ch, _, _, cap_ch, _ = capital_for(BOOK, s_start, s_end, model="challenger")
cap_compare = pd.DataFrame({"Champion: HS 250-day": cap, "Challenger: filtered HS 500-day": cap_ch})
extra = pd.DataFrame({"item": ["stress_window_start", "stress_window_end", "VaR_1d_99%_today", "ES_1d_97.5%_today",
                               "sVaR_1d_99%", "stressed_ES_1d_97.5%"],
                      "value": [s_start.date(), s_end.date(), var_hs.iloc[-1], es_today, svar, ses]})
cap_tbl = pd.concat([extra, cap_tbl], ignore_index=True)

# ---- component VaR ----
comp, total_param = re_.component_var(rets[BOOK.index].iloc[-WINDOW:], BOOK)
standalone = pd.Series({k: -(rets[k].iloc[-WINDOW:] * v).quantile(0.01) for k, v in BOOK.items()})
comp_tbl = pd.DataFrame({"position_usd": BOOK, "standalone_HS_VaR": standalone, "component_VaR": comp,
                         "share_of_VaR": comp / comp.sum()}).sort_values("component_VaR", ascending=False)

# ---- stress tests ----
hist = {"2008 financial crisis": ("2008-09-01", "2009-03-31"), "Brexit vote 2016": ("2016-06-20", "2016-07-15"),
        "COVID crash 2020": ("2020-02-20", "2020-04-30"), "Texas freeze gas spike 2021": ("2021-02-01", "2021-03-15"),
        "Energy shock 2022": ("2022-02-20", "2022-06-30")}
st = []
for name, (a, b) in hist.items():
    w = pnl.loc[a:b]
    ten = w.rolling(10).sum()
    st.append({"scenario": name, "type": "Historical", "worst_1d_pnl": w.min(), "worst_10d_pnl": ten.min(),
               "worst_10d_end": ten.idxmin().date()})
hypo = {"AUD and NZD fall 10%": {"AUD": -0.10, "NZD": -0.10}, "Brent falls 30%": {"BRENT": -0.30},
        "Natural gas doubles": {"NATGAS": 1.00}, "USD rallies 5% vs all currencies": {c: -0.05 for c in ["AUD", "NZD", "EUR", "GBP", "JPY", "INR"]}}
for name, shock in hypo.items():
    st.append({"scenario": name, "type": "Hypothetical", "worst_1d_pnl": sum(BOOK[k] * v for k, v in shock.items()),
               "worst_10d_pnl": np.nan, "worst_10d_end": None})
stress = pd.DataFrame(st)

# ---- new trade proposals: capital impact ----
TRADES = {"Energy spread: long $5m Brent / short $5m gas": {"BRENT": 5e6, "NATGAS": -5e6},
          "AUD/JPY carry: long $25m AUD / short $25m JPY": {"AUD": 25e6, "JPY": -25e6},
          "Hedge: sell $30m AUD": {"AUD": -30e6}}
nt = []
for name, t in TRADES.items():
    nb = BOOK.add(pd.Series(t), fill_value=0)
    _, v2, e2, sv2, ses2, c2, es2 = capital_for(nb, s_start, s_end)
    trade_alone = (rets[list(t)] * pd.Series(t)).sum(axis=1).iloc[-WINDOW:]
    nt.append({"trade": name, "standalone_VaR": -trade_alone.quantile(0.01),
               "change_in_VaR": v2.iloc[-1] - var_hs.iloc[-1], "change_in_ES": es2 - es_today,
               "change_in_sVaR": sv2 - svar, "new_zone": c2["zone"],
               "change_in_capital": c2["total_capital"] - cap["total_capital"],
               "capital_change_pct": (c2["total_capital"] - cap["total_capital"]) / cap["total_capital"]})
newtrades = pd.DataFrame(nt)

# ---- summary ----
hs_row = backtest.set_index("model").loc["Historical simulation"]
fhs_row = backtest.set_index("model").loc[CHALLENGER]
summary = pd.DataFrame({"metric": [
    "As-of date", "Risk factors", "Gross book size (USD)", "Days of market data", "Backtest days (out of sample)",
    "1-day 99% VaR, historical simulation (USD)", "1-day 97.5% Expected Shortfall (USD)",
    "Stressed VaR 1-day 99% (USD)", "Stress window", "Exceptions in last 250 days (HS)", "Basel traffic-light zone",
    "Basel 2.5 market-risk capital (USD)", "HS model: total exceptions vs expected",
    "Challenger model: total exceptions vs expected", "Capital under challenger model (USD)"],
    "value": [pnl.index[-1].date(), len(BOOK), BOOK.abs().sum(), len(prices), int(hs_row.days_tested),
              round(var_hs.iloc[-1]), round(es_today), round(svar), f"{s_start.date()} to {s_end.date()}",
              int(hs_row.exceptions_last_250d), cap["zone"], round(cap["total_capital"]),
              f"{int(hs_row.exceptions)} vs {hs_row['expected_at_99%']}",
              f"{int(fhs_row.exceptions)} vs {fhs_row['expected_at_99%']}", round(cap_ch["total_capital"])]})

dq = pd.read_csv("data/data_quality.csv")
big = rets.stack().rename("return").reset_index()
big.columns = ["date", "risk_factor", "return"]
big = big[big["return"].abs() > 0.25].sort_values("return", key=abs, ascending=False)
big["note"] = "Checked: genuine market event (spot price spike/crash), kept in data"

with pd.ExcelWriter(OUT / "CapitalLens_Market_Risk_Report.xlsx") as xw:
    summary.to_excel(xw, sheet_name="Summary", index=False)
    BOOK.to_frame().to_excel(xw, sheet_name="Positions")
    backtest.to_excel(xw, sheet_name="Backtest_Models", index=False)
    by_year.to_excel(xw, sheet_name="Exceptions_by_Year")
    cap_tbl.to_excel(xw, sheet_name="Capital", index=False)
    cap_compare.to_excel(xw, sheet_name="Capital_Champion_vs_Challenger")
    comp_tbl.to_excel(xw, sheet_name="Component_VaR")
    newtrades.to_excel(xw, sheet_name="New_Trade_Impact", index=False)
    stress.to_excel(xw, sheet_name="Stress_Tests", index=False)
    dq.to_excel(xw, sheet_name="Data_Quality", index=False)
    big.to_excel(xw, sheet_name="Large_Moves_Checked", index=False)

# ================= charts =================
M = 1e6
fig, ax = plt.subplots(figsize=(11, 4.5))
recent = actual.loc["2024-01-01":] / M
ax.bar(recent.index, recent.values, width=1.5, color="#9db4cc", label="Daily P&L")
ax.plot(-var.loc["2024-01-01":, "Historical simulation"] / M, color="#c0392b", lw=1.4, label="-VaR 99%, historical simulation")
ax.plot(-var.loc["2024-01-01":, CHALLENGER] / M, color="#1f4e79", lw=1.4, label="-VaR 99%, filtered HS 500-day (challenger)")
e = exc_frames["Historical simulation"].loc["2024-01-01":]
ed = e[e == 1].index
ax.scatter(ed, recent.loc[ed], color="#c0392b", zorder=5, s=22, label="Exception (loss > HS VaR)")
ax.set_ylabel("USD million"); ax.set_title("VaR backtest: daily P&L vs 99% VaR (2024 to Nov 2026)")
ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "var_backtest.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(comp_tbl.index[::-1], comp_tbl.component_VaR[::-1] / M, color=["#c0392b" if v > 0 else "#2e8b57" for v in comp_tbl.component_VaR[::-1]])
ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("Contribution to 99% VaR (USD million)")
ax.set_title("Where the risk comes from (component VaR)"); fig.tight_layout(); fig.savefig(OUT / "component_var.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(newtrades.trade[::-1].str.replace("$", r"\$", regex=False), newtrades.change_in_capital[::-1] / M,
        color=["#c0392b" if v > 0 else "#2e8b57" for v in newtrades.change_in_capital[::-1]])
ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("Change in Basel 2.5 market-risk capital (USD million)")
ax.set_title("Capital impact of proposed trades"); fig.tight_layout(); fig.savefig(OUT / "new_trade_capital.png", dpi=150)

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(summary.to_string(index=False)); print()
print(backtest[["model", "exceptions", "expected_at_99%", "kupiec_p_value", "kupiec_result", "christoffersen_p_value",
                "clustering_result", "exceptions_last_250d", "current_zone", "pct_days_red"]].to_string(index=False)); print()
print(cap_tbl.to_string(index=False)); print(); print(cap_compare.to_string()); print()
print(comp_tbl.round(0).to_string()); print()
print(newtrades.round(3).to_string(index=False)); print()
print(stress.round(0).to_string(index=False))

# ================= plain-English risk memo =================
def m(x):
    return f"USD {x / 1e6:,.1f}m"

top = comp_tbl.component_VaR.sort_values(ascending=False)
hedge = newtrades.iloc[2]; carry = newtrades.iloc[1]; spread = newtrades.iloc[0]
ch = backtest.set_index("model")
lines = [
    f"# CapitalLens market-risk summary, as of {pnl.index[-1].date()}", "",
    "## The short version", "",
    f"- On a bad day that comes about once every 100 trading days, the book could lose about **{m(var_hs.iloc[-1])}** (99% 1-day VaR). "
    f"When it does go past that, the average loss is **{m(es_today)}** (97.5% Expected Shortfall).",
    f"- Under the Basel 2.5 internal-models rules, the book needs **{m(cap['total_capital'])}** of market-risk capital. "
    f"**{cap['sVaR_charge'] / cap['total_capital']:.0%}** of it comes from stressed VaR, i.e. how the book would have done in the "
    f"{s_start.date()} to {s_end.date()} window (the 2008 financial crisis).",
    f"- The model is in the Basel **{cap['zone']}** zone ({int(hs_row.exceptions_last_250d)} exceptions in the last 250 days), so no capital add-on applies.", "",
    "## Is the model working?", "",
    f"- Over {int(hs_row.days_tested):,} days the current model (historical simulation, 250-day window) had "
    f"**{int(hs_row.exceptions)} losses bigger than VaR versus about {hs_row['expected_at_99%']:.0f} expected**. "
    f"The Kupiec test rejects it (p = {hs_row.kupiec_p_value:.3f}): it reacts too slowly when markets turn volatile.",
    f"- A challenger model, filtered historical simulation over 500 days, had **{int(fhs_row.exceptions)} exceptions** and passes both "
    f"the Kupiec test (p = {fhs_row.kupiec_p_value:.2f}) and the Christoffersen clustering test (p = {fhs_row.christoffersen_p_value:.2f}).",
    f"- Switching would change capital from {m(cap['total_capital'])} to {m(cap_ch['total_capital'])} "
    f"({(cap_ch['total_capital'] / cap['total_capital'] - 1):+.0%}). **Recommendation:** run the challenger in parallel for one quarter, then propose it to model validation.", "",
    "## Where the risk comes from", "",
    f"- The three biggest contributors to VaR are {top.index[0]}, {top.index[1]} and {top.index[2]}. "
    f"Together they make up {top.iloc[:3].sum() / top.sum():.0%} of the total.",
    f"- The short EUR position reduces risk (component VaR {m(comp_tbl.loc['EUR', 'component_VaR'])}) because it offsets the long AUD, GBP and NZD positions.", "",
    "## Proposed trades: capital impact", "",
    f"- **{hedge.trade}:** capital {m(hedge.change_in_capital)} ({hedge.capital_change_pct:+.0%}). It mostly lowers stressed VaR, since AUD fell sharply in 2008.",
    f"- **{carry.trade}:** today's VaR rises only {m(carry.change_in_VaR)}, but capital rises {m(carry.change_in_capital)} "
    f"({carry.capital_change_pct:+.0%}). Carry trades collapsed in 2008, which pushes up stressed VaR, and the book would move to the {carry.new_zone} zone.",
    f"- **{spread.trade}:** capital {m(spread.change_in_capital)} ({spread.capital_change_pct:+.0%}), driven by weather-related spikes in spot gas prices.", "",
    "## Stress tests", "",
] + [(f"- {r.scenario}: worst 1-day loss {m(-r.worst_1d_pnl)}, worst 10-day loss {m(-r.worst_10d_pnl)}"
       if r.type == "Historical" else f"- {r.scenario} (instant shock): loss {m(-r.worst_1d_pnl)}")
     for r in stress.itertuples()] + ["",
    "_Method notes: hypothetical P&L of today's positions on real daily market data; 10-day VaR = 1-day VaR x sqrt(10); "
    "stressed window = the 250-day window with the highest VaR for this book. Simplified Basel 2.5 IMA, for learning purposes._"]
(OUT / "CapitalLens_Risk_Memo.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:12]))
