# CapitalLens market-risk summary, as of 2026-11-13

## The short version

- On a bad day that comes about once every 100 trading days, the book could lose about **USD 1.2m** (99% 1-day VaR). When it does go past that, the average loss is **USD 1.4m** (97.5% Expected Shortfall).
- Under the Basel 2.5 internal-models rules, the book needs **USD 48.7m** of market-risk capital. **79%** of it comes from stressed VaR, i.e. how the book would have done in the 2007-11-12 to 2008-10-24 window (the 2008 financial crisis).
- The model is in the Basel **Green** zone (3 exceptions in the last 250 days), so no capital add-on applies.

## Is the model working?

- Over 4,533 days the current model (historical simulation, 250-day window) had **63 losses bigger than VaR versus about 45 expected**. The Kupiec test rejects it (p = 0.013): it reacts too slowly when markets turn volatile.
- A challenger model, filtered historical simulation over 500 days, had **52 exceptions** and passes both the Kupiec test (p = 0.33) and the Christoffersen clustering test (p = 0.27).
- Switching would change capital from USD 48.7m to USD 51.1m (+5%). **Recommendation:** run the challenger in parallel for one quarter, then propose it to model validation.

## Where the risk comes from

- The three biggest contributors to VaR are AUD, BRENT and NATGAS. Together they make up 83% of the total.
- The short EUR position reduces risk (component VaR USD -0.1m) because it offsets the long AUD, GBP and NZD positions.

## Proposed trades: capital impact

- **Hedge: sell $30m AUD:** capital USD -22.8m (-47%). It mostly lowers stressed VaR, since AUD fell sharply in 2008.
- **AUD/JPY carry: long $25m AUD / short $25m JPY:** today's VaR rises only USD 0.1m, but capital rises USD 37.9m (+78%). Carry trades collapsed in 2008, which pushes up stressed VaR, and the book would move to the Yellow zone.
- **Energy spread: long $5m Brent / short $5m gas:** capital USD 19.8m (+41%), driven by weather-related spikes in spot gas prices.

## Stress tests

- 2008 financial crisis: worst 1-day loss USD 5.0m, worst 10-day loss USD 13.8m
- Brexit vote 2016: worst 1-day loss USD 1.6m, worst 10-day loss USD 2.0m
- COVID crash 2020: worst 1-day loss USD 5.5m, worst 10-day loss USD 16.0m
- Texas freeze gas spike 2021: worst 1-day loss USD 1.8m, worst 10-day loss USD 2.5m
- Energy shock 2022: worst 1-day loss USD 1.9m, worst 10-day loss USD 4.8m
- AUD and NZD fall 10% (instant shock): loss USD 7.5m
- Brent falls 30% (instant shock): loss USD 2.4m
- Natural gas doubles (instant shock): loss USD 1.5m
- USD rallies 5% vs all currencies (instant shock): loss USD 3.5m

_Method notes: hypothetical P&L of today's positions on real daily market data; 10-day VaR = 1-day VaR x sqrt(10); stressed window = the 250-day window with the highest VaR for this book. Simplified Basel 2.5 IMA, for learning purposes._