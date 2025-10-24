from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_market_growth import aggregate_market_series


def build_sample_dataframe() -> pd.DataFrame:
    """Create a small synthetic dataset that matches the narrative example."""
    # Monthly total revenue for 2023-01 through 2024-03.
    totals = [
        833.3333,
        917.4312,
        950.0,
        960.0,
        980.0,
        990.0,
        1_005.0,
        1_010.0,
        1_015.0,
        1_020.0,
        1_030.0,
        1_040.0,
        900.0,
        1_000.0,
        1_050.0,
    ]
    dates = pd.date_range("2023-01-01", periods=len(totals), freq="MS")

    rows = []
    for date, total in zip(dates, totals, strict=True):
        rows.append(
            {
                "date": date,
                "code": "A001",
                "industry": "科技",
                "revenue": round(total * 0.6, 4),
            }
        )
        rows.append(
            {
                "date": date,
                "code": "B002",
                "industry": "零售",
                "revenue": round(total * 0.4, 4),
            }
        )
    return pd.DataFrame(rows)


def summarize_market_metrics(market: pd.DataFrame) -> pd.DataFrame:
    """Return the key metrics for 2024-01 through 2024-03."""
    cols = [
        "total_revenue",
        "yoy_pct",
        "yoy_pct_smooth_3m",
        "avg_revenue_3m",
        "avg_revenue_6m",
        "ytd_total",
        "ytd_yoy_pct",
        "ytd_yoy_pct_avg_3m",
        "index_2019_12_100",
    ]
    return market.loc["2024-01-01":"2024-03-01", cols].round(4)


def compute_industry_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by industry and compute YoY for 2024-01~03."""
    grouped = (
        df.groupby(["industry", "date"], as_index=False)["revenue"]
        .sum()
        .sort_values(["industry", "date"])
    )
    grouped["yoy_pct"] = grouped.groupby("industry")["revenue"].pct_change(12) * 100.0
    focus = grouped[
        (grouped["date"] >= "2024-01-01") & (grouped["date"] <= "2024-03-01")
    ].copy()
    return focus.pivot(index="date", columns="industry", values="yoy_pct").round(4)


def main() -> None:
    df = build_sample_dataframe()
    market, base_date = aggregate_market_series(pd, df)
    print("=== Market metrics (2024-01 to 2024-03) ===")
    print(summarize_market_metrics(market).to_string())
    print()
    print(f"Base date used for index: {base_date.date()}")
    print()
    print("=== Industry YoY (2024-01 to 2024-03) ===")
    print(compute_industry_yoy(df).to_string())


if __name__ == "__main__":
    main()
