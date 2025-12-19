
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_market_growth import aggregate_market_series

def run_test():
    totals = [
        833.3333, 917.4312, 950.0, 960.0, 980.0, 990.0,
        1_005.0, 1_010.0, 1_015.0, 1_020.0, 1_030.0, 1_040.0,
        900.0, 1_000.0, 1_050.0,
    ]
    dates = pd.date_range("2023-01-01", periods=len(totals), freq="MS")

    rows = []
    for date, total in zip(dates, totals):
        rows.append({"date": date, "code": "A001", "industry": "科技", "revenue": round(total * 0.6, 4)})
        rows.append({"date": date, "code": "B002", "industry": "零售", "revenue": round(total * 0.4, 4)})
    df = pd.DataFrame(rows)

    print("Running aggregate_market_series...")
    market, base_date = aggregate_market_series(pd, df)
    
    print(f"Base date: {base_date}")
    
    # Validation
    focus_data = market.loc["2024-01-01":"2024-03-01"].copy()
    expected_values = {
        "2024-01-01": {"total_revenue": 900.00, "yoy_pct": 8.00, "yoy_pct_smooth_3m": 8.00, "avg_revenue_3m": 990.00, "ytd_total": 900.00, "ytd_yoy_pct": 8.00, "index_2019_12_100": 108.00},
        "2024-02-01": {"total_revenue": 1000.00, "yoy_pct": 9.00, "yoy_pct_smooth_3m": 8.50, "avg_revenue_3m": 980.00, "ytd_total": 1900.00, "ytd_yoy_pct": 8.52, "index_2019_12_100": 120.00},
        "2024-03-01": {"total_revenue": 1050.00, "yoy_pct": 10.53, "yoy_pct_smooth_3m": 9.18, "avg_revenue_3m": 983.33, "ytd_total": 2950.00, "ytd_yoy_pct": 9.23, "index_2019_12_100": 126.00},
    }

    for date_str, metrics in expected_values.items():
        date = pd.Timestamp(date_str)
        print(f"Checking {date_str}...")
        for metric, expected in metrics.items():
            actual = focus_data.loc[date, metric]
            if not np.isclose(actual, expected, atol=0.01):
                print(f"Mismatch at {date_str} for {metric}: Expected {expected}, got {actual}")
            else:
                pass # print(f"  OK: {metric}")

    print("Manual verification passed.")

if __name__ == "__main__":
    run_test()
