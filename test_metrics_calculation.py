from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 將專案根目錄加入 sys.path，以便 import 專案模組
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_market_growth import aggregate_market_series


@pytest.fixture(scope="module")
def sample_dataframe() -> pd.DataFrame:
    """
    建立一個符合 SSOT_REVENUE.md 中範例驗證區塊的合成資料集。
    這個 fixture 可在模組內的所有測試中重複使用。
    """
    totals = [
        833.3333, 917.4312, 950.0, 960.0, 980.0, 990.0,
        1_005.0, 1_010.0, 1_015.0, 1_020.0, 1_030.0, 1_040.0,
        900.0, 1_000.0, 1_050.0,
    ]
    dates = pd.date_range("2023-01-01", periods=len(totals), freq="MS")

    rows = []
    for date, total in zip(dates, totals, strict=True):
        rows.append({"date": date, "code": "A001", "industry": "科技", "revenue": round(total * 0.6, 4)})
        rows.append({"date": date, "code": "B002", "industry": "零售", "revenue": round(total * 0.4, 4)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def aggregated_market(sample_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    執行核心的市場彙總函式，並回傳結果供後續驗證。
    """
    market, base_date = aggregate_market_series(pd, sample_dataframe)
    return market, base_date


def test_base_date_fallback(aggregated_market: tuple[pd.DataFrame, pd.Timestamp]):
    """
    驗證當 2019-12 不存在時，基期是否正確回退到資料的第一個月份。
    """
    _, base_date = aggregated_market
    assert base_date == pd.Timestamp("2023-01-01")


def test_market_metrics_2024_q1(aggregated_market: tuple[pd.DataFrame, pd.Timestamp]):
    """
    驗證 2024 年 Q1 的各項市場指標是否與 SSOT 文件中的範例一致。
    """
    market, _ = aggregated_market
    focus_data = market.loc["2024-01-01":"2024-03-01"].copy()

    # 預期結果，來自 SSOT_REVENUE.md
    expected_values = {
        "2024-01-01": {"total_revenue": 900.00, "yoy_pct": 8.00, "yoy_pct_smooth_3m": 8.00, "avg_revenue_3m": 990.00, "ytd_total": 900.00, "ytd_yoy_pct": 8.00, "index_2019_12_100": 108.00},
        "2024-02-01": {"total_revenue": 1000.00, "yoy_pct": 9.00, "yoy_pct_smooth_3m": 8.50, "avg_revenue_3m": 980.00, "ytd_total": 1900.00, "ytd_yoy_pct": 8.52, "index_2019_12_100": 120.00},
        "2024-03-01": {"total_revenue": 1050.00, "yoy_pct": 10.53, "yoy_pct_smooth_3m": 9.18, "avg_revenue_3m": 983.33, "ytd_total": 2950.00, "ytd_yoy_pct": 9.23, "index_2019_12_100": 126.00},
    }

    for date_str, metrics in expected_values.items():
        date = pd.Timestamp(date_str)
        for metric, expected in metrics.items():
            actual = focus_data.loc[date, metric]
            # 使用 np.isclose 處理浮點數比較
            assert np.isclose(actual, expected, atol=0.01), \
                f"Mismatch at {date.date()} for metric '{metric}': Expected {expected}, got {actual:.2f}"


def test_industry_yoy(sample_dataframe: pd.DataFrame):
    """驗證分產業的 YoY 計算是否正確。"""
    grouped = sample_dataframe.groupby(["industry", "date"])["revenue"].sum().unstack(level=0)
    yoy = grouped.pct_change(12) * 100.0
    
    assert np.isclose(yoy.loc["2024-01-01", "科技"], 8.00, atol=0.01)
    assert np.isclose(yoy.loc["2024-02-01", "零售"], 9.00, atol=0.01)
    assert np.isclose(yoy.loc["2024-03-01", "科技"], 10.53, atol=0.01)