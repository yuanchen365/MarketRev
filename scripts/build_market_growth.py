
import pandas as pd
import numpy as np

def aggregate_market_series(pd_module, df, base_date_str="2019-12-01"):
    """
    Aggregates company-level revenue data into market-wide metrics.
    
    Args:
        pd_module: The pandas module (dependency injection for testing/flexibility).
        df: DataFrame containing at least 'date' and 'revenue' columns.
            'revenue' should be numeric.
        base_date_str: String representing the base date for index calculation (YYYY-MM-DD).

    Returns:
        tuple: (market_df, actual_base_date)
            - market_df: DataFrame with 'date' as index and calculated metrics columns.
            - actual_base_date: Timestamp used as the denominator for the index.
    """
    # Ensure date is datetime
    df = df.copy()
    if not pd_module.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd_module.to_datetime(df['date'])

    # 1. Aggregate Total Revenue by Date
    market = df.groupby('date')['revenue'].sum().sort_index().to_frame(name='total_revenue')

    # 2. YoY Growth
    market['yoy_pct'] = market['total_revenue'].pct_change(12) * 100

    # 3. 3M Smooth YoY
    # "3M 平滑：yoy_pct_smooth_3m = rolling_mean(yoy_pct, 3)"
    market['yoy_pct_smooth_3m'] = market['yoy_pct'].rolling(window=3, min_periods=1).mean()

    # 4. Market Index (Base 2019-12 = 100)
    base_date = pd_module.Timestamp(base_date_str)
    if base_date in market.index:
        base_revenue = market.loc[base_date, 'total_revenue']
        actual_base_date = base_date
    else:
        # Fallback to the first available date if base_date not found
        if not market.empty:
            actual_base_date = market.index[0]
            base_revenue = market.iloc[0]['total_revenue']
        else:
            actual_base_date = None
            base_revenue = np.nan

    if base_revenue and base_revenue != 0:
        market['index_2019_12_100'] = (market['total_revenue'] / base_revenue) * 100
    else:
        market['index_2019_12_100'] = np.nan

    # 5. Moving Averages
    market['avg_revenue_3m'] = market['total_revenue'].rolling(window=3).mean()
    market['avg_revenue_6m'] = market['total_revenue'].rolling(window=6).mean()

    # 6. YTD Total
    # cumsum_by_calendar_year
    market['year'] = market.index.year
    market['ytd_total'] = market.groupby('year')['total_revenue'].cumsum()
    market.drop(columns=['year'], inplace=True)

    # 7. YTD YoY
    market['ytd_yoy_pct'] = market['ytd_total'].pct_change(12) * 100

    # 8. YTD YoY Smooth (3M, 6M)
    market['ytd_yoy_pct_avg_3m'] = market['ytd_yoy_pct'].rolling(window=3, min_periods=1).mean()
    market['ytd_yoy_pct_avg_6m'] = market['ytd_yoy_pct'].rolling(window=6, min_periods=1).mean()

    return market, actual_base_date

def _filter_out_finance(df, segment_df=None):
    """
    Filters out finance and insurance companies.
    
    Args:
        df: Input DataFrame with 'industry' or 'code' column.
        segment_df: (Optional) DataFrame mapping codes to segments if 'industry' not in df.
    
    Returns:
        DataFrame: Filtered DataFrame.
    """
    # Logic: Filter rows where 'industry' contains "金融" or "保險"
    # Assuming the input df has an 'industry' column as per the plan/SSOT
    if 'industry' not in df.columns:
         # If industry is missing, we might need to merge with segment_df first
         # For this implementation, we assume it's pre-merged or provided.
         return df 
    
    mask = df['industry'].astype(str).str.contains('金融|保險', na=False)
    return df[~mask].copy()
