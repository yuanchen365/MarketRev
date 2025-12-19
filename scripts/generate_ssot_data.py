
import pandas as pd
import finlab
from finlab import data
import sys
from pathlib import Path

# Add root to path to import scripts
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_market_growth import aggregate_market_series, _filter_out_finance

def main():
    print("Starting SSOT Data Generation...")

    # 1. Login to FinLab
    try:
        import streamlit as st
        # Try to get API key from Streamlit secrets (for Cloud)
        # Note: st.secrets works even if not running via `streamlit run` IF .streamlit/secrets.toml exists,
        # but locally we rely on the file.
        try:
            api_key = st.secrets["FINLAB_API_KEY"]
            print("Using API Key from Streamlit Secrets.")
        except (FileNotFoundError, KeyError, AttributeError):
            # Fallback to local file
            with open('finlabapi.txt', 'r') as f:
                api_key = f.read().strip()
            print("Using API Key from finlabapi.txt")
            
        finlab.login(api_key)
        print("FinLab Login Successful.")
    except Exception as e:
        print(f"Error reading API key or logging in: {e}")
        return

    # 2. Fetch Data
    print("Fetching monthly revenue data...")
    # 'monthly_revenue:當月營收' returns a wide dataframe: index=date, columns=stock_code
    rev_wide = data.get('monthly_revenue:當月營收')
    
    # 3. Process Data (Wide to Long)
    print("Processing revenue data...")
    rev_long = rev_wide.stack().reset_index()
    rev_long.columns = ['date', 'code', 'revenue']
    
    # Filter out 0 or NaN revenue
    rev_long = rev_long[rev_long['revenue'] > 0]

    # 4. Load Segment Data
    print("Loading segment classification...")
    try:
        segment_df = pd.read_csv('segment.csv', encoding='utf-8') # Encoding might be big5 or utf-8-sig
        # segment.csv structure based on previous `view_file`:
        # "代號","名稱"...,"產業別"...
        # The '代號' column seems to be "0050" or ="0050". Need to clean.
    except Exception:
        # Retry with different encoding if utf-8 fails
        segment_df = pd.read_csv('segment.csv', encoding='cp950')
    
    # Clean '代號' column: remove '="' and '"'
    if '代號' in segment_df.columns:
        segment_df['code'] = segment_df['代號'].astype(str).str.replace('=', '').str.replace('"', '').str.strip()
    
    # Create Code -> Industry map
    if '產業別' in segment_df.columns:
        code_industry_map = segment_df.set_index('code')['產業別'].to_dict()
    else:
        print("Error: '產業別' column not found in segment.csv")
        return

    # Map Industry to Revenue Data
    rev_long['industry'] = rev_long['code'].map(code_industry_map)
    
    # Fill missing industry with 'Other' or similar if needed, or drop?
    # SSOT doesn't specify, but for market total it shouldn't matter as long as we have revenue.
    # For Industry output, we need it.
    
    # 5. Aggregate Market Series (All)
    print("Aggregating Market Series (All)...")
    market_df, _ = aggregate_market_series(pd, rev_long)
    market_df.to_csv('out/market_ssot.csv')
    print("Saved out/market_ssot.csv")

    # 6. Aggregate Market Series (Ex-Finance)
    print("Aggregating Market Series (Ex-Finance)...")
    rev_ex_fin = _filter_out_finance(rev_long)
    market_ex_fin_df, _ = aggregate_market_series(pd, rev_ex_fin)
    market_ex_fin_df.to_csv('out/market_ssot_ex_fin.csv')
    print("Saved out/market_ssot_ex_fin.csv")

    # 7. Industry Level Data (All Industries)
    print("Calculating Industry Data...")
    # Group by Industry and Date
    ind_rev = rev_long.groupby(['industry', 'date'])['revenue'].sum().reset_index()
    
    # Calculate Pivot Table for Revenue (All Industries)
    ind_pivot = ind_rev.pivot(index='date', columns='industry', values='revenue')
    
    # Save Raw Revenue Pivot for All Industries (for dynamic calculation in UI)
    ind_pivot.to_csv('out/industry_revenue.csv')
    print("Saved out/industry_revenue.csv (All Industries)")

    # Legacy: Top 12 YoY (Optional, kept for reference or other tools)
    ind_yoy = ind_pivot.pct_change(12) * 100
    
    # Identify Top 12 Industries by latest total revenue (or average?)
    # "前 12 大產業年增率圖" -> usually by size
    latest_date = ind_rev['date'].max()
    # Use average of last 3 months to be more stable
    last_3m = ind_pivot.iloc[-3:].mean().sort_values(ascending=False)
    top_12_industries = last_3m.head(12).index.tolist()
    
    ind_yoy_top12 = ind_yoy[top_12_industries]
    ind_yoy_top12.to_csv('out/industry_yoy.csv')
    print("Saved out/industry_yoy.csv")

    # 8. Stock Level Details (Price & Metrics)
    print("Calculating Stock Level Details...")
    
    # Fetch Price Data
    print("Fetching price data...")
    price = data.get('price:收盤價')
    # Get latest price for each stock
    latest_price = price.iloc[-1].to_frame(name='close_price')
    latest_price.index.name = 'code'
    
    # Calculate Stock Metrics (YTD YoY, 3M, 6M)
    # We need to iterate through each stock in rev_wide (columns are codes)
    stock_details_list = []
    
    # Use code_industry_map to filter or label
    # rev_wide index is date, columns are codes
    
    for code in rev_wide.columns:
        if code not in code_industry_map:
            continue
            
        industry = code_industry_map[code]
        series = rev_wide[code].dropna()
        
        if series.empty:
            continue
            
        # Create temp df for calculation
        temp = series.to_frame(name='revenue')
        temp['year'] = temp.index.year
        
        # Calculate YTD
        temp['ytd_total'] = temp.groupby('year')['revenue'].cumsum()
        
        # Calculate YTD YoY
        temp['ytd_yoy'] = temp['ytd_total'].pct_change(12) * 100
        
        # Calculate Smooth
        temp['3m_smooth'] = temp['ytd_yoy'].rolling(window=3, min_periods=1).mean()
        temp['6m_smooth'] = temp['ytd_yoy'].rolling(window=6, min_periods=1).mean()
        
        # Get latest available data
        last_valid_idx = temp['revenue'].last_valid_index()
        if last_valid_idx is None:
            continue
            
        latest_metrics = temp.loc[last_valid_idx]
        
        # Get latest price (might be different date, usually T)
        p = latest_price.loc[code, 'close_price'] if code in latest_price.index else 0
        
        stock_details_list.append({
            'code': code,
            'industry': industry,
            'close_price': p,
            'ytd_yoy': latest_metrics['ytd_yoy'],
            '3m_smooth': latest_metrics['3m_smooth'],
            '6m_smooth': latest_metrics['6m_smooth'],
            'revenue_date': last_valid_idx
        })
        
    stock_details_df = pd.DataFrame(stock_details_list)
    
    # Add Name if available in segment_df
    if '名稱' in segment_df.columns:
        code_name_map = segment_df.set_index('code')['名稱'].to_dict()
        stock_details_df['name'] = stock_details_df['code'].map(code_name_map)
    else:
        stock_details_df['name'] = stock_details_df['code']
        
    # Reorder columns
    cols = ['code', 'name', 'industry', 'close_price', 'ytd_yoy', '3m_smooth', '6m_smooth', 'revenue_date']
    stock_details_df = stock_details_df[[c for c in cols if c in stock_details_df.columns]]
    
    stock_details_df.to_csv('out/stock_details.csv', index=False)
    print("Saved out/stock_details.csv")

    print("Data Generation Complete.")

if __name__ == "__main__":
    main()
