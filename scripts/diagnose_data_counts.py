
import pandas as pd
import finlab
from finlab import data
import sys
from pathlib import Path

# Add root to path to import scripts
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    print("Starting Data Counts Diagnosis...")

    # 1. Login to FinLab
    try:
        import streamlit as st
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
        segment_df = pd.read_csv('segment.csv', encoding='utf-8')
    except Exception:
        segment_df = pd.read_csv('segment.csv', encoding='cp950')
    
    # Clean '代號' column
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
    
    # 5. Count Companies per Industry per Date
    print("Calculating Company Counts per Industry...")
    # Group by Industry and Date, then count unique codes
    ind_counts = rev_long.groupby(['industry', 'date'])['code'].count().reset_index()
    
    # Pivot
    ind_counts_pivot = ind_counts.pivot(index='date', columns='industry', values='code')
    
    # Save
    import os
    os.makedirs('out', exist_ok=True)
    ind_counts_pivot.to_csv('out/industry_counts.csv')
    print("Saved out/industry_counts.csv")

    print("Diagnosis Generation Complete.")

if __name__ == "__main__":
    main()
