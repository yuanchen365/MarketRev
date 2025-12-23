import streamlit as st
import pandas as pd
import finlab
from finlab import data
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def login_finlab():
    # if finlab.is_logged_in():
    #     return True
    
    try:
        # Try to get API key from Streamlit secrets
        try:
            api_key = st.secrets["FINLAB_API_KEY"]
        except (FileNotFoundError, KeyError, AttributeError):
            # Fallback to local file
            if os.path.exists('finlabapi.txt'):
                with open('finlabapi.txt', 'r') as f:
                    api_key = f.read().strip()
            else:
                st.error("找不到 FinLab API Key。請設定 secrets 或建立 finlabapi.txt。")
                return False
        
        finlab.login(api_key)
        return True
    except Exception as e:
        st.error(f"FinLab 登入失敗: {e}")
        return False

@st.cache_data(ttl=3600*24) # Cache for 24 hours
def get_breadth_data():
    if not login_finlab():
        return None

    with st.spinner("正在下載與計算資料..."):
        # 1. Finlab Data
        # Revenue
        rev = data.get('monthly_revenue:當月營收')
        
        # Security Categories
        cats = data.get('security_categories')
        
        # Benchmark (Taiex)
        try:
            taiex = data.get('benchmark_return:發行量加權股價指數')
        except Exception:
            st.warning("無法從 Finlab 取得加權指數，改用 Yahoo Finance (^TWII)")
            taiex = yf.download("^TWII", start="2010-01-01", progress=False)['Close']
            if isinstance(taiex, pd.DataFrame):
                taiex = taiex.iloc[:, 0]
        
        # 2. Yahoo Finance Data (ACWI) - Removed
        # acwi = yf.download("ACWI", start="2010-01-01", progress=False)['Close']
        # if isinstance(acwi, pd.DataFrame):
        #      acwi = acwi.iloc[:, 0] # Handle multi-index if necessary
        
        # 3. Process Revenue & Categories
        # cats structure: usually stock_id as index or column. 
        # Let's inspect cats. If it's standard finlab, it might be a df with columns.
        # Assuming 'category' column exists. If not, we might need to adjust.
        # Standard finlab 'security_categories' usually has 'category' column.
        
        # Reset index if stock_id is index
        if 'stock_id' not in cats.columns and cats.index.name == 'stock_id':
            cats = cats.reset_index()
        elif 'stock_id' not in cats.columns:
             # Try to find the stock id column
             pass

        # Create a map: stock_id -> category
        # We need to ensure we have stock_id and category
        # If cats is simple, it might just be the map.
        # Let's try to be safe.
        if 'category' in cats.columns:
             # Assuming index is stock_id or there is a stock_id column
             if 'stock_id' in cats.columns:
                 cat_map = cats.set_index('stock_id')['category'].to_dict()
             else:
                 cat_map = cats['category'].to_dict()
        else:
            # Fallback or error
            st.error("無法解析產業分類資料 (security_categories)")
            return None

        # Stack revenue to long format
        rev_long = rev.stack().reset_index()
        rev_long.columns = ['date', 'stock_id', 'revenue']
        
        # Map category
        rev_long['category'] = rev_long['stock_id'].map(cat_map)
        
        # Drop stocks without category or revenue
        rev_long = rev_long.dropna(subset=['category', 'revenue'])
        
        # 4. Calculate Industry Revenue
        # Group by category and date
        ind_rev = rev_long.groupby(['category', 'date'])['revenue'].sum().reset_index()
        
        # Pivot to wide format for YoY calculation: index=date, columns=category
        ind_wide = ind_rev.pivot(index='date', columns='category', values='revenue')
        
        # Calculate YoY
        ind_yoy = ind_wide.pct_change(12)
        
        # 5. Construct Indicators
        # Count industries with YoY > 0
        pos_counts = (ind_yoy > 0).sum(axis=1)
        
        # Calculate MA
        ma3 = pos_counts.rolling(window=3).mean()
        ma6 = pos_counts.rolling(window=6).mean()
        
        # Combine into a DataFrame
        df_breadth = pd.DataFrame({
            'Positive_Count': pos_counts,
            'MA3': ma3,
            'MA6': ma6
        })
        
        # 6. Align Data (Taiex & ACWI)
        # Taiex and ACWI are daily. Revenue is monthly (usually 10th of next month, but indexed by month end or start).
        # Finlab monthly revenue index is usually the 10th of the next month (reporting date) or the 1st of the next month.
        # Actually Finlab 'monthly_revenue:當月營收' index is usually the date of the revenue month (e.g. 2023-01-01 for Jan revenue).
        # But we want to align with price.
        # Let's resample Price to Monthly (End of Month) to match the granularity, or keep daily and plot on same x-axis.
        # Plotly handles different x-axes well.
        
        # However, for "Breadth", it's monthly.
        # Let's keep Taiex and ACWI as daily series, Plotly can handle it.
        
        return df_breadth, taiex

def render_breadth_tab():
    st.header("台股產業營收成長廣度 vs 加權指數 (Taiwan Revenue Breadth)")
    
    st.markdown("""
    **指標解讀**:
    *   **營收成長廣度**: 統計每月營收年增率 (YoY) 為正的產業數量。
    *   **趨勢判斷**:
        *   當 **3MA (洋紅色)** 向上突破 **6MA (白色)**：顯示產業基本面轉佳，多頭訊號。
        *   當 **3MA (洋紅色)** 向下穿過 **6MA (白色)**：顯示產業基本面轉弱，警訊。
    """)
    
    data_tuple = get_breadth_data()
    
    if data_tuple is None:
        return

    df_breadth, taiex = data_tuple
    
    # Create Figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Right Axis: Breadth (Bars & MAs) - Moved to Right to allow Taiex on Left
    fig.add_trace(
        go.Bar(
            x=df_breadth.index, 
            y=df_breadth['Positive_Count'], 
            name='營收成長家數',
            marker_color='#00B4D8',
            opacity=0.6
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_breadth.index, 
            y=df_breadth['MA3'], 
            name='3MA (3個月均線)',
            line=dict(color='#D500F9', width=2)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_breadth.index, 
            y=df_breadth['MA6'], 
            name='6MA (6個月均線)',
            line=dict(color='#FFFFFF', width=2.5)
        ),
        secondary_y=True
    )
    
    # Left Axis: Taiex
    fig.add_trace(
        go.Scatter(
            x=taiex.index, 
            y=taiex['close'] if isinstance(taiex, pd.DataFrame) and 'close' in taiex.columns else taiex, # Handle series or df
            name='台灣加權指數',
            line=dict(color='red', width=1.5),
            opacity=0.8
        ),
        secondary_y=False
    )
    
    # Layout
    fig.update_layout(
        title='台股產業營收成長廣度 vs 加權指數',
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title="指數點位",
            side="left"
        ),
        yaxis2=dict(
            title="產業家數",
            side="right",
            overlaying="y",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
