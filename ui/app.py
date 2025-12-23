import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the new tab function
try:
    from ui.revenue_breadth import render_breadth_tab
except ImportError:
    # Fallback if file not found during dev
    def render_breadth_tab():
        st.error("Revenue Breadth module not found.")

OUTPUT_DIR = ROOT / 'out'

st.set_page_config(layout="wide", page_title="市場營收 SSOT 儀表板")

st.title("市場營收 SSOT 儀表板")

@st.cache_data
def get_data():
    try:
        market = pd.read_csv(OUTPUT_DIR / 'market_ssot.csv', index_col='date', parse_dates=True)
        market_ex = pd.read_csv(OUTPUT_DIR / 'market_ssot_ex_fin.csv', index_col='date', parse_dates=True)
        # Load raw industry revenue for dynamic calculation
        industry_rev = pd.read_csv(OUTPUT_DIR / 'industry_revenue.csv', index_col='date', parse_dates=True)
        # Load stock details
        try:
            stock_details = pd.read_csv(OUTPUT_DIR / 'stock_details.csv')
        except FileNotFoundError:
            stock_details = None
            
        return market, market_ex, industry_rev, stock_details
    except FileNotFoundError:
        return None, None, None, None

market, market_ex, industry_rev, stock_details = get_data()

# Filter data to start from 2020 for display
START_DATE = '2020-01-01'
if market is not None:
    market = market[market.index >= START_DATE]
if market_ex is not None:
    market_ex = market_ex[market_ex.index >= START_DATE]
if industry_rev is not None:
    industry_rev = industry_rev[industry_rev.index >= START_DATE]

def calculate_industry_metrics(df_rev):
    """
    Calculate YTD YoY, 3M Smooth, and 6M Smooth for each industry.
    Returns a dictionary of DataFrames, keyed by industry name.
    """
    metrics = {}
    
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_rev.index):
        df_rev.index = pd.to_datetime(df_rev.index)
        
    df_rev = df_rev.sort_index()
    
    for col in df_rev.columns:
        # Create a temp df for this industry
        temp = df_rev[[col]].rename(columns={col: 'revenue'})
        
        # 1. YTD Revenue
        temp['year'] = temp.index.year
        temp['ytd_total'] = temp.groupby('year')['revenue'].cumsum()
        
        # 2. YTD YoY %
        temp['ytd_yoy_pct'] = temp['ytd_total'].pct_change(12) * 100
        
        # 3. Smooth (3M, 6M) of YTD YoY
        temp['3m_smooth'] = temp['ytd_yoy_pct'].rolling(window=3, min_periods=1).mean()
        temp['6m_smooth'] = temp['ytd_yoy_pct'].rolling(window=6, min_periods=1).mean()
        
        metrics[col] = temp
        
    return metrics

def plot_market_yoy_plotly(df, title_suffix=""):
    fig = go.Figure()
    if 'yoy_pct' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['yoy_pct'], name='年增率 (YoY %)', marker_color='skyblue', opacity=0.6))
    if 'yoy_pct_smooth_3m' in df.columns:
        # Changed color to Crimson for contrast
        fig.add_trace(go.Scatter(x=df.index, y=df['yoy_pct_smooth_3m'], name='3個月平滑年增率', line=dict(color='crimson', width=2)))
    
    fig.update_layout(title=f'市場營收年增率 {title_suffix}', yaxis_title='年增率 (%)', xaxis_title='日期', hovermode="x unified", dragmode="pan")
    return fig

def plot_market_index_plotly(df, title_suffix=""):
    fig = go.Figure()
    if 'index_2019_12_100' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['index_2019_12_100'], name='指數 (2019-12=100)', line=dict(color='purple')))
    
    fig.add_hline(y=100, line_dash="dash", line_color="gray")
    fig.update_layout(title=f'市場營收指數 {title_suffix}', yaxis_title='指數', xaxis_title='日期', hovermode="x unified", dragmode="pan")
    return fig

def plot_market_ma_plotly(df, title_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['total_revenue'], name='營收', line=dict(color='lightgray'), opacity=0.5))
    fig.add_trace(go.Scatter(x=df.index, y=df['avg_revenue_3m'], name='3個月均線', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['avg_revenue_6m'], name='6個月均線', line=dict(color='orange')))
    
    fig.update_layout(title=f'市場營收與移動平均 {title_suffix}', yaxis_title='營收', xaxis_title='日期', hovermode="x unified", dragmode="pan")
    return fig

def plot_ytd_yoy_plotly(df, title_suffix=""):
    fig = go.Figure()
    if 'ytd_yoy_pct' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['ytd_yoy_pct'], name='YTD 年增率 %', marker_color='lightgreen', opacity=0.4))
    if 'ytd_yoy_pct_avg_3m' in df.columns:
        # Changed color to RoyalBlue for contrast
        fig.add_trace(go.Scatter(x=df.index, y=df['ytd_yoy_pct_avg_3m'], name='3個月平滑', line=dict(color='royalblue', width=2)))
    if 'ytd_yoy_pct_avg_6m' in df.columns:
        # Changed to Crimson for contrast (used to be olive)
        fig.add_trace(go.Scatter(x=df.index, y=df['ytd_yoy_pct_avg_6m'], name='6個月平滑', line=dict(color='crimson', width=2, dash='dash')))

    fig.update_layout(title=f'年初至今 (YTD) 市場營收年增率 {title_suffix}', yaxis_title='成長率 (%)', xaxis_title='日期', hovermode="x unified", dragmode="pan")
    return fig

def plot_industry_chart(industry_name, df_metrics):
    fig = go.Figure()
    
    # YTD YoY Bar
    fig.add_trace(go.Bar(
        x=df_metrics.index, 
        y=df_metrics['ytd_yoy_pct'], 
        name='YTD YoY %', 
        marker_color='lightgreen', 
        opacity=0.4
    ))
    
    # 3M Smooth Line - RoyalBlue for high contrast
    fig.add_trace(go.Scatter(
        x=df_metrics.index, 
        y=df_metrics['3m_smooth'], 
        name='3M Smooth', 
        line=dict(color='royalblue', width=2)
    ))
    
    # 6M Smooth Line - Crimson for high contrast
    fig.add_trace(go.Scatter(
        x=df_metrics.index, 
        y=df_metrics['6m_smooth'], 
        name='6M Smooth', 
        line=dict(color='crimson', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'{industry_name}',
        yaxis_title='成長率 (%)', 
        xaxis_title='日期', 
        hovermode="x unified",
        dragmode="pan",  # Enable panning/zooming interaction base
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    return fig


# TABS
tab1, tab2 = st.tabs(["市場總覽", "營收廣度"])

with tab1:
    if market is None:
        st.error("找不到資料檔案。請先生成資料。")
        if st.button("生成資料"):
            st.info("正在執行生成腳本...")
            from scripts import generate_ssot_data
            generate_ssot_data.main()
            st.success("資料已生成！請重新整理頁面。")
            st.experimental_rerun()
    else:
        # Sidebar
        st.sidebar.header("設定")
        exclude_finance = st.sidebar.checkbox("排除金融保險業", value=True)
        
        st.sidebar.markdown("### 資料更新管理")
        admin_password = st.sidebar.text_input("輸入管理員密碼解鎖更新功能", type="password")
        
        # Get password from secrets or default to 'admin' for local testing if not set
        # In Cloud, user MUST set ADMIN_PASSWORD in secrets
        try:
            correct_password = st.secrets["ADMIN_PASSWORD"]
        except (FileNotFoundError, KeyError, AttributeError):
            correct_password = "admin" # Fallback for local dev
            
        if admin_password == correct_password:
            if st.sidebar.button("更新資料"):
                with st.spinner("正在更新資料，請稍候..."):
                    from scripts import generate_ssot_data
                    generate_ssot_data.main()
                    st.cache_data.clear()
                    st.success("資料已更新！")
                    # Compatible rerun
                    if hasattr(st, 'rerun'):
                        st.rerun()
                    else:
                        st.experimental_rerun()
        elif admin_password:
            st.sidebar.error("密碼錯誤")
        else:
            st.sidebar.info("請輸入密碼以啟用更新按鈕")
        
        if exclude_finance:
            df_market = market_ex
            title_suffix = "(排除金融)"
        else:
            df_market = market
            title_suffix = "(全市場)"

        # Initialize session state for selected industry
        if 'selected_industry' not in st.session_state:
            st.session_state.selected_industry = None

        def set_industry(ind):
            st.session_state.selected_industry = ind

        # Sidebar Industry Details
        st.sidebar.divider()
        st.sidebar.subheader("產業詳情")
        
        selected_ind = st.session_state.selected_industry
        
        if selected_ind:
            st.sidebar.markdown(f"### {selected_ind}")
            st.sidebar.markdown("*(已選取)*")
            if stock_details is not None:
                 # Filter details
                ind_stocks = stock_details[stock_details['industry'] == selected_ind].copy()
                # Sort by YTD YoY desc
                ind_stocks = ind_stocks.sort_values('ytd_yoy', ascending=False)
                
                for idx, row in ind_stocks.iterrows():
                    code = row['code']
                    name = row['name']
                    price = row['close_price']
                    ytd = row['ytd_yoy']
                    s3 = row['3m_smooth']
                    s6 = row['6m_smooth']
                    
                    with st.sidebar.expander(f"{code} {name} (${price:,.1f})"):
                        st.write(f"**YTD YoY %**: {ytd:+.2f}%")
                        st.write(f"**3M Smooth**: {s3:+.2f}%")
                        st.write(f"**6M Smooth**: {s6:+.2f}%")
                        st.markdown("---")
        else:
            st.sidebar.info("請在右側圖表上方點選產業名稱以查看成分股詳情。")

        # Main Dashboard Metrics
        latest_date = df_market.index[-1]
        last_yoy = df_market.loc[latest_date, 'yoy_pct']
        last_revenue = df_market.loc[latest_date, 'total_revenue']
        
        st.header(f"市場總覽 {title_suffix}")
        st.write(f"資料截止日期: {latest_date.strftime('%Y-%m-%d')}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("當月總營收", f"{last_revenue:,.0f}")
        m2.metric("當月年增率 (YoY)", f"{last_yoy:.2f}%", delta_color="normal")
        
        st.divider()

        # Config for plotly chart to enable scroll zoom
        plotly_config = {'scrollZoom': True}

        # Market Charts
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_market_yoy_plotly(df_market, title_suffix), use_container_width=True, config=plotly_config)
        with c2:
            st.plotly_chart(plot_market_index_plotly(df_market, title_suffix), use_container_width=True, config=plotly_config)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_market_ma_plotly(df_market, title_suffix), use_container_width=True, config=plotly_config)
        with c4:
            st.plotly_chart(plot_ytd_yoy_plotly(df_market, title_suffix), use_container_width=True, config=plotly_config)
            
        st.divider()
        
        st.header("產業趨勢細項 (YTD YoY)")
        st.caption("💡 點選下方產業名稱按鈕，即可在左側選單查看該產業成分股詳情")
        
        # Calculate metrics for all industries
        ind_metrics_map = calculate_industry_metrics(industry_rev)
        
        # Identify all industries
        all_industries = []
        if industry_rev is not None:
            all_industries = list(industry_rev.columns)
            if exclude_finance:
                 all_industries = [ind for ind in all_industries if "金融" not in ind and "保險" not in ind]
        
        if industry_rev is not None:
             # Sort by latest revenue size
            latest_rev_row = industry_rev.iloc[-1]
            sorted_industries = sorted(all_industries, key=lambda x: latest_rev_row.get(x, 0), reverse=True)
            
            cols = st.columns(2)
            for i, ind in enumerate(sorted_industries):
                with cols[i % 2]:
                    # Use button as header/selector
                    # Use a unique key for each button
                    st.button(f"📊 {ind}", key=f"btn_{ind}", on_click=set_industry, args=(ind,), use_container_width=True)
                    
                    # Plot chart without title (since button is the title)
                    fig = plot_industry_chart(ind, ind_metrics_map[ind])
                    fig.update_layout(title_text="", margin=dict(t=10)) # Remove title, reduce top margin
                    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

        with st.expander("檢視原始資料"):
            st.subheader("市場資料")
            st.dataframe(df_market.sort_index(ascending=False))
            st.subheader("產業營收資料")
            st.dataframe(industry_rev.sort_index(ascending=False))

with tab2:
    render_breadth_tab()
