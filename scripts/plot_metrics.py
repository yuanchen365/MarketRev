
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

# Set font for Traditional Chinese support if available, else standard
# This is environment specific. In Streamlit cloud or local windows, it varies.
# For Windows 'Microsoft JhengHei' is common.
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('out')

def load_data():
    market = pd.read_csv(OUTPUT_DIR / 'market_ssot.csv', index_col='date', parse_dates=True)
    market_ex = pd.read_csv(OUTPUT_DIR / 'market_ssot_ex_fin.csv', index_col='date', parse_dates=True)
    industry_yoy = pd.read_csv(OUTPUT_DIR / 'industry_yoy.csv', index_col='date', parse_dates=True)
    return market, market_ex, industry_yoy

def plot_market_yoy(df, title_suffix="", filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if columns exist (start of series might differ)
    if 'yoy_pct' in df.columns:
        ax.bar(df.index, df['yoy_pct'], label='YoY %', alpha=0.5, color='skyblue', width=20)
    
    if 'yoy_pct_smooth_3m' in df.columns:
        ax.plot(df.index, df['yoy_pct_smooth_3m'], label='3M Smooth YoY %', color='red', linewidth=2)
    
    ax.set_title(f'Market Revenue YoY {title_suffix}')
    ax.set_ylabel('YoY Growth (%)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()
    return fig

def plot_market_index(df, title_suffix="", filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'index_2019_12_100' in df.columns:
        ax.plot(df.index, df['index_2019_12_100'], label='Index (2019-12=100)', color='purple')
        
    ax.set_title(f'Market Revenue Index {title_suffix}')
    ax.set_ylabel('Index')
    ax.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()
    return fig

def plot_market_ma(df, title_suffix="", filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df['total_revenue'], label='Revenue', color='lightgray', alpha=0.5)
    ax.plot(df.index, df['avg_revenue_3m'], label='3M MA', color='blue')
    ax.plot(df.index, df['avg_revenue_6m'], label='6M MA', color='orange')
    
    ax.set_title(f'Market Revenue & Moving Averages {title_suffix}')
    ax.set_ylabel('Revenue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()
    return fig

def plot_ytd_yoy(df, title_suffix="", filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'ytd_yoy_pct' in df.columns:
        ax.bar(df.index, df['ytd_yoy_pct'], label='YTD YoY %', alpha=0.3, color='green', width=20)
    
    if 'ytd_yoy_pct_avg_3m' in df.columns:
        ax.plot(df.index, df['ytd_yoy_pct_avg_3m'], label='3M Smooth', color='darkgreen')
    
    if 'ytd_yoy_pct_avg_6m' in df.columns:
        ax.plot(df.index, df['ytd_yoy_pct_avg_6m'], label='6M Smooth', color='olive', linestyle='--')

    ax.set_title(f'YTD Market Revenue YoY {title_suffix}')
    ax.set_ylabel('Growth (%)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()
    return fig

def plot_industry_facets(df_yoy, top_n=12, title_suffix="", filename=None):
    # df_yoy is just the industry yoy columns
    industries = df_yoy.columns[:top_n]
    
    # Calculate grid size
    rows = (len(industries) + 3) // 4
    cols = min(len(industries), 4)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), sharex=True)
    axes = axes.flatten()
    
    for i, ind in enumerate(industries):
        ax = axes[i]
        series = df_yoy[ind]
        ax.bar(series.index, series, color='teal', alpha=0.6)
        ax.set_title(ind)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    if filename:
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()
    return fig

def generate_all_plots():
    market, market_ex, industry_yoy = load_data()
    
    # Market All
    plot_market_yoy(market, filename='market_yoy.png')
    plot_market_index(market, filename='market_index.png')
    plot_market_ma(market, filename='market_revenue_ma.png')
    plot_ytd_yoy(market, filename='market_ytd_yoy.png')
    
    # Market Ex-Finance
    plot_market_yoy(market_ex, title_suffix="(Ex-Finance)", filename='market_yoy_ex_fin.png')
    plot_market_index(market_ex, title_suffix="(Ex-Finance)", filename='market_index_ex_fin.png')
    plot_market_ma(market_ex, title_suffix="(Ex-Finance)", filename='market_revenue_ma_ex_fin.png')
    plot_ytd_yoy(market_ex, title_suffix="(Ex-Finance)", filename='market_ytd_yoy_ex_fin.png')
    
    # Industry
    plot_industry_facets(industry_yoy, filename='industry_yoy_top12.png')
    
    # Ex-Finance Industry? The industry_yoy.csv is generally derived from All currently, 
    # unless we want to filter columns. The requirement asked for:
    # "out/industry_yoy_top12_ex_fin.png：排除金融保險後的前 12 大產業年增率圖"
    # To do this, we need to filter `industry_yoy` columns to exclude finance related names.
    # Simple keyword filter:
    cols_ex_fin = [c for c in industry_yoy.columns if "金融" not in c and "保險" not in c]
    plot_industry_facets(industry_yoy[cols_ex_fin], filename='industry_yoy_top12_ex_fin.png')
    
    print("All plots generated in out/")

if __name__ == "__main__":
    generate_all_plots()
