from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_market_growth import (  # noqa: E402
    aggregate_market_series,
    configure_cjk_font,
    load_and_prepare,
    plot_industry_facets,
    plot_market,
    plot_revenue_ma,
    plot_ytd_yoy,
)

st.set_page_config(page_title="上市櫃整體營收儀表板", layout="wide")

DEFAULT_FONT = "Microsoft JhengHei"
EX_FIN_SUFFIX = "_ex_fin"


def fmt_num(value) -> str:
    try:
        return f"{value:,.0f}"
    except Exception:
        return "-"


def fmt_pct(value) -> str:
    try:
        return f"{value:,.2f}%"
    except Exception:
        return "-"


def build_line_chart(
    df: pd.DataFrame,
    series: Sequence[Tuple[str, str]],
    title: str,
    y_title: str,
    hover_fmt: str,
    y_tickformat: Optional[str] = None,
    y_range: Optional[Tuple[float, float]] = None,
    showlegend: bool = True,
) -> go.Figure:
    fig = go.Figure()
    for label, column in series:
        if column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["月份"],
                y=df[column],
                mode="lines",
                name=label,
                hovertemplate=f"%{{x|%Y-%m}}<br>{label}: %{{y:{hover_fmt}}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        showlegend=showlegend,
        legend=dict(title="系列", itemclick="toggle", itemdoubleclick="toggleothers"),
        font=dict(family=DEFAULT_FONT, size=13),
        xaxis=dict(
            title="月份",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
        ),
        yaxis=dict(
            title=y_title,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
        ),
        margin=dict(l=60, r=25, t=60, b=60),
        template="plotly_white",
    )

    if y_tickformat is not None:
        fig.update_yaxes(tickformat=y_tickformat)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    return fig


def render_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def select_series(
    options: Sequence[Tuple[str, str, bool]],
    key_prefix: str,
) -> List[Tuple[str, str]]:
    if not options:
        return []
    cols = st.columns(min(len(options), 3))
    selected: List[Tuple[str, str]] = []
    for idx, (label, column, default) in enumerate(options):
        col = cols[idx % len(cols)]
        if col.checkbox(label, value=default, key=f"{key_prefix}_{column}"):
            selected.append((label, column))
    return selected


def split_ex_finance(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "industry" not in df.columns:
        return None
    industries = df["industry"].astype(str)
    mask_fin = industries.str.contains("金融", na=False)
    mask_fin = mask_fin | industries.str.contains("保險", na=False)
    if not mask_fin.any():
        return None
    filtered = df.loc[~mask_fin].copy()
    if filtered.empty:
        return None
    return filtered


st.title("上市櫃整體營收儀表板")
st.caption("來源：MOPS 月營收資訊；指標涵蓋 YoY、平滑 YoY、指數、YTD 與移動平均。")

with st.sidebar:
    st.header("設定")
    csv_path = st.text_input("CSV 路徑", value="mops_revenue_107_114.csv")
    out_dir = st.text_input("輸出資料夾", value="out")
    st.markdown("★ 字型設定 ★")
    st.caption("預設使用微軟正黑體，如需指定字型請提供檔案路徑。")
    font_file = st.text_input("字型檔路徑（可選）", value="")
    show_industry = st.checkbox("顯示產業圖（Top12）", value=False)
    also_export_png = st.checkbox("重新匯出圖檔到輸出資料夾", value=False)
    do_build = st.button("建置/更新圖表")

if do_build:
    font_file_opt = font_file.strip() or None
    configure_cjk_font(DEFAULT_FONT, font_file_opt)

    in_path = Path(csv_path)
    out_path = Path(out_dir)

    df_ex_fin: Optional[pd.DataFrame] = None
    market_ex_fin: Optional[pd.DataFrame] = None
    base_date_ex_fin = None

    with st.spinner("讀取並整理資料中…"):
        df = load_and_prepare(in_path)
        market, base_date = aggregate_market_series(pd, df)
        df_ex_fin = split_ex_finance(df)
        if df_ex_fin is not None:
            market_ex_fin, base_date_ex_fin = aggregate_market_series(pd, df_ex_fin)

    st.session_state["revenue_raw_df"] = df
    st.session_state["revenue_market_df"] = market
    st.session_state["revenue_base_date"] = base_date
    st.session_state["revenue_raw_df_ex_fin"] = df_ex_fin
    st.session_state["revenue_market_ex_fin_df"] = market_ex_fin
    st.session_state["revenue_base_date_ex_fin"] = base_date_ex_fin
    st.session_state["revenue_out_dir"] = out_path
    st.session_state["revenue_font_file"] = font_file_opt

    if also_export_png:
        with st.spinner("輸出圖檔中…"):
            plot_market(market, base_date, out_path)
            plot_ytd_yoy(market, out_path)
            plot_revenue_ma(market, out_path)
            if show_industry:
                plot_industry_facets(df, out_path)

            if market_ex_fin is not None and base_date_ex_fin is not None:
                plot_market(market_ex_fin, base_date_ex_fin, out_path, suffix=EX_FIN_SUFFIX)
                plot_ytd_yoy(market_ex_fin, out_path, suffix=EX_FIN_SUFFIX)
                plot_revenue_ma(market_ex_fin, out_path, suffix=EX_FIN_SUFFIX)
                if show_industry and df_ex_fin is not None and not df_ex_fin.empty:
                    plot_industry_facets(df_ex_fin, out_path, suffix=EX_FIN_SUFFIX)

stored_market = st.session_state.get("revenue_market_df")
stored_base_date = st.session_state.get("revenue_base_date")
stored_raw = st.session_state.get("revenue_raw_df")
stored_market_ex = st.session_state.get("revenue_market_ex_fin_df")
stored_base_date_ex = st.session_state.get("revenue_base_date_ex_fin")
stored_raw_ex = st.session_state.get("revenue_raw_df_ex_fin")
stored_font_file = st.session_state.get("revenue_font_file")

if stored_market is None or stored_base_date is None:
    st.info("請輸入檔案並點選「建置/更新圖表」。")
    st.stop()

configure_cjk_font(DEFAULT_FONT, stored_font_file)

dataset_options: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
    "整體市場": {
        "market": stored_market,
        "base": stored_base_date,
        "raw": stored_raw,
    }
}

if stored_market_ex is not None and stored_base_date_ex is not None and stored_raw_ex is not None:
    dataset_options["排除金融保險"] = {
        "market": stored_market_ex,
        "base": stored_base_date_ex,
        "raw": stored_raw_ex,
    }

selection = st.radio("資料範圍", list(dataset_options.keys()), horizontal=True)
current = dataset_options[selection]
market = current["market"]
base_date = current["base"]
df_current = current["raw"]

if market is None or base_date is None or df_current is None:
    st.error("資料載入失敗，請重新建置。")
    st.stop()

latest_period = market.index.max()
last_row = market.loc[latest_period]

st.subheader(f"最新月份：{latest_period.strftime('%Y-%m')}｜{selection}" if pd.notna(latest_period) else selection)
c1, c2, c3, c4 = st.columns(4)
c1.metric("年增率 (YoY)", fmt_pct(last_row.get("yoy_pct")))
c2.metric("年增率 3M 平滑", fmt_pct(last_row.get("yoy_pct_smooth_3m")))
c3.metric("3M 平均營收", fmt_num(last_row.get("avg_revenue_3m")))
c4.metric("6M 平均營收", fmt_num(last_row.get("avg_revenue_6m")))

c5, c6, c7, c8 = st.columns(4)
c5.metric("累計營收 (YTD)", fmt_num(last_row.get("ytd_total")))
c6.metric("YTD 年增率", fmt_pct(last_row.get("ytd_yoy_pct")))
c7.metric("YTD 年增率 3M 平滑", fmt_pct(last_row.get("ytd_yoy_pct_avg_3m")))
c8.metric("YTD 年增率 6M 平滑", fmt_pct(last_row.get("ytd_yoy_pct_avg_6m")))

st.metric(f"指數（基期 {base_date.strftime('%Y-%m')} = 100）", fmt_num(last_row.get("index_2019_12_100")))

st.divider()
st.subheader("互動圖表")
st.caption("勾選左側 Checkbox 控制線條顯示。滑鼠移動可查看數值。")

market_plot_df = market.reset_index().rename(columns={"date": "月份"})

with st.expander("總體年增率（含 3M 平滑）", expanded=True):
    yoy_options = [
        ("年增率 (YoY)", "yoy_pct", True),
        ("年增率 3M 平滑", "yoy_pct_smooth_3m", True),
    ]
    selected_series = select_series(yoy_options, f"series_yoy_{selection}")
    if not selected_series:
        st.info("請至少勾選一條線。")
    else:
        render_chart(
            build_line_chart(
                market_plot_df,
                selected_series,
                "年增率 (YoY)",
                "年增率 (%)",
                ".2f",
                y_tickformat=".1f",
            )
        )

with st.expander(f"營收指數（基期 {base_date.strftime('%Y-%m')} = 100）", expanded=True):
    idx_options = [("營收指數", "index_2019_12_100", True)]
    selected_series = select_series(idx_options, f"series_index_{selection}")
    if not selected_series:
        st.info("請至少勾選一條線。")
    else:
        render_chart(
            build_line_chart(
                market_plot_df,
                selected_series,
                "營收指數",
                "指數",
                ",.1f",
                y_tickformat=",.1f",
            )
        )

with st.expander("累計營收 (YTD) 年增率", expanded=True):
    ytd_options = [
        ("YTD 年增率", "ytd_yoy_pct", True),
        ("YTD 年增率 3M 平滑", "ytd_yoy_pct_avg_3m", True),
        ("YTD 年增率 6M 平滑", "ytd_yoy_pct_avg_6m", True),
    ]
    selected_series = select_series(ytd_options, f"series_ytd_{selection}")
    if not selected_series:
        st.info("請至少勾選一條線。")
    else:
        render_chart(
            build_line_chart(
                market_plot_df,
                selected_series,
                "累計營收 (YTD) 年增率",
                "年增率 (%)",
                ".2f",
                y_tickformat=".1f",
            )
        )

with st.expander("3/6 個月平均營收 vs 當月營收", expanded=False):
    ma_options = [
        ("3M 平均營收", "avg_revenue_3m", True),
        ("6M 平均營收", "avg_revenue_6m", True),
        ("當月營收", "total_revenue", False),
    ]
    selected_series = select_series(ma_options, f"series_ma_{selection}")
    if not selected_series:
        st.info("請至少勾選一條線。")
    else:
        render_chart(
            build_line_chart(
                market_plot_df,
                selected_series,
                "平滑營收趨勢",
                "金額",
                ",.0f",
            )
        )

with st.expander("產業年增率（自選）", expanded=False):
    if "industry" not in df_current.columns:
        st.info("來源資料缺少『產業別』欄位，無法繪製。")
    else:
        grouped = df_current.groupby(["industry", "date"], as_index=False)["revenue"].sum()
        total_by_ind = grouped.groupby("industry")["revenue"].sum().sort_values(ascending=False)
        top_n = int(st.number_input("Top 產業數（依累計營收排序）", min_value=3, max_value=30, value=12, step=1))
        industries = list(total_by_ind.head(top_n).index)
        if not industries:
            st.info("找不到符合條件的產業。")
        else:
            default_inds = industries[: min(6, len(industries))]
            selected_inds = st.multiselect("選擇要顯示的產業", options=industries, default=default_inds)
            if not selected_inds:
                st.info("請至少選擇一個產業。")
            else:
                sub = grouped[grouped["industry"].isin(selected_inds)].copy()
                sub.sort_values(["industry", "date"], inplace=True)
                sub["yoy_pct"] = sub.groupby("industry")["revenue"].pct_change(12) * 100.0
                sub = sub[pd.notna(sub["yoy_pct"])]
                if sub.empty:
                    st.info("選取產業缺少完整的 12 個月比較資料。")
                else:
                    chart_data = (
                        sub.rename(columns={"industry": "產業", "date": "月份", "yoy_pct": "年增率"})
                        .pivot(index="月份", columns="產業", values="年增率")
                        .reset_index()
                    )
                    chart_data["月份"] = pd.to_datetime(chart_data["月份"])
                    display_options = [
                        (ind, ind, True) for ind in selected_inds if ind in chart_data.columns
                    ]
                    selected_series = select_series(display_options, f"series_industry_{selection}")
                    if not selected_series:
                        st.info("請至少勾選一條線。")
                    else:
                        render_chart(
                            build_line_chart(
                                chart_data,
                                selected_series,
                                "產業年增率",
                                "年增率 (%)",
                                ".2f",
                                y_tickformat=".1f",
                            )
                        )

st.success("完成！如需匯出最新圖檔，可在左側勾選『重新匯出圖檔』後再建置一次。")
