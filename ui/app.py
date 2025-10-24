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

from scripts.build_macro_business_cycle import build_macro_business_cycle
from scripts.build_market_growth import (
    aggregate_market_series,
    configure_cjk_font,
    load_and_prepare,
    plot_industry_facets,
    plot_market,
    plot_revenue_ma,
    plot_ytd_yoy,
)

st.set_page_config(page_title="Revenue & Macro Dashboard", layout="wide")

DEFAULT_FONT = "Microsoft JhengHei"
EX_FIN_SUFFIX = "_ex_fin"
MACRO_DEFAULT_START = "2020-M1"
MACRO_DEFAULT_END = "2025-M8"


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
        legend=dict(title="Series", itemclick="toggle", itemdoubleclick="toggleothers"),
        font=dict(family=DEFAULT_FONT, size=13),
        xaxis=dict(
            title="\u6708\u4efd",
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
    mask_fin = industries.str.contains("\u91d1\u878d", na=False)
    mask_fin = mask_fin | industries.str.contains("\u4fdd\u96aa", na=False)
    if not mask_fin.any():
        return None
    filtered = df.loc[~mask_fin].copy()
    if filtered.empty:
        return None
    return filtered


st.title("\u4e0a\u5e02\u6ac3\u71df\u6536\u8207\u666f\u6c23\u6307\u6a19\u5100\u8868\u677f")
st.caption("Revenue SSOT + DGBAS business-cycle indicators with interactive controls.")

with st.sidebar:
    st.header("\u8a2d\u5b9a")
    csv_path = st.text_input("CSV \u8def\u5f91", value="mops_revenue_107_114.csv")
    out_dir = st.text_input("\u8f38\u51fa\u8cc7\u6599\u593e", value="out")
    st.markdown("---")
    st.markdown("\u5b57\u578b\u8a2d\u5b9a")
    font_file = st.text_input("\u5b57\u578b\u6a94\u8def\u5f91 (\u53ef\u9078)", value="")
    show_industry = st.checkbox("\u986f\u793a\u7522\u696d\u5716 (Top 12)", value=False)
    also_export_png = st.checkbox("\u91cd\u65b0\u532f\u51fa\u5716\u6a94", value=False)
    st.markdown("---")
    st.markdown("\u666f\u6c23\u6307\u6a19\u7bc4\u570d")
    macro_start = st.text_input("\u8d77\u59cb (YYYY-Mm)", value=MACRO_DEFAULT_START)
    macro_end = st.text_input("\u7d50\u675f (YYYY-Mm)", value=MACRO_DEFAULT_END)
    do_build = st.button("\u5efa\u7acb / \u66f4\u65b0 \u5716\u8868")

if do_build:
    font_file_opt = font_file.strip() or None
    configure_cjk_font(DEFAULT_FONT, font_file_opt)

    in_path = Path(csv_path)
    out_path = Path(out_dir)

    df_ex_fin: Optional[pd.DataFrame] = None
    market_ex_fin: Optional[pd.DataFrame] = None
    base_date_ex_fin = None

    with st.spinner("\u8b80\u53d6\u71df\u6536\u8cc7\u6599..."):
        revenue_df = load_and_prepare(in_path)
        revenue_market, revenue_base_date = aggregate_market_series(pd, revenue_df)
        df_ex_fin = split_ex_finance(revenue_df)
        if df_ex_fin is not None:
            market_ex_fin, base_date_ex_fin = aggregate_market_series(pd, df_ex_fin)

    st.session_state["revenue_raw_df"] = revenue_df
    st.session_state["revenue_market_df"] = revenue_market
    st.session_state["revenue_base_date"] = revenue_base_date
    st.session_state["revenue_raw_df_ex_fin"] = df_ex_fin
    st.session_state["revenue_market_ex_fin_df"] = market_ex_fin
    st.session_state["revenue_base_date_ex_fin"] = base_date_ex_fin
    st.session_state["revenue_out_dir"] = out_path
    st.session_state["revenue_font_file"] = font_file_opt

    if also_export_png:
        with st.spinner("\u91cd\u65b0\u7522\u751f\u5716\u8868..."):
            plot_market(revenue_market, revenue_base_date, out_path)
            plot_ytd_yoy(revenue_market, out_path)
            plot_revenue_ma(revenue_market, out_path)
            if show_industry:
                plot_industry_facets(revenue_df, out_path)
            if market_ex_fin is not None and base_date_ex_fin is not None:
                plot_market(market_ex_fin, base_date_ex_fin, out_path, suffix=EX_FIN_SUFFIX)
                plot_ytd_yoy(market_ex_fin, out_path, suffix=EX_FIN_SUFFIX)
                plot_revenue_ma(market_ex_fin, out_path, suffix=EX_FIN_SUFFIX)
                if show_industry and df_ex_fin is not None and not df_ex_fin.empty:
                    plot_industry_facets(df_ex_fin, out_path, suffix=EX_FIN_SUFFIX)

    macro_result = None
    try:
        with st.spinner("\u4e0b\u8f09\u666f\u6c23\u6307\u6a19..."):
            macro_result = build_macro_business_cycle(
                start=macro_start,
                end=macro_end,
                out_dir=out_path,
                verify_ssl=True,
                allow_insecure_fallback=True,
                font_family=DEFAULT_FONT,
            )
    except Exception as exc:
        st.error(f"Macro data download failed: {exc}")
        macro_result = None

    if macro_result is not None:
        st.session_state["macro_wide_df"] = macro_result["df_wide"]
        st.session_state["macro_yoy_df"] = macro_result["df_yoy"]
        st.session_state["macro_latest_dict"] = macro_result["latest"].to_dict()
        st.session_state["macro_latest_yoy_dict"] = macro_result["latest_yoy"].to_dict()
        st.session_state["macro_latest_period"] = macro_result["latest_period"]
        st.session_state["macro_out_dir"] = out_path

revenue_market = st.session_state.get("revenue_market_df")
revenue_base_date = st.session_state.get("revenue_base_date")
revenue_raw = st.session_state.get("revenue_raw_df")
revenue_market_ex = st.session_state.get("revenue_market_ex_fin_df")
revenue_base_date_ex = st.session_state.get("revenue_base_date_ex_fin")
revenue_raw_ex = st.session_state.get("revenue_raw_df_ex_fin")
revenue_font_file = st.session_state.get("revenue_font_file")

macro_wide = st.session_state.get("macro_wide_df")
macro_yoy = st.session_state.get("macro_yoy_df")
macro_latest = st.session_state.get("macro_latest_dict")
macro_latest_yoy = st.session_state.get("macro_latest_yoy_dict")
macro_latest_period = st.session_state.get("macro_latest_period")

tab_revenue, tab_macro = st.tabs(["\u4e0a\u5e02\u6ac3\u71df\u6536", "\u666f\u6c23\u6307\u6a19"])

with tab_revenue:
    if revenue_market is None or revenue_base_date is None or revenue_raw is None:
        st.info("\u8acb\u5148\u5728\u5de6\u5074\u8f38\u5165 CSV \u8def\u5f91\u4e26\u57f7\u884c\u300c\u5efa\u7acb / \u66f4\u65b0 \u5716\u8868\u300d\u3002")
    else:
        configure_cjk_font(DEFAULT_FONT, revenue_font_file)

        dataset_options: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
            "\u6574\u9ad4\u5e02\u5834": {
                "market": revenue_market,
                "base": revenue_base_date,
                "raw": revenue_raw,
            }
        }
        if (
            revenue_market_ex is not None
            and revenue_base_date_ex is not None
            and revenue_raw_ex is not None
        ):
            dataset_options["\u6392\u9664\u91d1\u878d\u4fdd\u96aa"] = {
                "market": revenue_market_ex,
                "base": revenue_base_date_ex,
                "raw": revenue_raw_ex,
            }

        selection = st.radio(
            "\u8cc7\u6599\u7bc4\u570d",
            list(dataset_options.keys()),
            horizontal=True,
            key="revenue_scope",
        )
        current = dataset_options[selection]
        market_df = current["market"]
        base_date = current["base"]
        raw_df = current["raw"]

        if market_df is None or base_date is None or raw_df is None:
            st.error("\u8cc7\u6599\u8f09\u5165\u5931\u6557\uff0c\u8acb\u91cd\u65b0\u5efa\u7acb\u5716\u8868\u3002")
        else:
            latest_period = market_df.index.max()
            latest_row = market_df.loc[latest_period]
            st.subheader(f"\u6700\u8fd1\u6708\u4efd\uff1a{latest_period.strftime('%Y-%m')}｜{selection}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("YoY", fmt_pct(latest_row.get("yoy_pct")))
            c2.metric("YoY 3M Avg", fmt_pct(latest_row.get("yoy_pct_smooth_3m")))
            c3.metric("3M Avg Revenue", fmt_num(latest_row.get("avg_revenue_3m")))
            c4.metric("6M Avg Revenue", fmt_num(latest_row.get("avg_revenue_6m")))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("YTD Revenue", fmt_num(latest_row.get("ytd_total")))
            c6.metric("YTD YoY", fmt_pct(latest_row.get("ytd_yoy_pct")))
            c7.metric("YTD YoY 3M Avg", fmt_pct(latest_row.get("ytd_yoy_pct_avg_3m")))
            c8.metric("YTD YoY 6M Avg", fmt_pct(latest_row.get("ytd_yoy_pct_avg_6m")))

            st.metric(
                f"\u71df\u6536\u6307\u6578 (\u57fa\u671f {base_date.strftime('%Y-%m')} = 100)",
                fmt_num(latest_row.get("index_2019_12_100")),
            )

            st.divider()
            st.subheader("\u4ea4\u4e92\u5716\u8868")
            st.caption("\u52fe\u9078 Checkbox \u4ee5\u63a7\u5236\u7dda\u689d\u3002")

            market_plot_df = market_df.reset_index().rename(columns={"date": "月份"})

            with st.expander("\u7e3e\u671f YoY (\u542b 3M \u5e73\u6e05)", expanded=True):
                yoy_options = [
                    ("\u5e74\u589e\u7387 YoY", "yoy_pct", True),
                    ("\u5e74\u589e\u7387 YoY (3M Avg)", "yoy_pct_smooth_3m", True),
                ]
                selected_series = select_series(yoy_options, f"revenue_yoy_{selection}")
                if not selected_series:
                    st.info("\u8acb\u81f3\u5c11\u52fe\u9078\u4e00\u689d\u7dda\u3002")
                else:
                    render_chart(
                        build_line_chart(
                            market_plot_df[["月份"] + [col for _, col in selected_series]],
                            selected_series,
                            "YoY",
                            "YoY (%)",
                            ".2f",
                            y_tickformat=".1f",
                        )
                    )

            with st.expander(
                f"\u71df\u6536\u6307\u6578 (\u57fa\u671f {base_date.strftime('%Y-%m')} = 100)", expanded=True
            ):
                idx_options = [("\u71df\u6536\u6307\u6578", "index_2019_12_100", True)]
                selected_series = select_series(idx_options, f"revenue_index_{selection}")
                if not selected_series:
                    st.info("\u8acb\u81f3\u5c11\u52fe\u9078\u4e00\u689d\u7dda\u3002")
                else:
                    render_chart(
                        build_line_chart(
                            market_plot_df[["月份"] + [col for _, col in selected_series]],
                            selected_series,
                            "\u71df\u6536\u6307\u6578",
                            "\u6307\u6578",
                            ",.1f",
                            y_tickformat=",.1f",
                        )
                    )

            with st.expander("YTD YoY", expanded=True):
                ytd_options = [
                    ("YTD YoY", "ytd_yoy_pct", True),
                    ("YTD YoY (3M Avg)", "ytd_yoy_pct_avg_3m", True),
                    ("YTD YoY (6M Avg)", "ytd_yoy_pct_avg_6m", True),
                ]
                selected_series = select_series(ytd_options, f"revenue_ytd_{selection}")
                if not selected_series:
                    st.info("\u8acb\u81f3\u5c11\u52fe\u9078\u4e00\u689d\u7dda\u3002")
                else:
                    render_chart(
                        build_line_chart(
                            market_plot_df[["月份"] + [col for _, col in selected_series]],
                            selected_series,
                            "YTD YoY",
                            "YoY (%)",
                            ".2f",
                            y_tickformat=".1f",
                        )
                    )

            with st.expander("3 / 6 \u500b\u6708\u5e73\u5747\u71df\u6536", expanded=False):
                ma_options = [
                    ("3M Avg Revenue", "avg_revenue_3m", True),
                    ("6M Avg Revenue", "avg_revenue_6m", True),
                    ("Monthly Revenue", "total_revenue", False),
                ]
                selected_series = select_series(ma_options, f"revenue_ma_{selection}")
                if not selected_series:
                    st.info("\u8acb\u81f3\u5c11\u52fe\u9078\u4e00\u689d\u7dda\u3002")
                else:
                    render_chart(
                        build_line_chart(
                            market_plot_df[["月份"] + [col for _, col in selected_series]],
                            selected_series,
                            "Revenue Moving Average",
                            "Amount",
                            ",.0f",
                        )
                    )

            with st.expander("\u7522\u696d YoY", expanded=False):
                if "industry" not in raw_df.columns:
                    st.info("\u8cc7\u6599\u7f3a\u5c11\u300c\u7522\u696d\u5225\u300d\u6b04\u4f4d\u3002")
                else:
                    grouped = raw_df.groupby(["industry", "date"], as_index=False)["revenue"].sum()
                    total_by_ind = grouped.groupby("industry")["revenue"].sum().sort_values(ascending=False)
                    top_n = int(
                        st.number_input(
                            "Top N \u7522\u696d (by revenue)",
                            min_value=3,
                            max_value=30,
                            value=12,
                            step=1,
                        )
                    )
                    industries = list(total_by_ind.head(top_n).index)
                    if not industries:
                        st.info("\u627e\u4e0d\u5230\u7b26\u5408\u689d\u4ef6\u7684\u7522\u696d\u3002")
                    else:
                        default_inds = industries[: min(6, len(industries))]
                        selected_inds = st.multiselect(
                            "\u9078\u64c7\u7522\u696d",
                            options=industries,
                            default=default_inds,
                            key=f"industry_{selection}",
                        )
                        if not selected_inds:
                            st.info("\u8acb\u81f3\u5c11\u9078\u64c7\u4e00\u500b\u7522\u696d\u3002")
                        else:
                            sub = grouped[grouped["industry"].isin(selected_inds)].copy()
                            sub.sort_values(["industry", "date"], inplace=True)
                            sub["yoy_pct"] = sub.groupby("industry")["revenue"].pct_change(12) * 100.0
                            sub = sub[pd.notna(sub["yoy_pct"])]
                            if sub.empty:
                                st.info("\u6c92\u6709\u8db3 12 \u500b\u6708\u6bd4\u8f03\u7684\u8cc7\u6599\u3002")
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
                                selected_series = select_series(
                                    display_options, f"revenue_industry_{selection}"
                                )
                                if not selected_series:
                                    st.info("\u8acb\u81f3\u5c11\u52fe\u9078\u4e00\u689d\u7dda\u3002")
                                else:
                                    render_chart(
                                        build_line_chart(
                                            chart_data[["月份"] + [col for _, col in selected_series]],
                                            selected_series,
                                            "\u7522\u696d YoY",
                                            "YoY (%)",
                                            ".2f",
                                            y_tickformat=".1f",
                                        )
                                    )

with tab_macro:
    if macro_wide is None or macro_yoy is None or macro_latest is None or macro_latest_period is None:
        st.info("\u5c1a\u672a\u4e0b\u8f09\u666f\u6c23\u6307\u6a19\u8cc7\u6599\uff0c\u8acb\u4f7f\u7528\u5de6\u5074\u6309\u9215\u5efa\u7acb\u3002")
    else:
        latest_period_str = macro_latest_period.strftime("%Y-%m")
        st.subheader(f"\u6700\u65b0\u6708\u4efd\uff1a{latest_period_str}")

        latest_series = macro_latest
        latest_yoy_series = macro_latest_yoy or {}

        cols = st.columns(len(latest_series))
        for col_idx, (name, value) in enumerate(latest_series.items()):
            yoy_value = fmt_pct(latest_yoy_series.get(name))
            cols[col_idx].metric(name, fmt_num(value), delta=yoy_value)

        st.divider()
        st.subheader("\u666f\u6c23\u6307\u6a19\u5716\u8868")
        st.caption("\u591a\u689d\u7dda\u63a7\u5236\u4f9d\u7167\u7b2c\u4e00\u9801\u7c3d\u6a21\u5f0f\u8a2d\u8a08\u3002")

        macro_plot_df = macro_wide.reset_index().rename(columns={"date": "月份"})
        macro_yoy_plot_df = macro_yoy.reset_index().rename(columns={"date": "月份"})

        index_columns = [col for col in macro_plot_df.columns if col != "月份" and "信號" not in col]
        yoy_columns = [col for col in macro_yoy_plot_df.columns if col != "月份" and "信號" not in col]

        with st.expander("\u6307\u6a19\u539f\u503c", expanded=True):
            options = [(col, col, True) for col in index_columns]
            selected_series = select_series(options, "macro_index")
            if not selected_series:
                st.info("\u8acb\u81f3\u5c11\u52fe\u9078\u4e00\u689d\u7dda\u3002")
            else:
                render_chart(
                    build_line_chart(
                        macro_plot_df[["月份"] + [col for _, col in selected_series]],
                        selected_series,
                        "\u666f\u6c23\u6307\u6a19",
                        "\u6307\u6578",
                        ".2f",
                    )
                )

        with st.expander("YoY", expanded=True):
            options = [(col, col, True) for col in yoy_columns]
            selected_series = select_series(options, "macro_yoy")
            if not selected_series:
                st.info("\u8acb\u81f3\u5c11\u52fe\u9078\u4e00\u689d\u7dda\u3002")
            else:
                render_chart(
                    build_line_chart(
                        macro_yoy_plot_df[["月份"] + [col for _, col in selected_series]],
                        selected_series,
                        "\u666f\u6c23\u6307\u6a19 YoY",
                        "YoY (%)",
                        ".2f",
                        y_tickformat=".1f",
                    )
                )

        if "景氣對策信號(分)" in macro_plot_df.columns:
            with st.expander("\u666f\u6c23\u5c0d\u7b56\u4fe1\u865f", expanded=False):
                signal_df = macro_plot_df[["月份", "景氣對策信號(分)"]].copy()
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=signal_df["月份"],
                        y=signal_df["景氣對策信號(分)"],
                        mode="lines",
                        line=dict(shape="hv", color="#d62728"),
                        name="\u666f\u6c23\u5c0d\u7b56\u4fe1\u865f",
                        hovertemplate="%{x|%Y-%m}<br>Score: %{y:.0f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="\u666f\u6c23\u5c0d\u7b56\u4fe1\u865f",
                    font=dict(family=DEFAULT_FONT, size=13),
                    xaxis=dict(title="\u6708\u4efd"),
                    yaxis=dict(title="\u5206\u6578"),
                    template="plotly_white",
                )
                render_chart(fig)

st.success("\u5982\u9700\u91cd\u767c\u5716\u8868\u6216\u8cc7\u6e90\uff0c\u8acb\u4f7f\u7528\u5de6\u5074\u6309\u9215\u91cd\u65b0\u5efa\u7acb\u3002")
