import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import warnings

# Selected CJK font family determined at runtime
CJK_FONT_FAMILY: Optional[str] = None


def _format_png_name(base: str, suffix: str) -> str:
    """Return a png filename with optional suffix like '_ex_fin'."""
    return f"{base}{suffix}.png" if suffix else f"{base}.png"


def _try_imports():
    try:
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        return pd, plt, sns
    except Exception as e:
        print("Missing dependencies. Please install with: pip install -r requirements.txt", file=sys.stderr)
        raise


def _read_csv_best_effort(pd, path: Path):
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "big5hkscs", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except Exception as e:
            last_err = e
    # Last attempt: engine python with more tolerance
    try:
        df = pd.read_csv(path, encoding="latin1", engine="python")
        return df
    except Exception:
        pass
    raise RuntimeError(f"Unable to read CSV with common encodings. Last error: {last_err}")


def _normalize_columns(cols: List[str]) -> List[str]:
    # Trim whitespace and unify visually similar chars
    out = []
    for c in cols:
        if not isinstance(c, str):
            c = str(c)
        out.append(c.strip())
    return out


def _find_col(cols: List[str], keywords: List[str]) -> Optional[str]:
    for kw in keywords:
        for c in cols:
            if kw in c:
                return c
    return None


def _infer_schema(df):
    cols = _normalize_columns(list(df.columns))
    df.columns = cols

    # Likely candidates based on MOPS monthly revenue CSV
    col_release = _find_col(cols, ["發布日期", "發佈日期", "出表日", "出表日期", "公告日期", "資料日期"])
    col_year = None
    col_month = None

    # Exact '年'/'月' if present
    if "年" in cols:
        col_year = "年"
    if "月" in cols:
        col_month = "月"

    # 資料年月 as ROC 'YYY/M' or 'YYY/MM'
    col_roc_ym = _find_col(cols, ["資料年月", "年月", "統計年月"])

    # Revenue column
    rev_candidates = [
        "當月營收-當月金額",
        "本月營收-本月金額",
        "當月營收",
        "本月營收",
        "月營收",
        "營收金額",
        "營收",
    ]
    col_revenue = None
    for c in cols:
        for kw in rev_candidates:
            if kw in c:
                col_revenue = c
                break
        if col_revenue:
            break

    col_industry = _find_col(cols, ["產業別", "產業分類", "產業"])
    col_code = _find_col(cols, ["公司代號", "股票代號", "證券代號"])
    col_name = _find_col(cols, ["公司名稱", "證券名稱"])

    return {
        "release": col_release,
        "year": col_year,
        "month": col_month,
        "roc_ym": col_roc_ym,
        "revenue": col_revenue,
        "industry": col_industry,
        "code": col_code,
        "name": col_name,
        "cols": cols,
    }


def _parse_roc_year_month_to_date(pd, year: int, month: int):
    # ROC year to AD
    ad_year = int(year) + 1911
    m = int(month)
    return pd.Timestamp(year=ad_year, month=m, day=1)


def _parse_roc_date_str(pd, s: str):
    # Accept formats like '114/08/23' or '114/8/23'
    parts = str(s).split('/')
    if len(parts) >= 3:
        y, m, d = parts[:3]
        try:
            return pd.Timestamp(year=int(y) + 1911, month=int(m), day=int(d))
        except Exception:
            return pd.NaT
    return pd.NaT


def _ensure_numeric(pd, s):
    # Remove commas and coercing to numeric
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_and_prepare(path: Path):
    pd, _, _ = _try_imports()
    df = _read_csv_best_effort(pd, path)
    schema = _infer_schema(df)

    if not schema["revenue"]:
        raise RuntimeError("找不到營收欄位（例如 '當月營收-當月金額'）。請確認檔案欄位名稱。")

    # Build year-month
    if schema["year"] and schema["month"]:
        years = df[schema["year"]]
        months = df[schema["month"]]
        # coerce numeric
        # Some inputs may be strings
        try:
            years = years.astype(int)
            months = months.astype(int)
        except Exception:
            years = _ensure_numeric(pd, years).astype("Int64")
            months = _ensure_numeric(pd, months).astype("Int64")
        df["date"] = [
            _parse_roc_year_month_to_date(pd, y, m)
            if (pd.notna(y) and pd.notna(m)) else pd.NaT
            for y, m in zip(years, months)
        ]
    elif schema["roc_ym"]:
        # Expect 'YYY/M' style
        ym = df[schema["roc_ym"]].astype(str)
        def parse_ym(x: str):
            parts = x.split('/')
            if len(parts) >= 2:
                y, m = parts[:2]
                try:
                    return _parse_roc_year_month_to_date(pd, int(y), int(m))
                except Exception:
                    return pd.NaT
            return pd.NaT
        df["date"] = ym.map(parse_ym)
    else:
        raise RuntimeError("找不到年月欄位（'年'/'月' 或 '資料年月'）。")

    # Release date for deduplication if present
    if schema["release"]:
        rel = df[schema["release"]]
        def to_dt(x):
            try:
                return _parse_roc_date_str(pd, x)
            except Exception:
                return pd.NaT
        df["release_dt"] = rel.map(to_dt)
    else:
        df["release_dt"] = pd.NaT

    # Revenue numeric
    df["revenue"] = _ensure_numeric(pd, df[schema["revenue"]])

    # Select essential columns
    keep = ["date", "release_dt", "revenue"]
    if schema["industry"]:
        df["industry"] = df[schema["industry"]].astype(str)
        keep.append("industry")
    if schema["code"]:
        df["code"] = df[schema["code"]].astype(str)
        keep.append("code")
    if schema["name"]:
        df["name"] = df[schema["name"]].astype(str)
        keep.append("name")

    df = df[keep]

    # Drop rows without date or revenue
    df = df[pd.notna(df["date"]) & pd.notna(df["revenue"])]

    # Deduplicate by code+date using latest release_dt if available
    if "code" in df.columns and "release_dt" in df.columns:
        df = df.sort_values(["code", "date", "release_dt"]).drop_duplicates(subset=["code", "date"], keep="last")

    return df


def aggregate_market_series(pd, df):
    # Sum revenue by month across all companies
    s = df.groupby("date", as_index=True)["revenue"].sum().sort_index()
    market = pd.DataFrame({"total_revenue": s})
    # Year-over-year based on monthly totals
    market["yoy_pct"] = market["total_revenue"].pct_change(12) * 100.0
    market["yoy_pct_smooth_3m"] = market["yoy_pct"].rolling(3, min_periods=1).mean()

    # 3M/6M average revenue (simple moving average)
    market["avg_revenue_3m"] = market["total_revenue"].rolling(3, min_periods=1).mean()
    market["avg_revenue_6m"] = market["total_revenue"].rolling(6, min_periods=1).mean()

    # YTD cumulative revenue and its YoY
    idx = market.index
    years = idx.year
    # cumsum by calendar year
    market["ytd_total"] = (
        market["total_revenue"].groupby(years).cumsum()
    )
    # YTD YoY = yoy on ytd_total (shift by 12 months)
    market["ytd_yoy_pct"] = market["ytd_total"].pct_change(12) * 100.0
    # 3M/6M averages of YTD YoY
    market["ytd_yoy_pct_avg_3m"] = market["ytd_yoy_pct"].rolling(3, min_periods=1).mean()
    market["ytd_yoy_pct_avg_6m"] = market["ytd_yoy_pct"].rolling(6, min_periods=1).mean()
    # Build 2019-12 base if available, else first
    base_date = pd.Timestamp("2019-12-01")
    if base_date in market.index:
        base_val = market.loc[base_date, "total_revenue"]
    else:
        base_val = market["total_revenue"].iloc[0]
        base_date = market.index[0]
    market["index_2019_12_100"] = (market["total_revenue"] / base_val) * 100.0
    return market, base_date


def _filter_out_finance(df):
    """Return a dataframe excluding rows tagged as finance/insurance, or None."""
    if "industry" not in df.columns:
        return None
    industries = df["industry"].astype(str)
    mask_fin = industries.str.contains("金融", na=False)
    mask_fin = mask_fin | industries.str.contains("保險", na=False)
    if not mask_fin.any():
        return None
    filtered = df[~mask_fin].copy()
    if filtered.empty:
        return None
    return filtered


def plot_market(market, base_date: "pd.Timestamp", out_dir: Path, suffix: str = ""):
    pd, plt, sns = _try_imports()
    if CJK_FONT_FAMILY:
        sns.set_theme(style="whitegrid", rc={
            "font.family": CJK_FONT_FAMILY,
            "font.sans-serif": [CJK_FONT_FAMILY],
            "font.serif": [CJK_FONT_FAMILY],
            "font.cursive": [CJK_FONT_FAMILY],
            "font.fantasy": [CJK_FONT_FAMILY],
            "font.monospace": [CJK_FONT_FAMILY],
            "axes.unicode_minus": False,
        })
    else:
        sns.set_theme(style="whitegrid")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: YoY with 3M smoothing
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(market.index, market["yoy_pct"], label="年增率（當月）", color="#1f77b4", alpha=0.7)
    ax.plot(market.index, market["yoy_pct_smooth_3m"], label="年增率（3個月平均）", color="#d62728", linewidth=2)
    ax.axhline(0, color="#888", linewidth=1)
    ax.set_title("上市櫃整體月營收 年增率（含3個月平滑）")
    ax.set_ylabel("年增率 %")
    ax.set_xlabel("月份")
    # 移除圖例以簡化版面
    fig.tight_layout()
    p1 = out_dir / _format_png_name("market_yoy", suffix)
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # Chart 2: Index (2019-12=100 or first)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(market.index, market["index_2019_12_100"], color="#2ca02c")
    ax.set_title(f"上市櫃整體月營收 指數（基期 {base_date.strftime('%Y-%m')}=100）")
    ax.set_ylabel("指數")
    ax.set_xlabel("月份")
    fig.tight_layout()
    p2 = out_dir / _format_png_name("market_index", suffix)
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    return p1, p2


def plot_industry_facets(df, out_dir: Path, top_n: int = 12, suffix: str = ""):
    pd, plt, sns = _try_imports()
    if "industry" not in df.columns:
        return None
    if CJK_FONT_FAMILY:
        sns.set_theme(style="whitegrid", rc={
            "font.family": CJK_FONT_FAMILY,
            "font.sans-serif": [CJK_FONT_FAMILY],
            "font.serif": [CJK_FONT_FAMILY],
            "font.cursive": [CJK_FONT_FAMILY],
            "font.fantasy": [CJK_FONT_FAMILY],
            "font.monospace": [CJK_FONT_FAMILY],
            "axes.unicode_minus": False,
        })
    else:
        sns.set_theme(style="whitegrid")

    # Aggregate revenue by industry-month
    g = df.groupby(["industry", "date"], as_index=False)["revenue"].sum()

    # Compute market share per industry over the whole period
    total_by_ind = g.groupby("industry")["revenue"].sum().sort_values(ascending=False)
    top_inds = list(total_by_ind.head(top_n).index)
    g_top = g[g["industry"].isin(top_inds)].copy()

    # Compute YoY by industry
    def calc_yoy(x):
        x = x.sort_values("date").set_index("date")
        x["yoy_pct"] = x["revenue"].pct_change(12) * 100.0
        return x.reset_index()
    g_top = g_top.groupby("industry", group_keys=False).apply(calc_yoy)

    # Faceted line plots
    n = len(top_inds)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ind in enumerate(top_inds):
        ax = axes[i]
        sub = g_top[g_top["industry"] == ind]
        ax.plot(sub["date"], sub["yoy_pct"], color="#1f77b4")
        ax.axhline(0, color="#aaa", linewidth=0.8)
        ax.set_title(str(ind))
        if i % cols == 0:
            ax.set_ylabel("年增率 %")
    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("分產業月營收年增率（前12大產業）", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / _format_png_name("industry_yoy_top12", suffix)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_ytd_yoy(market, out_dir: Path, suffix: str = ""):
    pd, plt, sns = _try_imports()
    if CJK_FONT_FAMILY:
        sns.set_theme(style="whitegrid", rc={
            "font.family": CJK_FONT_FAMILY,
            "font.sans-serif": [CJK_FONT_FAMILY],
            "font.serif": [CJK_FONT_FAMILY],
            "font.cursive": [CJK_FONT_FAMILY],
            "font.fantasy": [CJK_FONT_FAMILY],
            "font.monospace": [CJK_FONT_FAMILY],
            "axes.unicode_minus": False,
        })
    else:
        sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(market.index, market["ytd_yoy_pct"], label="累計營收年增率（當月）", color="#1f77b4", alpha=0.7)
    ax.plot(market.index, market["ytd_yoy_pct_avg_3m"], label="累計營收年增率（3個月平均）", color="#d62728", linewidth=2)
    ax.plot(market.index, market["ytd_yoy_pct_avg_6m"], label="累計營收年增率（6個月平均）", color="#9467bd", linewidth=2)
    ax.axhline(0, color="#888", linewidth=1)
    ax.set_title("上市櫃整體累計(年初至今)營收 年增率")
    ax.set_ylabel("年增率 %")
    ax.set_xlabel("月份")
    ax.legend()
    fig.tight_layout()
    p = out_dir / _format_png_name("market_ytd_yoy", suffix)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_revenue_ma(market, out_dir: Path, suffix: str = ""):
    pd, plt, sns = _try_imports()
    if CJK_FONT_FAMILY:
        sns.set_theme(style="whitegrid", rc={
            "font.family": CJK_FONT_FAMILY,
            "font.sans-serif": [CJK_FONT_FAMILY],
            "font.serif": [CJK_FONT_FAMILY],
            "font.cursive": [CJK_FONT_FAMILY],
            "font.fantasy": [CJK_FONT_FAMILY],
            "font.monospace": [CJK_FONT_FAMILY],
            "axes.unicode_minus": False,
        })
    else:
        sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(market.index, market["avg_revenue_3m"], label="3個月平均營收", color="#2ca02c", linewidth=2)
    ax.plot(market.index, market["avg_revenue_6m"], label="6個月平均營收", color="#ff7f0e", linewidth=2)
    ax.plot(market.index, market["total_revenue"], label="當月營收", color="#999999", alpha=0.4)
    ax.set_title("上市櫃整體 3個月/6個月 平均營收 與 當月營收")
    ax.set_ylabel("金額")
    ax.set_xlabel("月份")
    ax.legend()
    fig.tight_layout()
    p = out_dir / _format_png_name("market_revenue_ma", suffix)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def print_stream_ui(market, base_date):
    pd, _, _ = _try_imports()
    # 最新期
    last_idx = market.index.max()
    row = market.loc[last_idx]

    def fmt_num(x):
        try:
            return f"{x:,.0f}"
        except Exception:
            return "-"

    def fmt_pct(x):
        try:
            return f"{x:,.2f}%"
        except Exception:
            return "-"

    print("== 上市櫃整體月營收：指標摘要 ==")
    print(f"最新月份：{last_idx.strftime('%Y-%m')}")
    print("[月度年增率]")
    print(f"  年增率（當月）：{fmt_pct(row.get('yoy_pct'))}")
    print(f"  年增率（3個月平均）：{fmt_pct(row.get('yoy_pct_smooth_3m'))}")
    print("[平均營收]")
    print(f"  3個月平均營收：{fmt_num(row.get('avg_revenue_3m'))}")
    print(f"  6個月平均營收：{fmt_num(row.get('avg_revenue_6m'))}")
    print("[累計（年初至今）]")
    print(f"  累計營收（YTD）：{fmt_num(row.get('ytd_total'))}")
    print(f"  累計營收年增率（當月）：{fmt_pct(row.get('ytd_yoy_pct'))}")
    print(f"  累計營收年增率（3個月平均）：{fmt_pct(row.get('ytd_yoy_pct_avg_3m'))}")
    print(f"  累計營收年增率（6個月平均）：{fmt_pct(row.get('ytd_yoy_pct_avg_6m'))}")
    print("[指數]")
    print(f"  指數（基期 {base_date.strftime('%Y-%m')}=100）：{fmt_num(row.get('index_2019_12_100'))}")


def main():
    pd, _, _ = _try_imports()
    ap = argparse.ArgumentParser(description="Aggregate Taiwan MOPS monthly revenue and plot YoY / Index charts.")
    ap.add_argument("input", type=str, help="Path to MOPS monthly revenue CSV (107-114)")
    ap.add_argument("--out", type=str, default="out", help="Output folder for charts")
    ap.add_argument("--no-industry", action="store_true", help="Skip industry facet chart")
    ap.add_argument("--stream", action="store_true", help="以文字輸出關鍵指標，適合串流介面（不影響圖檔輸出）")
    ap.add_argument("--font", type=str, default="auto", help="強制使用指定字型（例如：Microsoft JhengHei）。預設 auto 將自動挑選常見中文字型。")
    ap.add_argument("--font-file", type=str, default=None, help="自訂字型檔案路徑（.ttf/.otf/.ttc），會自動註冊並優先使用。")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)

    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    warnings.simplefilter("ignore", category=FutureWarning)

    # Configure CJK-capable font to avoid missing glyphs
    configure_cjk_font(args.font, args.font_file)

    df = load_and_prepare(in_path)
    market, base_date = aggregate_market_series(pd, df)

    df_ex_fin = _filter_out_finance(df)
    market_ex_fin = None
    base_date_ex_fin = None
    if df_ex_fin is not None:
        market_ex_fin, base_date_ex_fin = aggregate_market_series(pd, df_ex_fin)

    p1, p2 = plot_market(market, base_date, out_dir)
    p4 = plot_ytd_yoy(market, out_dir)
    p5 = plot_revenue_ma(market, out_dir)
    p3 = None
    if not args.no_industry:
        p3 = plot_industry_facets(df, out_dir)

    ex_paths: List[Path] = []
    if market_ex_fin is not None and base_date_ex_fin is not None:
        suffix = "_ex_fin"
        ex_p1, ex_p2 = plot_market(market_ex_fin, base_date_ex_fin, out_dir, suffix=suffix)
        ex_paths.extend([ex_p1, ex_p2])
        if not args.no_industry:
            ex_p3 = plot_industry_facets(df_ex_fin, out_dir, suffix=suffix)
            if ex_p3 is not None:
                ex_paths.append(ex_p3)
        ex_p4 = plot_ytd_yoy(market_ex_fin, out_dir, suffix=suffix)
        ex_p5 = plot_revenue_ma(market_ex_fin, out_dir, suffix=suffix)
        ex_paths.extend([ex_p4, ex_p5])

    if args.stream:
        print_stream_ui(market, base_date)
        if market_ex_fin is not None:
            print()
            print("[排除金融保險業]")
            print_stream_ui(market_ex_fin, base_date_ex_fin)
    else:
        generated: List[Path] = [p1, p2]
        if p3 is not None:
            generated.append(p3)
        generated.extend([p4, p5])
        generated.extend(ex_paths)
        print("Generated:")
        for path in generated:
            print(str(path))


def configure_cjk_font(requested: str = "auto", font_file: Optional[str] = None):
    """Force a CJK-capable font on Matplotlib to prevent missing glyph warnings.
    If requested is a specific family name, try to use it. Otherwise, pick the
    first available from a candidate list. Works best on Windows.
    """
    global CJK_FONT_FAMILY
    try:
        import matplotlib as mpl
        from matplotlib.font_manager import FontProperties, findfont, fontManager
        from pathlib import Path as _Path

        # If a specific font file is provided, register and use it directly
        if font_file:
            fp = _Path(font_file)
            if fp.exists():
                try:
                    fontManager.addfont(str(fp))
                    fam = FontProperties(fname=str(fp)).get_name()
                    CJK_FONT_FAMILY = fam
                    mpl.rcParams.update({
                        "font.family": fam,
                        "font.sans-serif": [fam],
                        "font.serif": [fam],
                        "font.cursive": [fam],
                        "font.fantasy": [fam],
                        "font.monospace": [fam],
                        "axes.unicode_minus": False,
                    })
                    return
                except Exception:
                    # If registration fails, continue with auto selection
                    pass

        candidates = [
            # Common on Windows (Traditional/Sim)
            "Microsoft JhengHei", "PMingLiU", "MingLiU", "SimHei", "SimSun",
            # Modern Windows CJK
            "Microsoft YaHei", "Microsoft YaHei UI", "Yu Gothic UI", "Meiryo",
            # Apple / Noto (if installed locally)
            "PingFang TC", "PingFang HK", "Noto Sans CJK TC", "Arial Unicode MS",
        ]

        preferred = []
        if requested and requested.lower() != "auto":
            preferred = [requested]
        names = preferred + candidates

        selected = None
        for name in names:
            try:
                fp = FontProperties(family=name)
                path = findfont(fp, fallback_to_default=False)
                if path and _Path(path).exists():
                    selected = name
                    break
            except Exception:
                continue

        # As a fallback, try registering common Windows font files explicitly
        if not selected:
            for p in [
                r"C:\\Windows\\Fonts\\msjh.ttc",
                r"C:\\Windows\\Fonts\\msjhbd.ttc",
                r"C:\\Windows\\Fonts\\msyh.ttc",
                r"C:\\Windows\\Fonts\\msyhbd.ttc",
                r"C:\\Windows\\Fonts\\mingliu.ttc",
                r"C:\\Windows\\Fonts\\pmingliu.ttc",
                r"C:\\Windows\\Fonts\\simsun.ttc",
                r"C:\\Windows\\Fonts\\simhei.ttf",
                r"C:\\Windows\\Fonts\\YuGothR.ttc",
                r"C:\\Windows\\Fonts\\meiryo.ttc",
            ]:
                try:
                    path_obj = _Path(p)
                    if path_obj.exists():
                        mpl.font_manager.fontManager.addfont(str(path_obj))
                except Exception:
                    pass
            # Re-try selection
            for name in names:
                try:
                    fp = FontProperties(family=name)
                    path = findfont(fp, fallback_to_default=False)
                    if path and _Path(path).exists():
                        selected = name
                        break
                except Exception:
                    continue

        if selected:
            CJK_FONT_FAMILY = selected
            # Apply to all generic families to avoid fallback to Arial
            mpl.rcParams.update({
                "font.family": selected,
                "font.sans-serif": [selected],
                "font.serif": [selected],
                "font.cursive": [selected],
                "font.fantasy": [selected],
                "font.monospace": [selected],
                "axes.unicode_minus": False,
            })
        else:
            # Keep sans-serif but disable unicode minus to avoid glyph issues
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "axes.unicode_minus": False,
            })
    except Exception:
        # Silent failure: do not break plotting
        pass


if __name__ == "__main__":
    main()
