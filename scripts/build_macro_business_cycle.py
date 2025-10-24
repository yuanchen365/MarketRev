from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

BASE_URL = "https://nstatdb.dgbas.gov.tw/dgbasall/webMain.aspx"
DATASET_ID = "a120101010"
DEFAULT_SERIES_IDS = ("1", "2", "3", "4", "5")
DEFAULT_FREQUENCY = "M"


class MacroDataError(RuntimeError):
    """Raised when macro data cannot be fetched or parsed."""


@dataclass(frozen=True)
class MacroSeriesMeta:
    series_id: str
    name: str


def _build_series_path(series_ids: Iterable[str], frequency: str) -> str:
    if not series_ids:
        raise ValueError("series_ids must not be empty")
    return f"{'+'.join(series_ids)}...{frequency}."


def fetch_sdmx_json(
    series_ids: Iterable[str],
    start: str,
    end: str,
    *,
    frequency: str = DEFAULT_FREQUENCY,
    verify_ssl: bool = True,
    allow_insecure_fallback: bool = True,
    timeout: int = 30,
) -> dict:
    series_path = _build_series_path(series_ids, frequency)
    url = f"{BASE_URL}?sdmx/{DATASET_ID}/{series_path}"

    params = {"startTime": start, "endTime": end}

    try:
        resp = requests.get(url, params=params, timeout=timeout, verify=verify_ssl)
        resp.raise_for_status()
    except requests.exceptions.SSLError as exc:
        if verify_ssl and allow_insecure_fallback:
            warnings.warn(
                "SSL verification failed for DGBAS endpoint; retrying with "
                "certificate verification disabled. For production usage, "
                "supply a trusted CA bundle or set --no-insecure-fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            resp = requests.get(url, params=params, timeout=timeout, verify=False)
            resp.raise_for_status()
        else:
            raise MacroDataError("SSL verification failed and no fallback allowed") from exc
    except requests.RequestException as exc:  # pragma: no cover - network issues
        raise MacroDataError(f"Failed to download macro data: {exc}") from exc

    try:
        return resp.json()
    except ValueError as exc:  # pragma: no cover - malformed payload
        raise MacroDataError("Invalid JSON payload received from DGBAS") from exc


def _parse_sdmx_payload(payload: dict) -> Tuple[pd.DataFrame, List[MacroSeriesMeta]]:
    try:
        structure = payload["data"]["structure"]["dimensions"]
        dataset = payload["data"]["dataSets"][0]["series"]
    except KeyError as exc:  # pragma: no cover - structure change
        raise MacroDataError("Unexpected SDMX JSON structure") from exc

    series_values = structure["series"][0]["values"]
    observation_values = structure["observation"][0]["values"]

    series_meta = [
        MacroSeriesMeta(series_id=entry["id"], name=entry["name"])
        for entry in series_values
    ]

    records: List[dict] = []
    for series_key, data in dataset.items():
        meta = series_meta[int(series_key)]
        for obs_key, obs_value in data.get("observations", {}).items():
            period_info = observation_values[int(obs_key)]
            value = obs_value[0] if obs_value else None
            records.append(
                {
                    "series_id": meta.series_id,
                    "series_name": meta.name,
                    "period_id": period_info["id"],
                    "date": period_info["id"],
                    "value": value,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise MacroDataError("No observations returned for the requested period.")

    def _parse_period_to_timestamp(label: str) -> pd.Timestamp:
        try:
            year_part, month_part = label.split("-M")
            return pd.Timestamp(year=int(year_part), month=int(month_part), day=1)
        except Exception as exc:  # pragma: no cover - unexpected format
            raise MacroDataError(f"Invalid period format: {label}") from exc

    df["date"] = df["period_id"].map(_parse_period_to_timestamp)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.sort_values(["series_id", "date"], inplace=True)
    return df, series_meta


def create_wide_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="date", columns="series_name", values="value")
    return pivot.sort_index()


def compute_yoy(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(12) * 100.0


def _apply_font(font_family: Optional[str]) -> None:
    if font_family:
        plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False


def plot_indices(df: pd.DataFrame, out_path: Path, font_family: Optional[str]) -> Path:
    sns.set_theme(style="whitegrid")
    _apply_font(font_family)

    index_cols = [col for col in df.columns if "景氣對策信號" not in col]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for col in index_cols:
        ax.plot(df.index, df[col], label=col)
    ax.set_title("景氣指標（含趨勢/不含趨勢）")
    ax.set_ylabel("指數")
    ax.set_xlabel("月份")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_yoy(df_yoy: pd.DataFrame, out_path: Path, font_family: Optional[str]) -> Path:
    sns.set_theme(style="whitegrid")
    _apply_font(font_family)

    index_cols = [col for col in df_yoy.columns if "景氣對策信號" not in col]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for col in index_cols:
        ax.plot(df_yoy.index, df_yoy[col], label=col)
    ax.axhline(0, color="#888", linewidth=1)
    ax.set_title("景氣指標年增率（YoY）")
    ax.set_ylabel("年增率 (%)")
    ax.set_xlabel("月份")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_signal(df: pd.DataFrame, out_path: Path, font_family: Optional[str]) -> Path:
    if "景氣對策信號(分)" not in df.columns:
        return out_path

    sns.set_theme(style="whitegrid")
    _apply_font(font_family)

    signal = df["景氣對策信號(分)"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.step(signal.index, signal, where="post", color="#d62728")
    ax.set_title("景氣對策信號")
    ax.set_ylabel("分數")
    ax.set_xlabel("月份")
    ax.set_yticks(range(int(signal.min()), int(signal.max()) + 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_macro_business_cycle(
    start: str,
    end: str,
    out_dir: Path,
    *,
    series_ids: Iterable[str] = DEFAULT_SERIES_IDS,
    frequency: str = DEFAULT_FREQUENCY,
    verify_ssl: bool = True,
    allow_insecure_fallback: bool = True,
    font_family: Optional[str] = "Microsoft JhengHei",
) -> dict:
    payload = fetch_sdmx_json(
        series_ids,
        start,
        end,
        frequency=frequency,
        verify_ssl=verify_ssl,
        allow_insecure_fallback=allow_insecure_fallback,
    )
    df_long, _ = _parse_sdmx_payload(payload)
    df_wide = create_wide_dataframe(df_long)
    df_yoy = compute_yoy(df_wide)

    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "macro_business_cycle.csv"
    yoy_csv = out_dir / "macro_business_cycle_yoy.csv"
    df_wide.to_csv(raw_csv, encoding="utf-8-sig")
    df_yoy.to_csv(yoy_csv, encoding="utf-8-sig")

    index_plot = out_dir / "macro_business_cycle_index.png"
    yoy_plot = out_dir / "macro_business_cycle_yoy.png"
    signal_plot = out_dir / "macro_business_cycle_signal.png"

    plot_indices(df_wide, index_plot, font_family)
    plot_yoy(df_yoy, yoy_plot, font_family)
    plot_signal(df_wide, signal_plot, font_family)

    latest = df_wide.iloc[-1]
    latest_period = df_wide.index[-1]

    return {
        "raw_csv": raw_csv,
        "yoy_csv": yoy_csv,
        "index_plot": index_plot,
        "yoy_plot": yoy_plot,
        "signal_plot": signal_plot,
        "df_wide": df_wide,
        "df_yoy": df_yoy,
        "latest": latest,
        "latest_yoy": df_yoy.iloc[-1],
        "latest_period": latest_period,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and plot DGBAS macroeconomic business cycle indicators."
    )
    parser.add_argument("--start", default="2020-M1", help="起始時間（例如 2020-M1）")
    parser.add_argument("--end", default="2025-M8", help="結束時間（例如 2025-M8）")
    parser.add_argument("--out", default="out", help="輸出資料夾")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="停用 SSL 憑證驗證（預設會驗證並在失敗時自動退回不驗證模式）。",
    )
    parser.add_argument(
        "--no-insecure-fallback",
        action="store_true",
        help="禁止 SSL 驗證失敗時使用不驗證模式。",
    )
    parser.add_argument(
        "--font",
        default="Microsoft JhengHei",
        help="Matplotlib 字型（預設 Microsoft JhengHei）。",
    )
    args = parser.parse_args(argv)

    verify_ssl = not args.no_verify_ssl

    result = build_macro_business_cycle(
        start=args.start,
        end=args.end,
        out_dir=Path(args.out),
        verify_ssl=verify_ssl,
        allow_insecure_fallback=not args.no_insecure_fallback,
        font_family=args.font,
    )

    print("Generated files:")
    print(result["raw_csv"])
    print(result["yoy_csv"])
    print(result["index_plot"])
    print(result["yoy_plot"])
    print(result["signal_plot"])
    print()
    latest_period = result["latest_period"].strftime("%Y-%m")
    print(f"Latest period: {latest_period}")
    latest = result["latest"]
    for name, value in latest.items():
        print(f"  {name}: {value:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
