單一事實來源（SSOT）：景氣指標資料模組
===========================================

目的（Scope）
------------
- 以主計總處 NSTAT 資料庫提供之 SDMX 服務，取得「景氣指標」資料集（代碼 `a120101010`），生成領先 / 同時指標及景氣對策信號的標準化指標表與圖表。
- 作為總體經濟觀測的基礎資料模組，供後續分析與儀表板視覺化之用。

資料來源（Input Schema）
------------------------
- 來源：<https://nstatdb.dgbas.gov.tw/dgbasall/webMain.aspx>（SDMX API）
- 路徑模式：`sdmx/{dataset}/{series...frequency.}`，本模組預設 `dataset=a120101010`、`series=1+2+3+4+5`、`frequency=M`。
- 時間範圍：預設 `startTime=2020-M1`、`endTime=2025-M8`；可透過 CLI 參數調整。
- 指標說明（series_id → series_name）：
  1. 景氣領先指標綜合指數(點)
  2. 景氣領先指標不含趨勢指數(點)
  3. 景氣同時指標綜合指數(點)
  4. 景氣同時指標不含趨勢指數(點)
  5. 景氣對策信號(分)

清理與轉換規則（Data Rules）
---------------------------
- 透過 SDMX 回傳的 JSON 結構解析 series 與 observation 維度，轉為長表格（長格式）後，再 pivot 為寬表格。
- 期間欄位 `YYYY-Mm` 解析為西元年月，日期統一為 `YYYY-MM-01`（每月第一天）。
- 數值欄位轉為浮點數；無法解析時以 `NaN` 表示。
- 為避免 SSL 憑證問題，程式預設嘗試驗證；若驗證失敗且允許 fallback，會以警告方式改用不驗證連線（可透過 `--no-insecure-fallback` 阻止）。

產出（Outputs）
---------------
- `out/macro_business_cycle.csv`：景氣指標寬表格（各指標列為欄，索引用月）。
- `out/macro_business_cycle_yoy.csv`：各指標年增率（12 個月滯後百分比變化）。
- `out/macro_business_cycle_index.png`：領先與同時指標（含 / 不含趨勢）折線圖。
- `out/macro_business_cycle_yoy.png`：領先 / 同時指標年增率折線圖（含 0% 基準線）。
- `out/macro_business_cycle_signal.png`：景氣對策信號步階圖。

指標計算（Metrics）
-------------------
- `business_cycle[index]`：原始指標值。
- `business_cycle[index]_yoy`：`pct_change(12) * 100` 計算的年增率。
- 圖表僅針對前四項指標繪製 YoY；景氣對策信號保留原值。

執行方式（Automation）
----------------------
```
python scripts/build_macro_business_cycle.py \
  --start 2020-M1 \
  --end 2025-M8 \
  --out out_macro
```
- `--no-verify-ssl`：直接停用 SSL 驗證。
- `--no-insecure-fallback`：在驗證失敗時停止，而非使用非驗證模式。
- `--font`：指定 Matplotlib 字型（預設為 `Microsoft JhengHei`）。

維護與修改（Change Guide）
-------------------------
- 若要新增指標或改用其他 SDMX 資料集，調整 `DEFAULT_SERIES_IDS` 或 `DATASET_ID` 並更新本文檔。
- 如需改寫圖表樣式，可調整 `plot_*` 函式；增加輸出需同步更新「產出」章節。
- SSL 行為可依部署環境調整（例如提供企業內部 CA 憑證）。
- 範例資料更新或測試流程需同步於專案 README / CI 進行說明。
