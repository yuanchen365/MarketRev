單一事實來源（SSOT）：上市櫃整體月營收指標

目的（Scope）
- 以公開資訊觀測站（MOPS）公司層級月營收資料，彙總為「整體上市櫃」的月度序列，產出一致定義的指標與圖表，作為後續分析與報表的單一事實來源（SSOT）。

資料來源（Input Schema）
- 檔案：mops_revenue_107_114.csv（或相同結構）
- 欄位需求（自動偵測）：
  - 時間：年/月 或 資料年月（ROC 年月，如 107/1）
  - 公司識別：公司代號（選配）、公司名稱（選配）、產業別（選配）
  - 營收金額：當月營收-當月金額（或同義欄位）
  - 發布日期（選配）：若同公司同月多列，保留最新發布

清理規則（Data Rules）
- 營收轉數值：移除千分位、空值轉遺漏。
- 年月轉換：ROC 轉西元，月第一天為代表期（YYYY-MM-01）。
- 去重：以公司代號＋月份為索引，同公司同月保留最新發布。
- 彙總：以月份為索引，對全部公司總和為 	otal_revenue。

計算定義（Metrics Definitions）
1) 年增率（YoY）
   - yoy_pct = pct_change(total_revenue, 12) * 100
   - 3M 平滑：yoy_pct_smooth_3m = rolling_mean(yoy_pct, 3)
2) 指數（2019-12=100）
   - 基期為 2019-12（若不存在則使用首期），index_2019_12_100 = total_revenue / base * 100
3) 平均營收（Moving Average）
   - vg_revenue_3m = rolling_mean(total_revenue, 3)
   - vg_revenue_6m = rolling_mean(total_revenue, 6)
4) 累計（年初至今，YTD）與年增率
   - ytd_total = cumsum_by_calendar_year(total_revenue)
   - ytd_yoy_pct = pct_change(ytd_total, 12) * 100
   - ytd_yoy_pct_avg_3m = rolling_mean(ytd_yoy_pct, 3)
   - ytd_yoy_pct_avg_6m = rolling_mean(ytd_yoy_pct, 6)

衍生資料集（Derived Series）
- 排除金融保險：比對 industry 欄位含「金融」或「保險」字樣的公司，移除後重新彙總一組 	otal_revenue，並套用與整體市場相同的指標流程。輸出檔名統一加上 _ex_fin 後綴。

不範圍內（Out of Scope）
- 月增率（MoM-Annualized）等短期波動調整，不列入 SSOT 指標。

輸出（Outputs）
- 圖表：
  - out/market_yoy.png：整體營收年增率含 3M 平滑
  - out/market_index.png：整體營收指數（2019-12=100）
  - out/market_ytd_yoy.png：YTD 年增率含 3M/6M 平滑
  - out/market_revenue_ma.png：3M/6M 平均營收 vs 當月營收
  - out/industry_yoy_top12.png：前 12 大產業年增率圖
  - out/market_yoy_ex_fin.png：排除金融保險後的年增率（含 3M 平滑）
  - out/market_index_ex_fin.png：排除金融保險後的營收指數
  - out/market_ytd_yoy_ex_fin.png：排除金融保險後的 YTD 年增率（含平滑）
  - out/market_revenue_ma_ex_fin.png：排除金融保險後的移動平均 vs 當月營收
  - out/industry_yoy_top12_ex_fin.png：排除金融保險後的前 12 大產業年增率圖

檔案欄位（DataFrame Columns）
- 市場層級 DataFrame：
  - 	otal_revenue、yoy_pct、yoy_pct_smooth_3m
  - index_2019_12_100
  - vg_revenue_3m、vg_revenue_6m
  - ytd_total、ytd_yoy_pct、ytd_yoy_pct_avg_3m、ytd_yoy_pct_avg_6m

範例驗證（Validation）
- 使用 scripts/verify_sample_metrics.py 建立 2023-01 至 2024-03 的合成資料（科技、零售兩個產業，各佔 60%/40%），對 ggregate_market_series() 進行實算驗證。
- 排除金融保險測試：同一批資料另新增「金融及保險業」樣本，驗證 _ex_fin 流程會排除該類別並生成相同欄位。
- 市場層級指標（2024-01~2024-03）：

  | 月份 | 總營收 | 年增率 | 3M 平滑 YoY | 3M 平均營收 | 6M 平均營收 | YTD 總額 | YTD 年增率 | YTD YoY 3M 均值 | 指數（基期 2023-01=100） |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | 2024-01 | 900.00 | 8.00% | 8.00% | 990.00 | 1,002.50 | 900.00 | 8.00% | 8.00% | 108.00 |
  | 2024-02 | 1,000.00 | 9.00% | 8.50% | 980.00 | 1,000.83 | 1,900.00 | 8.52% | 8.26% | 120.00 |
  | 2024-03 | 1,050.00 | 10.53% | 9.18% | 983.33 | 1,006.67 | 2,950.00 | 9.23% | 8.58% | 126.00 |
- 產業層級 YoY（2024-01~2024-03）：

  | 月份 | 科技 | 零售 |
  | --- | --- | --- |
  | 2024-01 | 8.00% | 8.00% |
  | 2024-02 | 9.00% | 9.00% |
  | 2024-03 | 10.53% | 10.53% |
- 指數基期回傳為 2023-01，符合未偵測到 2019-12 時改以首筆為基期的設計。

維護與修改（Change Guide）
- 新增指標：在 ggregate_market_series() 中計算並命名欄位，於相應 plot_* 函式加入視覺化。
- 調整基期：在 ggregate_market_series() 內修改 ase_date 邏輯（目前預設 2019-12，否則首期）。
- 產業圖選項：在 plot_industry_facets() 調整 	op_n 或改為全產業 Facet。
- 排除金融保險：如需調整規則，更新 scripts.build_market_growth._filter_out_finance() 與 ui/app.py 中的 split_ex_finance()。
- 字型：如需指定字型，於 plot_* 函式內 
cParams['font.family'] 調整。

版本政策（Versioning）
- 指標名稱與公式為 SSOT，修改需同步更新本文檔與 README。
