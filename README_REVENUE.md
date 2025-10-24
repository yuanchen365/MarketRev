目的
- 根據 MOPS 月營收資料，彙總整體上市櫃的月營收，計算年增率（YoY）與 3 個月平滑線，並輸出圖表；同時提供指數化（2019-12=100，若不存在則以首期作為基期）。

輸入
- `mops_revenue_107_114.csv`（或相同結構的 CSV）。常見欄位：
  - 發布日期（例：`114/08/23`）、資料年月（例：`107/1`）或 `年`、`月`
  - 公司代號、公司名稱、產業別
  - 當月營收-當月金額（或同義欄位名）

輸出
- `out/market_yoy.png`：整體市場月營收年增率（含 3M 平滑）
- `out/market_index.png`：整體市場月營收指數（基期 2019-12=100，若無則以資料首期）
- `out/industry_yoy_top12.png`（若有產業別且未停用）：前 12 大產業 YoY 小圖
- `out/market_ytd_yoy.png`：整體市場「累計（年初至今，YTD）」營收年增率，含 3M/6M 平均
- `out/market_revenue_ma.png`：整體市場 3M/6M 平均營收 與 當月營收

使用方式（Windows）
1) 建立虛擬環境並安裝套件
```
py -3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) 產生圖表
```
py -3 scripts\build_market_growth.py mops_revenue_107_114.csv --out out
```

3) 若不需要分產業圖
```
py -3 scripts\build_market_growth.py mops_revenue_107_114.csv --out out --no-industry
```

4) 以串流介面輸出指標（純文字摘要）
```
py -3 scripts\build_market_growth.py mops_revenue_107_114.csv --out out --stream
```
說明：`--stream` 會在終端機輸出指標摘要（中文），方便串流 UI 呈現；同時仍會輸出圖檔。

5) 強制中文字型設定（避免字形缺漏警告）
```
# 直接指定常見字型（系統需已安裝）
py -3 scripts\build_market_growth.py mops_revenue_107_114.csv --out out --font "Microsoft JhengHei"

# 或提供字型檔路徑（.ttf/.otf/.ttc）
py -3 scripts\build_market_growth.py mops_revenue_107_114.csv --out out --font-file "C:\\Windows\\Fonts\\msjh.ttc"
```
備註：若未指定，程式會自動嘗試挑選可顯示中文的字型；提供 `--font-file` 可確保強制使用該字型。

說明
- 程式會自動偵測常見中文欄位名稱；若找不到「當月營收-當月金額」或年月欄位，會提示錯誤。
- 資料重複（同公司同月份多列）時，若含發布日期，會保留「最新發布」的一列。
- ROC 年月（如 107/1）會自動轉成西元年月。
- 新增指標與中文化：
  - 累計營收（YTD）：每年內對當月為止的累加值。
  - 累計營收年增率（YTD YoY）：`YTD_total / 去年同期YTD_total - 1`。
  - 3M/6M 累計營收年增率平均：對 YTD YoY 做 3 個月與 6 個月移動平均。
  - 3M/6M 平均營收：對整體當月營收做 3 個月與 6 個月移動平均。
  - 所有圖表標題、圖例、座標軸改為中文。
  - 不使用月增年化指標。
