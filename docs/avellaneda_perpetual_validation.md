# 合約版 Avellaneda 執行驗證紀錄

驗證日期：2026-09-30。設定為 100 USDT 假設本金、1 倍槓桿、單筆上限
10 USDT、絕對淨持倉上限 40 USDT、計算預算 50 USDT、費用緩衝 5 USDT。
Maker 0.02%、taker 0.05%、反傭 60% 是使用者提供的假設，沒有登入 Gate
確認帳戶費率或反傭實際入帳。

## 可重現的公開行情校準

本分支保存 [公開行情 CSV](../examples/avellaneda_perpetual_validation/gate_btc_capture_20260930.csv)
和 [校準與回放 JSON](../examples/avellaneda_perpetual_validation/report_20260930.json)。
來源是 Gate 公開 USDT 永續合約規則及 top-of-book REST，不需要 API 金鑰。
資料共 1,224 筆，UTC 05:06:39–05:32:13（台北 13:06:39–13:32:13），
包含一段 306.37 秒的收集間隔。首次校準只用間隔之前連續有效的 902.01 秒，
按整秒桶去重後 881 筆，不用之後的行情反向修正參考值。

| 校準項目 | 結果 |
| --- | --- |
| 交易對 | BTC-USDT |
| 參考中間價 | 83,238.65 USDT |
| 價格 tick | 0.1 USDT |
| 張數／最小量 | 0.0001 BTC，至少一張 |
| 最小名目金額 | 1 USDT，依本分支 Gate 連接器規則 |
| 真正第一張庫存 | 0.0001 BTC，參考名目 8.323865 USDT |
| q_ref | 0.208096625，約占 40U 上限 20.8% |
| v_ref | 1.26893306477 × 10⁻⁹ / 秒 |
| 30 秒參考變異數 V_ref | 3.80679919431 × 10⁻⁸ |
| 30 秒參考 sigma | 0.019511% |
| gamma_base | 303.30563849 |
| 自適應 gamma 範圍 | 約 151.65–909.92 |
| 第一張多單報價差 | bid −2 ticks、ask −2 ticks |
| 第一張空單報價差 | bid +2 ticks、ask +2 ticks |

校準目標是參考條件下保留價格偏移兩個 tick；之後波動度改變，當時偏移
不必仍是兩個 tick。此份報告是當次行情的敏感度示例，不是收益最適參數。
程式預設校準 24 小時失效，交易規格或影響報價的設定不合也會失效。

重現命令（輸出路徑須尚未存在）：

```bash
python tools/avellaneda_perpetual_calibrate.py \
  --capture examples/avellaneda_perpetual_validation/gate_btc_capture_20260930.csv \
  --report logs/avellaneda_reproduction.json \
  --quotes logs/avellaneda_reproduction_quotes.csv \
  --write-preview-config conf/scripts/conf_avellaneda_perpetual_reproduction.yml
```

## 相同行情的報價敏感度比較

只比較校準完成後的行情，間隔後重新暖機；每組 287 筆有效觀測。
每組使用相同交易規則、單量、持倉上限、費率和價差地板，並明示庫存是假設。
數值是**保留價格的平均絕對偏移 tick**，不是成交收益，也不是最終委託價差：

| 假設庫存 | 固定 gamma 1 | 校準固定 gamma | 自適應 gamma |
| --- | ---: | ---: | ---: |
| 零庫存 | 0 | 0 | 0 |
| 多一張 | 0.00993 | 3.01193 | 3.45921 |
| 空一張 | 0.00993 | 3.01194 | 3.45922 |
| 多三張 | 0.02979 | 9.03575 | 13.57461 |

本次所有比較都由 0.04% 最低完整價差生效。gamma 已影響中心偏移，但沒有
改掉這個價差下限。回放沒有建立排隊、觸價成交、部分成交、延遲或資金費模型，
所以不產生收益率或正期望結論。餘額欄位全是 100U 的假設值。

## 程式與編譯執行

| 檢查 | 結果 |
| --- | --- |
| 核心／adapter、校準、限幅、40U、markout、模擬及 API 保護測試 | 108 項通過 |
| 完整 Gate 永續連接器測試目錄 | 87 項通過 |
| 真正 Hummingbot Clock／ScriptStrategyBase／Gate 整合測試 | 6 項通過 |
| Cython 模組 | 58 個編譯完成 |
| 正式 TradingCore 策略／設定載入器 | 校準後 adaptive 預覽 YAML 載入成功，確認 40U／50U 及 gamma 校準資料 |
| flake8、isort、Python 語法與 git diff 空白檢查 | 通過 |

核心 suite 保留原本明確 20U／30U 的測試情境，另外測試新的 40U／50U 設定，
不藉改動舊斷言掩蓋回歸。新增測試包含 15 分鐘資料門檻、tick 可見性、零波動、
資金／最小張數不足、規格失配、校準過期、整秒時間抖動、5 秒頻率、20% 平滑、
±10% 限幅、上限、資料不可信時凍結，以及行情缺口暖機時撤掉舊單。

整合 suite 使用真正的 compiled Clock、OrderBook、ScriptStrategyBase、Gate
連接器和訂單／成交事件追蹤器，替換交易 transport 和網路就緒狀態。驗證了
1,000 秒時鐘回放的預覽校準不呼叫任何 API、更改帳戶或建立委託；實際策略
委託路由送出 post-only／reduce-only 欄位，成交重複事件去重、撤單終止事件
回到引擎，以及緊急市場平倉使用 `reduce_only: true`、`tif: ioc`、`price: 0`。
這些是本機帶 transport 替身的執行驗證，不是 Gate 已接受真實委託的證據。

驗證環境 Python 3.12、Cython 3.0.12、gcc/g++；XRPL 依專案環境使用 4.1.0。
HTTP 測試用 aiohttp 3.9.5、aioresponses 0.7.8，避免新版本 aiohttp 的
`stream_writer` 建構參數與舊測試替身衝突。原有 Gate suite 會留下未匹配的
mock 背景連線警告，但全部 87 項測試斷言通過；新增六項整合測試不啟動真實網路。

另補齊隔離環境的 UI、MQTT 與技術指標依賴，透過正式
`TradingCore.load_script_class` 識別 `AvellanedaPerpetual`，載入工具匯出的
巢狀校準 YAML，確認 `adaptive`、40U 上限、50U 預算及預覽旗標。
此檢查使用本機 pandas-ta 0.4.71b0、Numba 0.62.1、protobuf 6.32.1、
prompt_toolkit 3.0.51、paho-mqtt 2.1.0，沒有改動專案的依賴設定。
沒有啟動真實帳戶、MQTT 連線或實盤時鐘。

## 尚未完成的實盤驗證

沒有使用你的 Gate API、沒有更改真實帳戶模式或槓桿、沒有下單或合併 master。
實際費率、反傭入帳、資金費歸因、排隊成交、API 延遲和長期盈虧仍待帳戶驗證。
`actual_fee_quote` 保存成交事件回報費用，估計反傭與已入帳金額分欄；無法獨立
確認的入帳反傭為 null。錢包權益已包含實際現金流，不能再把估計反傭加進停損權益。
原連接器 funding-payment 查詢尚為空實作，因此記錄明示歸因不完整。
PnL 線上學習仍列為後續功能。預設及匯出設定均維持 `dry_run: true`。

## API 保護追加驗證

新增實際 RESTAssistant pre/post-processor 路徑測試，以 transport 替身回覆 429，
驗證 reset header 保留、送出失敗計數、冷卻時完全不呼叫 transport、重置後恢復、
撤單 429 保留追蹤以及真實 FAILED 事件不消耗緊急減倉次數。
純核心測試涵蓋每日與每秒保留容量、秒／毫秒 reset、伺服器時差、退避、
剩餘 0、失敗請求、日誌重啟與程序鎖、日誌損壞及磁碟寫入失敗、過期排隊報價、
未變報價保留、部分成交、撤單冷卻與每日額度停止平倉。

介面已檢查 429 注入、冷卻計數、低每日額度停機、JSON／CSV 匯出與離線 HTML，
桌面及 768／390／320px 沒有整頁水平溢出。兩份範例互動 HTML 已依新生命週期重算；
舊策略的模擬成交和損益不能直接當成新版本的結果。
實盤 API、真實反傭或盈利未驗證，預設仍為 dry_run。
