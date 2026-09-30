# Gate 永續合約 Avellaneda

這是新增的可設定策略腳本 `scripts/avellaneda_perpetual.py`，只支援
`gate_io_perpetual` 的 USDT 線性永續合約、ONEWAY 單向持倉。
使用標準 USDT 合約錢包；統一授信帳戶與 split-position 模式不受支援。
原本的現貨 `avellaneda_market_making` 不受影響。此版本使用固定 gamma
和固定 kappa，沒有啟用現貨版的自適應學習器。

## 預覽與啟動

使用包含本分支程式碼的 Hummingbot 執行環境；舊版官方 Docker image
沒有新增的核心模組或 Gate 平倉修正，單獨複製腳本不足以運行。
在專案根目錄執行：

```bash
cp examples/avellaneda_perpetual_gate.yml conf/scripts/conf_avellaneda_perpetual_gate.yml
```

在 Hummingbot CLI 連接自己的 Gate API，再啟動：

```text
connect gate_io_perpetual
start --script avellaneda_perpetual.py --conf conf_avellaneda_perpetual_gate.yml
status
```

也可用 `create --script-config avellaneda_perpetual` 產生預設設定。
預設 `dry_run: true` 只觀察行情並顯示候選報價，不模擬成交、績效或保證金，
也不下單、撤單、變更帳戶模式或槓桿。預覽仍使用 Gate 連接器，可能需要
API 讀取權限；不要將它當成交易所測試網或完整 paper trading。

確認預覽的最小下單量、價差、餘額與交易對正確後，停止腳本，將自己的
設定檔改成 `dry_run: false` 再重新啟動，才會開始真實交易。首次實盤要求
整個 USDT 合約帳戶沒有持倉或未成交委託，並等待交易所確認單向模式及
槓桿。使用專用帳戶，不同時手動交易、執行其他策略或進行入出金。
切換 Gate 持倉模式是帳戶層級操作。

## 100 USDT 的起始範例

| 設定 | 範例 | 意義 |
| --- | --- | --- |
| trading_pair | BTC-USDT | 起始候選；依即時交易規則取整 |
| leverage | 1 | 仍為永續合約，具有資金費與清算風險 |
| order_amount_quote | 10 USDT | 每筆最大名目金額，向下取整 |
| max_position_quote | 20 USDT | 絕對淨持倉名目上限 |
| capital_budget_quote | 30 USDT | 策略計算的保證金與掛單成本預算，不是交易所隔離資金 |
| available_balance_reserve | 5 USDT | 最低可用餘額緩衝，另留原始 taker 費用 |
| min_spread | 0.0004 | 買賣報價的完整價差下限 0.04% |
| max_session_loss_quote | 1 USDT | 本次執行的錢包＋中間價估值持倉損失停機門檻 |

幣價、最小合約張數與取整會改變實際下單金額。若 10 USDT 不足以買入
一張合約或不符合最小名目金額，該側不掛單，不會自行增加本金或槓桿。
預算不會改變 Gate 的全倉／逐倉設定，策略也不承諾最多只虧 1 USDT。

## 報價模型與費用

以中間價 `S`、有時間戳的 log-return 平方估計每秒變異數 `v`，定義
`q = 有符號持倉數量 × S / max_position_quote`，目標淨持倉為零。
此版本是使用相對價格單位的 Avellaneda–Stoikov 近似：

```text
r = S × exp(-q × gamma × v × horizon_seconds)
full_spread = gamma × v × horizon_seconds + 2/gamma × log(1 + gamma/kappa)
```

做多會降低保留價格，做空會提高保留價格；波動度增加會擴大價差。
`risk_factor` 是無量綱 gamma，`kappa` 是相對價格距離的成交強度衰減係數。
預設 kappa 為手動設定，沒有用成交資料校準；這些數值不能直接套用原本
現貨策略的 gamma/kappa。價差是完整價差比例，買價向下、賣價向上依 tick
取整並保持在最佳買賣價外側，使用 `LIMIT_MAKER` post-only 委託。
超過 `max_spread` 時暫停報價；行情中斷後會重新暖機。

設定中的費率全部使用小數比例：maker `0.0002` = 0.02%，taker
`0.0005` = 0.05%，反傭 `0.6` = 支付手續費的 60%。依此假設：

| 成交方式 | 估計淨費用，按兩腿名目金額近似相同 |
| --- | --- |
| 單腿 maker | 0.008% |
| 單腿 taker | 0.020% |
| maker + maker | 0.016% |
| maker + taker | 0.028% |

報價下限為 `max(min_spread, 2 × maker_fee × (1-rebate_rate) + adverse_selection_buffer)`。
緩衝是人工設定，不是逆向選擇損失的實證估計。保證金和委託預算按未反傭的
完整手續費計算；反傭尚未到帳不能拿來支付費用。反傭不保證正期望，價差
收入仍可能被價格趨勢、逆向選擇、資金費、滑價與緊急平倉吃掉。

Gate 目前連接器的 funding-payment 查詢是空實作，因此本版本沒有宣稱
能提供完整的資金費績效明細。定期刷新實際錢包餘額，已結算資金費會影響
損失停機；未結算資金費沒有被預測或計入報價。

## 持倉、撤單與停機

- 用合約有符號淨持倉計算庫存，不用 BTC 現貨餘額。
- 每側一筆委託；到期先撤單，所有舊單收到終止事件後才建立下一輪。
- 成交後撤去剩餘報價，等待成交數量與交易所持倉精確同步；不同步會停機。
- 減倉側最多掛目前持倉量，使用 `PositionAction.CLOSE`；此分支同時修正
  Gate 連接器，將 CLOSE 委託真正送成 `reduce_only: true`，不跨零反向開倉。
- 達到損失／持倉上限、過多下單失敗、撤單或同步逾時時停止報價。先等舊單
  終止並透過 REST 確認持倉，再以 reduce-only 市價嘗試平倉，最多三次。
- 行情或帳戶資料過期時先撤單；資料恢復才報價。沒有可信資料時不盲目補單。
- `stop` 預設取消自己的委託並嘗試平倉，最多等待 30 秒；
  `flatten_on_stop: false` 則只取消委託並留下持倉。
- 撤單未確認、未解決的市價單、API 中斷或平倉失敗可能使持倉繼續存在。
  超時會寫出錯誤，需在 Gate 確認掛單與持倉。停機不等同成交或損失保證。
- 強制結束程序／主機斷電不會執行 `on_stop`；REST 和 websocket 仍有延遲。

此策略的報價與風控並非交易所原生原子操作。100 USDT 範例是功能測試
起點；尚未透過真實 Gate 成交或歷史成交回測證實收益。

## 驗證

核心與腳本 adapter 的測試只需 Python、pydantic >= 2、PyYAML，
可不編譯 Cython 執行：

```bash
python -m unittest discover -s test/hummingbot/strategy_v2/avellaneda_perpetual -v
```

adapter 測試以替身 connector 和 ScriptStrategyBase 邊界執行實際腳本，
涵蓋預覽不下單、模式／槓桿確認、post-only 和 CLOSE 路由及停機。
它們不取代完整 Hummingbot 安裝或 Gate 測試環境的驗證。完整連接器
回歸測試位於 `test/hummingbot/connector/derivative/gate_io_perpetual/`。

帳戶餘額與委託欄位依 Gate 的官方 SDK 定義：
[FuturesAccount](https://github.com/gate/gateapi-python/blob/master/docs/FuturesAccount.md)、
[FuturesOrder](https://github.com/gate/gateapi-python/blob/master/docs/FuturesOrder.md)。
