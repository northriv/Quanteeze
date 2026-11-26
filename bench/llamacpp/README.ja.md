# llama.cpp と同じ基準のベンチマーク（Quanteeze）

**[English](README.md)** | **日本語**

llama.cpp の `llama-bench` と `llama-batched-bench` の測定定義（テスト名と順序、時間の取り方、統計、表の列）を、配信中のサーバーに向けて Python から測れるように移植したベンチマークツールです。

| llama.cpp のツール | ここ | 測るもの |
|---|---|---|
| `llama-bench` | `llama_bench.py` | `pp512`（プロンプト 512 トークン）、`tg128`（128 トークン生成）、`pp512+tg128`、`@ d4096`（4,096 トークン文脈のあと）。繰り返し 5 回の t/s の平均 ± 標準偏差 |
| `llama-batched-bench` | `batched_bench.py` | B 本同時実行時の `PP`・`TG`・`B`・`N_KV`・`T_PP`・`S_PP`・`T_TG`・`S_TG`・`T`・`S` |

## 出典とライセンス

- テスト定義・統計・テーブル形式は [llama.cpp](https://github.com/ggml-org/llama.cpp)（`master` `8d60873`）の `tools/llama-bench/llama-bench.cpp` と `tools/batched-bench/batched-bench.cpp` に準拠して適応しています（MIT License、詳細は [LICENSE](LICENSE) を参照）。
- llama.cpp 本体はライブラリ直結のため、本ツールでは HTTP 経由で配信中のサーバーに対して測定するよう適応しています。

## 使い方

サーバーは測定用に別ポートで起動してください。

- `--log-level INFO` を付けてログをファイルへ出してください（`--feed forced` の時間計測に使用します）。
- llama.cpp の `llama-bench` / `llama-batched-bench` には投機的デコードがないため、比較測定では通常 `QUANTEEZE_MTP=off` で測定します。

```bash
env QUANTEEZE_SETTINGS=off QUANTEEZE_MTP=off APC_DISK_ENABLED=0 \
  uv run quanteeze --model <チェックポイント> --host 127.0.0.1 --port 8766 \
  --enable-thinking --log-level INFO > serve.log 2>&1 &
```

### `llama_bench.py`

```bash
uv run python bench/llamacpp/llama_bench.py --url 127.0.0.1:8766 --checkpoint <チェックポイント> --log serve.log \
    -p 512 -n 128 -pg 512,128 -d 0,4096,16384 -r 5
```

- **`--feed forced`（既定、Quanteeze）**：
  - 管理トークン（`QUANTEEZE_CONTROL_TOKEN` または `~/.quanteeze/control-token`）付きの `quanteeze_forced` フィールドを使い、ちょうど N 個の乱数トークン ID（BOS があれば先頭は BOS）を入力し、出力も乱数トークン ID を 1 つずつ強制します（サンプリングなし）。
  - 時間はサーバーログの `Prefill completed` / `Decode completed` から取得します（`tg` はプレフィルが出す最初のトークンを除く `generated - 1` ステップ）。
  - **深さ `d`**：d 個の乱数トークンの直後にテスト用プロンプトを続けます。`Prefill progress` で `d` トークン目に区切りがあればそこからの差分時間を使い（mlx-vlm は 2,048 トークン単位で進捗を出すため 2,048 の倍数を推奨）、APC が `d` トークンをヒットしていればプレフィル全体時間、どちらでもない場合は `d` 個単独の事前計測時間を引いた概算値（`~` 印）になります。
- **`--feed text`（任意の OpenAI 互換サーバー）**：
  - 乱数単語列と物語プロンプトを `temperature=0` で送信します。`--log` を省略した場合はクライアント側の TTFT と完了時刻で計測します（`c` 印）。
- **注意点**：
  - `--feed forced` は logits processor を通るため、融合済み貪欲ステップではなく通常のサンプリング前ロジット経路を通ります。また乱数トークンは通常文章と異なるエキスパート選択になります。

### `batched_bench.py`

```bash
uv run python bench/llamacpp/batched_bench.py --url 127.0.0.1:8766 --checkpoint <チェックポイント> --log serve.log \
    -npp 128,512 -ntg 128 -npl 1,2,4,8
```

- B 本のリクエストを B スレッドから同時に送信します（プロンプト非共有）。
- `T` は最初のリクエストの `Generation queued` から最後の `Decode completed` まで、`T_PP` は各リクエストのプレフィル区間の和集合（重なりを統合した時間）、`T_TG = T - T_PP` として llama-batched-bench と同じ列（`S_PP = B*PP/T_PP`、`S_TG = B*TG/T_TG`、`S = B*(PP+TG)/T`）を算出します。
- 複数リクエストの並列処理にはサーバー設定（`QUANTEEZE_REQUEST_THREADS` または mlx-vlm の継続バッチング）が必要です。

## 制限事項

- llama.cpp 本体の `-pps`（プロンプト共有）、`-b`/`-ub`（バッチサイズ）、`-ngl` などライブラリ固有オプションは対象外です。
