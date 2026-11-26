# Quanteeze

[![License: GPL-2.0 OR PolyForm-Shield](https://img.shields.io/badge/License-GPL--2.0_OR_PolyForm--Shield-blue.svg)](#ライセンスlicense)
[![GitHub](https://img.shields.io/badge/GitHub-northriv%2FQuanteeze-181717?logo=github)](https://github.com/northriv/Quanteeze)
[![Python 3.12+ | C++20](https://img.shields.io/badge/Python_3.12%2B_%7C_C%2B%2B-20-3776AB)]()
[![Platform: macOS (Metal) | Linux (CUDA)](https://img.shields.io/badge/platform-macOS_(Metal)_%7C_Linux_(CUDA)-lightgrey)]()
[![kamestm: arXiv 2608.12024](https://img.shields.io/badge/kamestm-arXiv%3A2608.12024-b31b1b.svg)](https://arxiv.org/abs/2608.12024)

**[English](README.md)** | **日本語**

Apple Silicon Mac や NVIDIA GPU 搭載の Linux マシンで、巨大な MoE（Mixture-of-Experts）や VLM/LLM モデルをそのまま、または手軽に省メモリ化して配信できる MLX 推論サーバーです。均一な低ビット量子化や、形式検証済みのロックフリー C++20 STM（[`kamestm`](https://github.com/northriv/KAME/tree/master/kamestm)）および 4 階層プールアロケータ（[`kamepoolalloc`](https://github.com/northriv/kamepoolalloc)）による自動 SSD / ホスト RAM ストリーミング、内蔵ブラウザ UI、さらに **OpenAI 互換 API（Responses API / Codex CLI 対応）** と **Anthropic Messages API（Claude Code 対応）** を標準で備えています。

[クイックスタート（macOS）](#はじめて試す方へクイックスタート) · [アプリ接続（Claude Code / Codex / KAME）](#7-claude-codecodex-cli-や他のアプリからつなぐ) · [Linux（NVIDIA GPU）](#linuxnvidia-gpuのクイックスタート) · [実機ベンチマーク](#実機での動作速度例--experts-gguf) · [主な特徴・仕組み](#主な特徴仕組み) · [対応モデル](#対応モデルアーキテクチャ) · [環境変数リファレンス](#主な環境変数設定リファレンス) · [引用（Citing）](#引用citing) · [ライセンス](#ライセンスlicense)

---

## はじめて試す方へ（クイックスタート）

ここでは、インストールからブラウザ UI での確認、Claude Code・Codex CLI・KAME などのアプリからの接続までを順番に説明します。

### 1. 必要なもの

- **Apple Silicon の Mac**（大きな MoE モデルを快適に動かすならメモリ 32〜48 GB 以上推奨。均一な 2〜3 ビット量子化や自動 SSD ストリーミングにより、メモリ容量を超えるモデルも動作します）、または **NVIDIA GPU 搭載の Linux 機**（手順は [Linux（NVIDIA GPU）のクイックスタート](#linuxnvidia-gpuのクイックスタート)）
- モデルを置く **SSD の空き容量**（モデルによって数十〜数百 GB。Thunderbolt / USB4 接続の高速な外付け NVMe SSD でも使えます）
- **[uv](https://docs.astral.sh/uv/)**（Python 3.12 以上の環境と依存パッケージを自動で用意します）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **補足（任意・C++20 コンパイラ）**：C++20 コンパイラ（Mac の Xcode Command Line Tools の `clang++`、または Linux の `g++` 10 以上）が入っている環境では、初回起動時に同梱のロックフリー・ロックレス 4 階層アロケータ [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc)（`QUANTEEZE_ALLOCATOR=auto`）と C++20 ネイティブ拡張群（`QUANTEEZE_NATIVE=on`：[`kamestm`](https://github.com/northriv/KAME/tree/master/kamestm) ベースの STM オンデマンドエンジン `stm_core`、ロックフリー `pread` リーダー `native_reads`、および Linux x86-64 AVX2/AVX-512 環境の CPU エキスパート計算 `cpu_experts`）を自動でビルド・有効化します。コンパイラが無い場合は標準の `malloc` と純 Python 実装へ静かにフォールバックするため、事前インストールは必須ではありません。

macOS の Unified Memory 環境では、Metal が 1 プロセスに推奨する作業領域（`max_recommended_working_set_size`）から `QUANTEEZE_MEMORY_RESERVE`（既定 `0.3`）を差し引いた残りにエキスパートが収まるかどうかで、「全重みをメモリに保持」と「SSD からオンデマンド読込」を自動で切り替えます。

| Mac の構成例 | 搭載メモリ | Metal 推奨作業領域の目安 |
|---|---|---|
| Mac mini / MacBook Pro (M5 Pro) | 48 GB | 約 40.2 GB |
| Mac Studio (M5 Ultra) | 96 GB | 約 83.5 GB |
| MacBook Pro (M5 Max) | 128 GB | 約 115.4 GB |

手元の Mac の推奨作業領域は次のコマンドで確認できます：

```bash
.venv/bin/python -c "import mlx.core as mx; print(round(mx.device_info()['max_recommended_working_set_size'] / 1e9, 1), 'GB')"
```

### 2. 取得とインストール（Mac）

置き場所は `~/Documents` や `~/Desktop` の外（たとえば `~/Quanteeze`）をおすすめします。macOS では `~/Documents` などが OS の保護対象フォルダになっており、Claude Code などのエージェントツールから使う際にアクセス許可を求められたり止められたりすることがあるためです。

```bash
git clone https://github.com/northriv/Quanteeze.git ~/Quanteeze
cd ~/Quanteeze
uv sync
```

`pip` を使う場合は Python 3.12 以上の仮想環境で `pip install -e .` を実行します。なお `pyproject.toml` ではカーネルのビット一致検証と C++ 拡張 ABI の安定性のため、`mlx==0.32.4`・`mlx-lm==0.32.2`・`mlx-vlm==0.7.9`・`nanobind==3.1.0` を厳密に固定しています。Linux（NVIDIA GPU）の手順は少し異なるため、[Linux（NVIDIA GPU）のクイックスタート](#linuxnvidia-gpuのクイックスタート) を見てください。

### 3. モデル：LM Studio や Hugging Face のフォルダがそのまま使えます

- **LM Studio でダウンロードした MLX 形式のモデル**（`~/.lmstudio/models/<組織>/<名前>`）は、コピー不要でそのまま Web UI の一覧に出ます（ベースモデルは MLX の safetensors 形式を使用します。配布済みの `UD-IQ*` 等の GGUF も併せて持っている場合は、`--experts-gguf` でエキスパート重みだけ GGUF から直接配信することもできます）。
- **Hugging Face のキャッシュ**（`~/.cache/huggingface/hub`）にあるモデルも自動で一覧に出ます。
- **外付け SSD** などにモデルフォルダを置いている場合は、起動時に `--model <その中にあるモデルフォルダ1つ>` を付けて起動してください。同じ親フォルダ配下にある他のモデルも一覧に出ます。
- 手元にモデルがまだない場合は、起動後に Web UI から Hugging Face のリポジトリ名（`org/model-name`）を入れてサイズを確認したうえでダウンロードできます。

### 4. 起動

```bash
cd ~/Quanteeze
uv run quanteeze
```

既定では `127.0.0.1:8765` で待ち受けます。特定のモデルを最初から読み込んで起動する場合は `--model <モデルのフォルダ>` を、思考モード（reasoning）を使うエージェントにつなぐ場合は `--enable-thinking` を付けて起動してください。停止するときは `Ctrl+C` です。

### 5. Web UI を開く（トークン付き URL）

起動すると、ターミナルに次のような 1 行が表示されます。

```text
quanteeze: control routes ask for a token; Browser UI at http://127.0.0.1:8765/quanteeze/#token=..., all settings at http://127.0.0.1:8765/quanteeze/settings#token=...
```

- 最初の `Browser UI at` の URL（`/quanteeze/#token=...`）を、**末尾の `#token=...` まで含めて**ブラウザで開いてください。トークンが欠けると画面は開いても操作が拒否されます。
- トークンはサーバーを再起動しても変わりません（`~/.quanteeze/control-token` に保存されます）。一度開いてブックマークしておくと、次回からはブックマークからそのまま開けます。
- URL を見失ったときは、`http://127.0.0.1:8765/quanteeze/` を開き、**トークン**（Token）欄に次のコマンドの出力を貼り付けて **使う**（Use）を押してください。

```bash
cat ~/.quanteeze/control-token
```

### 6. Web UI での使い方

Web UI は上から順番に使っていきます。ブラウザの言語設定が日本語なら自動で日本語表示になります（右上の **English / 日本語** で切り替えでき、詳細設定は **詳細設定 →**（Every setting →、`/quanteeze/settings`）から変更できます）。

1. **モデルと量子化精度**（Model & expert precision）
   このマシンで見つかったローカルモデルの一覧から使いたいモデルを選びます。まだ手元にないモデルを使う場合は、`org/model-name` を入力して **サイズ確認**（Check size）を押し、表示された合計ファイル数と容量（GB）を確認してから **Download ... GB** で取得できます。
   - **量子化されていない元の重み（`f32` / `bf16` / `f16`）の MLX 変換**：未量子化の `safetensors` チェックポイントには **`not quantized`** バッジが付きます。モデルを読み込んでいない状態であれば、**4 ビットに変換…** / **8 ビットに変換…** から使用ツール（`mlx-lm` または `mlx-vlm`）、保存先（LM Studio フォルダの隣または `~/.quanteeze/models`）、予想書き込みサイズ・空き容量を確認したうえで、リポジトリ付属コードを実行しない隔離プロセスで変換できます。
2. **ルーティング・エキスパートの精度**（Routed expert precision）
   MoE モデルを選ぶと、元の精度や各量子化形式（**4・3・2・1 ビット（最小二乗）**、**MXFP4 / Q3_K / Q2_K / IQ2_XS**）ごとに、エキスパートが使う容量と **メモリに収まる**（fits in memory）/ **SSD からオンデマンド読込**（streams from SSD）が表示されます。
   - **迷ったときの選び方**：まずは **メモリに収まる**（fits in memory）と表示される選択肢の中で一番ビット数の高いものを選ぶのがおすすめです（元の精度のままでも収まる場合は **元の精度** でそのまま高速に動きます）。メモリに入りきらないモデルを試したい場合は **SSD からオンデマンド読込** でもそのまま動きます。
   - 選んだら **このモデルを配信**（Serve this model）を押します。読み込みの進捗や現在のメモリ・発熱・速度は、画面一番上の **いま配信中**（Serving now）にリアルタイムで表示されます（C++20 [`kamestm`](https://github.com/northriv/KAME/tree/master/kamestm) エンジンでオンデマンド配信している場合は **`C++ STM`** のバッジも表示されます）。
   - **モデルの解放と再読み込み**（Unload / Load it again）：**いま配信中** の **モデルを解放**（Unload）を押すと、実行中のリクエスト完了後にモデルを解放してメモリを返します。再度 **もう一度読み込む**（Load it again）または別のモデルを読み込むまで生成リクエストは 503 で拒否されるため、接続中のクライアントにメモリを再確保させません。
3. **質問してみる**（Try a prompt）
   ブラウザからそのままモデルに質問を送って、応答・生成速度（`tok/s`）・**思考の内容**（Its thinking）を確認できます（`Cmd+Enter` / `Ctrl+Enter` でも送信できます）。
4. **メモリ・PLE・MTP・発熱**（Memory, PLE, MTP & heat）
   よく使う設定をワンクリックで切り替えられます。
   - **メモリ確保枠**（Memory reserve）：**ゆったり** / **標準** / **エキスパート最大**（Most experts）
   - **メモリの固定**（Wired memory、macOS Metal のみ）：**オフ**（既定、生成中のみ固定） / **自動**（`QUANTEEZE_WIRED_GB=auto`：モデル読み込み時から Metal 推奨作業領域までワイヤード固定し、初回リクエスト前のメモリ圧縮を抑制。GPU アイドル時は約 30 秒で macOS が解放）
   - **PLE n-gram 表**（PLE n-gram table）：**自動** / **SSD** / **Memory**
   - **KV キャッシュ**（KV cache）：**bfloat16** / **8 ビット**（`QUANTEEZE_KV_BITS=8`：Qwen3.8 Flash-Next で約 0.53 倍、Qwen3.8-27B など通常 `KVCache` モデルで約 0.56 倍に省メモリ化）
   - **MTP 投機デコード**（MTP speculative decoding）：**自動** / **オン（完全一致）** / **On (fast)** / **オフ**
   - **下書き役（Drafter）**（Speculative drafter）：対応モデル選択時に **MTP ヘッド** / **DFlash2**（`QUANTEEZE_DRAFTER`）を切り替え、未保存の場合は **取得** ボタンからサイズ確認後にダウンロードできます（未取得・非対応時はモデル同梱ヘッドへ自動フォールバック）。
   - **発熱による減速**（Thermal throttle、macOS のみ）：**静か** / **標準** / **全速**
5. **アプリをつなぐ**（Connect an app）
   OpenAI 互換 URL・Anthropic 互換 URL・Claude Code / Codex CLI 用コマンド・`curl` 例が表示され、**コピー**（Copy）ボタンでそのままコピーできます。

### 7. Claude Code・Codex CLI や他のアプリからつなぐ

サーバーは **OpenAI 互換 API** と **Anthropic 互換 API** の両方を同じポートで待ち受けています。

| API | 接続先（base URL） | 主な用途 |
|---|---|---|
| OpenAI Chat Completions（`/v1/chat/completions`） | `http://127.0.0.1:8765/v1` | OpenAI SDK、KAME、各種 OpenAI 互換アプリ |
| Anthropic Messages（`/v1/messages`） | `http://127.0.0.1:8765` | Claude Code、Anthropic SDK |
| OpenAI Responses（`/v1/responses`） | `http://127.0.0.1:8765/v1` | Codex CLI、Responses API 対応クライアント |

- **モデル名と API キーは何でも構いません**：クライアント側で `local` や `claude-...` などどんな名前を指定しても、現在 Web UI（または起動オプション）で配信中のモデルが応答します。API キーも `--api-key` を付けて起動しない限り任意の文字列で通ります。
- **思考を使うエージェントでは `--enable-thinking` を付けて起動します**（付けない場合、クライアントが明示指定しない限り既定の思考モードはオフになります）。
- **長い会話でも 2 ターン目以降は高速です**：会話履歴を毎回送り直すエージェントでも、自動プレフィックスキャッシュ（APC）により過去の共通部分を再計算しません（ディスクに保存するキャッシュは AES-256-GCM で自動暗号化されるため、平文のトークンや KV キャッシュがディスクに残りません）。
- **同じ LAN 内の別マシンから接続するとき**は、`--host 0.0.0.0` を付けて起動してください。接続用 URL は Web UI の **アプリをつなぐ**（Connect an app）に表示されます。

**Claude Code**

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8765 ANTHROPIC_AUTH_TOKEN=local claude
```

**Claude Code にセットアップと起動ごと任せる場合**

`~/Quanteeze` で Claude Code を開き、たとえば次のように頼むと、環境確認から起動・URL 案内まで進めてくれます。

> README.ja.md のクイックスタートに従って Quanteeze を起動して。モデルは LM Studio や Hugging Face キャッシュにあるものを使い、無ければダウンロード前にサイズを教えて。起動したら、Web UI の `#token=` 付き URL を `#token=` まで含めてそのまま教えて。

**Codex（OpenAI Codex CLI・デスクトップアプリ）**

`~/.codex/config.toml` に Responses API（`/v1/responses`）のプロバイダ設定とプロファイルを足します（`requires_openai_auth = false` が既定／明示により OpenAI ログイン不要で動作し、ChatGPT OAuth トークンがローカル／LAN サーバーへ誤送信されるのも防げます）。

```toml
[model_providers.quanteeze]
name = "Quanteeze"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
requires_openai_auth = false
stream_idle_timeout_ms = 7200000   # 長いプロンプトの読み込みや思考で切れないように（既定は 5 分）

[profiles.local]
model_provider = "quanteeze"
model = "local"
```

- **CLI**：`codex --profile local` で、このサーバーを使います。
- **デスクトップアプリ**：ファイルの先頭（最初の `[...]` より前）に次の 2 行を書くと、このサーバーを使います（既存の `model = ...` の行があればコメントアウトし、元に戻すときはこの 2 行を消します）。

  ```toml
  model_provider = "quanteeze"
  model = "local"
  ```

- `--api-key` を付けて起動したときは、`[model_providers.quanteeze]` に `env_key = "MLX_VLM_SERVER_API_KEY"` などを足し、Codex を起動するシェルの環境変数にキーを入れてください。
- **承認の自動審査（auto-review）を使うときは、審査の締め切りを延ばしてください。** Codex の承認審査は別会話で毎回抜粋を選び直して送るためプロンプトキャッシュがほぼ効かず、約 2 万トークンのプリフィルで既定の 90 秒（`90000`）に掛かることがあります。`~/.codex/config.toml` に次を足すと 300 秒へ延長できます（Codex 0.171.0 以降、`0` でタイムアウト無効）:

  ```toml
  [auto_review]
  timeout_ms = 300000   # 自動審査の持ち時間（ミリ秒、既定は 90000）
  ```

`config.toml` を書き換えずに CLI をワンライナーで起動する場合（Web UI の **アプリをつなぐ** からもコピーできます）：

```bash
codex -c model_provider="quanteeze" -c model="local" \
  -c 'model_providers.quanteeze={name="Quanteeze",base_url="http://127.0.0.1:8765/v1",wire_api="responses",requires_openai_auth=false,stream_idle_timeout_ms=7200000}'
```

- **（任意）画像生成 MCP ツール（`quanteeze.image_mcp`）**：Quanteeze の推論サーバー自体は画像を返さないため、Codex や各種 MCP クライアントからローカルの画像生成モデルを呼び出せる独立した stdio MCP サーバー `quanteeze.image_mcp`（ツール名 `generate_image(prompt, model, width, height, seed, path)`）を用意しています。配置済みのモデルを引数 `model` で選択でき、既定値は下表の上側で見つかったもの（商用利用可能な Apache-2.0 モデルが優先）になります。

  | `model` | モデル | ライセンス | 配置するモデルフォルダ | 1 枚（1024×1024）目安 |
  |---|---|---|---|---|
  | `z-image-turbo` | Z-Image-Turbo（Tongyi-MAI, 6B） | Apache-2.0（商用可。リポジトリに LICENSE ファイルは無くモデルカード表記） | `mlx-community/Z-Image-Turbo-bf16`（20.5 GB） | 約 12 秒（9 ステップ） |
  | `flux2-klein` | FLUX.2-klein-4B（Black Forest Labs） | Apache-2.0（商用可） | `black-forest-labs/FLUX.2-klein-4B` の元重み（約 16 GB、`mlx-community` 4〜8 ビット版は mflux 形式のため `mlx-vlm` 不可） | 配置時に自動認識 |
  | `qwen-image` | Qwen-Image-2.1 | Qwen Research License（研究・評価用途のみ） | `mlx-community/Qwen-Image-2.1-MLX-4bit`（10.5 GB） | 約 44 秒（30 ステップ） |
  | `nemotron-image` | Nemotron-Labs-Image-30B-A3B | NVIDIA License（研究・評価用途のみ） | MLX 4/8 ビット変換済み（UI の **4 ビットに変換…** 等）＋元リポジトリの `Nemotron-Labs-Image-Tokenizer` | 約 23 秒 |

  上 3 つは `mlx-vlm` の画像生成パイプライン（`quanteeze.vlm_image`、解像度 256〜2048 px・16 刻み）、`nemotron-image` は `quanteeze.nemotron_image`（512〜1536 px・64 刻み）で描画します。いずれかをモデルフォルダ（`~/.lmstudio/models`・`/Volumes/*/lmstudio`・`~/.quanteeze/models`）に置き、`~/.codex/config.toml` に次を足します。

  ```toml
  [mcp_servers.images]
  command = "/path/to/Quanteeze/.venv/bin/python"   # このリポジトリの .venv の python（絶対パス）
  args = ["-m", "quanteeze.image_mcp"]               # 既定モデルを固定する場合は "--default", "flux2-klein" 等を指定
  tool_timeout_sec = 300
  ```

  - **モデル別ライセンスへの注意（Qwen-Image-2.1 / Nemotron は研究・評価目的限定・非商用）：** `Qwen-Image-2.1`（Qwen Research License）および `Nemotron-Labs-Image-30B-A3B`（NVIDIA License §3.3 — 変換済み重み・生成 PNG 画像・任意コンポーネント [`src/quanteeze/vendor/nemotron_image/l24sq.py`](src/quanteeze/vendor/nemotron_image/l24sq.py) / [`LICENSE`](src/quanteeze/vendor/nemotron_image/LICENSE) / [`NOTICE.md`](src/quanteeze/vendor/nemotron_image/NOTICE.md) を含む）は**研究または評価の目的限定（非商用）**です。商用製品や受託業務には `z-image-turbo` または `flux2-klein`（Apache-2.0）をご利用ください。MCP ツールもツール説明と応答に各モデルのライセンスを明示します（重みはこのリポジトリに含まれません）。
  - 呼び出しごとに空きメモリ（モデル容量 + 6 GB）を確認してから読み込み、1 枚描くたびに解放します。画像は既定で `~/.quanteeze/images/` に保存され、総保持容量・プロセスごとの書き込み量（既定各 2 GB、`--keep-gb` / `--write-gb`）を超えると書き込みを拒否します。

**Python（OpenAI SDK）**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="local")
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "こんにちは"}],
)
print(r.choices[0].message.content)
```

**[KAME](https://github.com/northriv/KAME)（pydantic AI エージェント）**

物性測定エージェント [KAME](https://github.com/northriv/KAME) は pydantic AI の `OpenAIChatModel`（`/v1/chat/completions`）で接続します。`~/.kame_pyai.env` に次の 2 行を書くだけで接続できます（`local:` の後ろの名前は何でも構いません）。

```bash
KAME_PYAI_MODEL=local:quanteeze
KAME_PYAI_LOCAL_URL=http://127.0.0.1:8765/v1
```

- 思考付きでツール呼び出しを行う KAME エージェントには `--enable-thinking` 付きでの起動を推奨します（Quanteeze は既定の `MLX_VLM_MAX_TOKENS` を `32768` に引き上げて長い思考後のツール呼び出し途中切れを防ぎ、`quanteeze.toolcalls` / `quanteeze.reask` で未閉じ・不正なツール呼び出しも自動補正します）。
- KAME をヘッドレス Docker（`kame-headless` イメージ）で動かす場合も、コンテナ内で動くのは KAME 本体と MCP サーバーだけで、pydantic AI エージェントはホストの Mac/Linux 上から `127.0.0.1:8765` へ直接接続します。モデルへの要求はコンテナを通らないため `--host 0.0.0.0` は不要です。

**curl**

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

### Linux（NVIDIA GPU）のクイックスタート

Linux + NVIDIA GPU 環境でもそのまま動きます。MoE エキスパートが GPU の VRAM に入りきらない場合、Quanteeze は空いている**メインメモリ（ホスト RAM）**にエキスパートや PLE / Engram テーブル（PLE はピン留め対応）を自動キャッシュし（統合メモリ型 CUDA GPU や `/dev/shm` 上では重複確保を自動回避）、PCIe 経由で GPU に転送しながら推論できます。また `g++` が利用可能な AVX2 / AVX-512 ホストでは、`QUANTEEZE_CPU_EXPERTS=auto`（既定）により 1〜8 リクエスト同時デコード時のホスト RAM 常駐ミスを GPU 常駐エキスパートと並行して GPU と同一 NUMA ノードの CPU 側（`cpuset` 内での自動譲歩＋バッチジョブ時は通常優先度）で自動計算します。また SSD からのオンデマンド読み込みではロックフリー・GIL フリーな C++ `pread` リーダー（`QUANTEEZE_DEMAND_READ_NATIVE=auto`）が並列実行します。さらに複数の GPU がある場合は `QUANTEEZE_PARALLEL=auto` で `qwen4_exp`・`glm5_next`・`deepseek_v41` を自動的にレイヤー並列で分割配信します。

**1. 必要なもの**

- **NVIDIA GPU** と CUDA 12 以上（CUDA 13 も選択可）に対応したドライバ（`nvidia-smi` 右上の CUDA Version で確認）
- モデルを置く **SSD の空き容量** と、なるべく余裕のある **メインメモリ**
- **[uv](https://docs.astral.sh/uv/)**：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- **（任意）C++20 対応 `g++`（`g++` 10 以上）**：初回起動時に同梱の [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc) と C++20 拡張群（[`kamestm`](https://github.com/northriv/KAME/tree/master/kamestm) ベースの STM オンデマンドエンジン `stm_core`、`native_reads`、AVX2/AVX-512 `cpu_experts`）を自動ビルドしたい場合（例：Ubuntu/Debian なら `sudo apt install g++`）

**2. 取得とインストール**

Linux では MLX 用の CUDA バックエンド（`cuda` または `cuda13` の extra）を指定してインストールします。

```bash
git clone https://github.com/northriv/Quanteeze.git ~/Quanteeze
cd ~/Quanteeze

# CUDA 12（既定）
uv sync --extra cuda

# CUDA 13 を使う場合
# uv sync --extra cuda13
```

> `pip` を使う場合は `pip install -e ".[cuda]"`（または `".[cuda13]"`）、GPU なしの CPU 環境で動作確認だけする場合は `uv sync --extra cpu` を指定します。

GPU が認識されているかは、次の 1 行で確かめられます（`Device(gpu, 0)` と表示されれば準備完了です）。

```bash
.venv/bin/python -c "import mlx.core as mx; print(mx.default_device())"
```

**3. モデル**

Linux 版 LM Studio は MLX 形式を扱わないため、Hugging Face のキャッシュ（`~/.cache/huggingface/hub`）にあるモデルか、`--model <モデルのフォルダ>` で指定したチェックポイントを使います。手元にモデルがない場合は、起動後に Web UI の検索欄からダウンロードできます。

**4. 起動**

```bash
cd ~/Quanteeze
uv run --extra cuda quanteeze --port 8765
# または仮想環境のコマンドを直接実行：
# .venv/bin/quanteeze --port 8765
```

> **重要**：Linux で `uv run` や `uv sync` を使うときは、必ず `--extra cuda`（CUDA 13 なら `--extra cuda13`）を付けてください。`--extra` を付けずに `uv run quanteeze` を実行すると、`uv` が `mlx-cuda-12` などを環境から外してしまいます。`.venv/bin/quanteeze` で直接起動する場合はそのまま実行できます。

**5. 手元の PC から Web UI を開く（SSH ポート転送）**

サーバーが `127.0.0.1:8765` で待ち受けるため、モニターのない Linux 機で動かす場合は、手元の Mac や PC から SSH でポート転送し、手元のブラウザでターミナルに出た `#token=` 付き URL を開きます。

```bash
ssh -N -L 8765:127.0.0.1:8765 <Linux機>
```

> **Linux での補足**：発熱に応じた自動減速（`QUANTEEZE_THROTTLE_AT`）は macOS 専用のため、Linux では動作しません。

---

## コマンドラインから直接指定して起動する例

Web UI からモデルを切り替える代わりに、起動時からモデルや量子化・メモリ設定を指定することもできます。

```bash
# チェックポイント元の精度で配信（メモリに収まればメモリ保持、収まらなければ自動で SSD ストリーミング）
uv run quanteeze --model /path/to/checkpoint

# 全ルーティング・エキスパートを均一に 2 ビット（最小二乗）へ量子化して配信
QUANTEEZE_BITS=2 uv run quanteeze --model /path/to/checkpoint

# 均一 3 ビットに量子化し、計算結果をディスクにキャッシュして次回起動を高速化
QUANTEEZE_BITS=3 QUANTEEZE_QUANT_CACHE=~/.quanteeze/quant-cache uv run quanteeze --model /path/to/checkpoint

# K-quant（Q2_K）＋コンパクトモード（PLE/Engram・Vision・Embedding を SSD ストリーミングしてエキスパート枠を最大化）
QUANTEEZE_BITS=q2_k QUANTEEZE_COMPACT=on uv run quanteeze --model /path/to/checkpoint

# 本体（Attention・PLE/Engram・Vision・MTP 等）は MLX のまま、ルーティング・エキスパートだけ配布済み UD-IQ* GGUF から直接読む
# （単一 .gguf / 分割シャードの *-00001-of-0000N.gguf / .gguf を含むフォルダのいずれも指定可能。収まらなければ自動で SSD ストリーミング）
uv run quanteeze --model /path/to/DeepSeek-V4.1-Flash-1017 --experts-gguf /path/to/UD-IQ1_M

# 他プロセスとメモリを分け合う場合など、GGUF エキスパートの常駐枠と総メモリ上限を明示して絞る例
QUANTEEZE_FROM_SSD_GB=16 QUANTEEZE_KEEP_GB=38 uv run quanteeze --model /path/to/mlx-checkpoint --experts-gguf /path/to/UD-IQ1_M

# 思考モードを有効にし、LAN 内の他端末からも接続できるようにして起動
uv run quanteeze --model /path/to/checkpoint --enable-thinking --host 0.0.0.0 --port 8765
```

---

## 実機での動作・速度例（`--experts-gguf`）

Apple Silicon MacBook Pro（128 GB Unified Memory / M5 Max / 内蔵 SSD）で、`DeepSeek-V4.1-Flash-1017`（MLX）と Unsloth 配布の `UD-IQ1_M` GGUF を単体・既定設定（`QUANTEEZE_FROM_SSD_GB`・`QUANTEEZE_KEEP_GB` 未指定）で組み合わせて配信した実測例です（2026-11-15 計測）。

- **構成**：
  - **ベースモデル（MLX）**：`DeepSeek-V4.1-Flash-1017`（Attention・FP8 射影・Engram・Vision・2 ビット DSpark MTP ヘッドを MLX チェックポイント側から使用。常駐本体 **9.9 GB**、DSpark MTP **4.9 GB**）
  - **エキスパート（`--experts-gguf`）**：Unsloth `UD-IQ1_M`（層・射影別に `IQ1_S` / `IQ2_XXS` / `IQ2_XS` / `IQ2_S` が混在。ルーティング・エキスパート総量 **132.2 GB** で、元の MXFP4 **288.8 GB** から約 **2.18 倍圧縮**）
  - **既定の自動メモリ配分**（Metal 推奨作業領域 **115.4 GB**、`QUANTEEZE_MEMORY_RESERVE=0.3`）：エキスパート常駐枠 **71.0 GB**（各層 **206 / 384 個 = 53.6%** を RAM 常駐、残りを SSD からオンデマンド読込）

固定 3 プロンプト（英語解説・Python コード生成・日本語解説）× ウォームアップ 1 回 + 計測 2 回 = 各 6 ラン（`max_tokens=512`, `temperature=1.0`, `top_p=0.95`, `top_k=20`）での結果：

| 設定 | ロード時間 | Pure Decode（平均 / 中央値 / 最小〜最大） | Wall 速度（平均 / 最大, prefill 込） | SSD ヒット率 | DSpark 採択率 | `gguf_qmv` カーネル | 最大保持量（上限 `115.4 GB`） |
|---|---|---|---|---|---|---|---|
| `QUANTEEZE_MTP=auto`（既定・2-bit DSpark ON） | 31.0 秒 | **16.5 tok/s** / 16.2 / 12.2〜22.2 | 15.9 / 22.1 tok/s | 93.8% | **90.1%**（2.26 tok/round） | mismatch 0（150,171 回） | **91.0 GB**（超過 0 回） |
| `QUANTEEZE_MTP=off`（1 トークンずつ） | 24.4 秒 | **13.5 tok/s** / 13.4 / 11.7〜15.8 | 13.0 / 14.6 tok/s | 96.2% | — | mismatch 0（292,158 回） | **85.6 GB**（超過 0 回、スワップ増分 0.0 GB） |

### 運用のポイント

- **MTP（DSpark）との相性**：MLX 側の 2 ビット DSpark MTP ドラフターと GGUF `UD-IQ1_M` エキスパートは **90.1%** の高い採択率で噛み合い、Python コード生成では **最大 22.2 tok/s**（`MTP=off` の 13.5 tok/s 比 **+64%**）まで高速化しました（連続生成の後半では macOS のサーマル制御 `thermal pressure 2`＝GPU 稼働 85% 制限が作動）。
- **Engram 専用シャードは省略可能**：DeepSeek V4.1 の 5 分割 GGUF のように末尾シャード（`00005`）が Engram テーブル専用の場合、エキスパートを含む `00001`〜`00004` の 4 シャードだけで全 40 層を自動認識します（Engram は MLX チェックポイント側を参照します）。
- **同一 Mac での二重起動は非推奨**：macOS の Unified Memory 環境では単独稼働を前提にエキスパート枠（128 GB 機で約 71 GB）を確保するため、複数の MLX サーバーを既定設定のまま同時起動するとメモリ不足・スワップ逼迫を招きます（起動時に別サーバーを検知すると `WARNING another MLX server process is already running` を出力します）。基本的には 1 本ずつ起動し、どうしても並走させる場合は `QUANTEEZE_FROM_SSD_GB` と `QUANTEEZE_KEEP_GB` でメモリ上限を明示的に絞ってください。

### Linux / NVIDIA GPU での実測例（RTX 5060 Ti 16 GB・CUDA 13）と運用のポイント

**RTX 5060 Ti（16 GB VRAM）**・**ホストメモリ 62 GB**・**CUDA 13**（`mlx-cuda-13` 0.32.4、`QUANTEEZE_QUANT_CACHE` 未指定、`temperature=0`、`max_tokens=320`、カーネル不一致 0 件）での実測例です：

| モデルと設定 | 配置 | デコード速度 | 備考 |
|---|---|---|---|
| `Qwen3.6-35B-A3B`（`bf16`・66 GB）＋ `QUANTEEZE_BITS=2` | 自動 SSD ストリーミング（各層 `175〜178 / 256` 個を VRAM 保持、エキスパート 9.1 GB） | **15.1〜15.8 tok/s** | 16/16 リクエスト正常完了 |
| `Qwen3.6-35B-A3B` ＋ `QUANTEEZE_BITS=2 QUANTEEZE_LOAD_WHOLE=on` | 16 GB VRAM へ強制全保持（MLX 14.9 GB / GPU 15.4 GB） | **77.4 tok/s** | 長めのプロンプト取り込み時に OOM（`cudaMallocAsync`、成功 9/16） |
| `Qwen3.8-Flash-Next`（4 ビット・エキスパート 67.9 GB、`MTP=auto`） | 自動（各層 `43 / 512` 個を VRAM ＋ 42.1 GB をホスト RAM 保持） | **2.6〜2.8 tok/s** | VRAM ミスの 61% をホスト RAM から供給。`QUANTEEZE_QUANT_CACHE` 無しで `QUANTEEZE_BITS=2`/`3` に下げると **1.6〜1.9 tok/s** に低下 |

- **VRAM に収まらないモデルでは `QUANTEEZE_QUANT_CACHE` 無しの `QUANTEEZE_BITS` を使わない**：ディスクキャッシュ未指定のまま均一再量子化するとホスト RAM キャッシュ（`HostCopy`）が使われず、ミスのたびに SSD から読み直して再量子化するため、元精度のまま（ホスト RAM キャッシュ併用）や `--experts-gguf` より遅くなります。`QUANTEEZE_BITS` は VRAM に収まる場合、または `QUANTEEZE_QUANT_CACHE` 併用時に使ってください。
- **プリフィル用の VRAM 余裕と 1 ビット量子化の注意**：空き VRAM がほとんど残らない状態で `QUANTEEZE_LOAD_WHOLE=on` を強制するとプリフィル時に OOM になります。また `QUANTEEZE_BITS=1` は 16 GB VRAM に収まりますが（35B で 5.0 GB）、出力が大きく崩れるため通常は 2 ビット以上または `--experts-gguf` を推奨します。
- **CUDA 13 と CPU オフロード**：`QUANTEEZE_NVRTC_PCH=on` は読み込み済みの `libnvrtc.so.12` / `libnvrtc.so.13` を自動検出します。`QUANTEEZE_CPU_EXPERTS=auto` は現在 1 トークンずつのデコードパスで動作します（マルチトークンの MTP 検証パスは GPU 側で実行されます）。

---

## 主な特徴・仕組み

### 日常運用で大切にしている設計方針

- **専用形式への再パッキングや事前キャリブレーションが不要**：独自のディスク配置形式への変換やオフラインの共起プロファイリングを行わなくても、LM Studio や Hugging Face キャッシュにある標準の MLX `safetensors` や配布済みの `UD-IQ*` GGUF を**置いたまま・コピーなしで直接**読み込めます。数百 GB の重みを別形式で二重に保持する必要がなく、公開されたばかりのチェックポイントも取得してすぐに動かせます。
- **ユニファイドメモリとディスクリート GPU の構造差に素直なメモリ階層**：CPU と GPU が同一物理メモリを共有する Apple Silicon（Unified Memory）ではホスト RAM への二重キャッシュを行わず Metal 推奨作業領域・ワイヤード制御・サーマル制御に合わせ、VRAM が独立した Linux + NVIDIA GPU ではホスト RAM キャッシュ・GPU 直結 NUMA ノードでの AVX2/AVX-512 CPU 並列計算（他プロセス負荷時の自動譲歩つき）・マルチ GPU 自動分割を組み合わせるなど、ハードウェアごとのメモリ構造に無理のない経路を自動で選びます。
- **画像を含む長いエージェント対話と再起動をまたぐ永続キャッシュ**：テキストだけでなく画像入力（Vision tower のオンデマンド `mmap` 読込を含む）をそのまま扱え、画像コンテンツハッシュ対応・AES-256-GCM 暗号化・伸長軸の差分チェイン保存を備えた永続プレフィックスキャッシュ（APC）や、プリフィル途中キャンセル時の既読トークン保存により、サーバー再起動後や十数万トークン規模の対話でも過去の文脈や画像を再計算しません。長文脈向けにフルアテンション層の 8-bit KV キャッシュ（`QUANTEEZE_KV_BITS=8`）と選択キーのみのスパースデコードも備えています。
- **複数エージェントの同時リクエストを止めない並行調停**：形式検証済みロックフリー C++20 STM（[`kamestm`](https://github.com/northriv/KAME/tree/master/kamestm)）によるスロット保護リースと、プロンプト処理／デコード間の GPU ターン調停（`QUANTEEZE_PROMPT_DECODE_TURNS`）により、コーディングエージェントが複数のサブエージェントを並走させても、長いプリフィルの裏で進行中のデコード応答を滑らかに返し続けます。
- **速さのために数値の確かさを犠牲にしない実行時照合**：自作の Metal / CUDA カーネルや投機的デコード検証（`QUANTEEZE_MTP_EXACT=bits`・DSpark・DFlash2）は、実際のテンソル形状ごとに標準演算とビット単位（量子化 KV パスでは許容誤差内）で照合してから適用され、万一合わなければ自動で基準経路へフォールバックします。

### 各機能の詳細

- **エキスパート量子化（`QUANTEEZE_BITS`）**
  - ルーティング・エキスパートを GPU 上の閉形式最小二乗フィッティングで **8 / 4 / 3 / 2 / 1 ビット** へ量子化、または **MXFP4**（`mxfp4`）・GGML K-quant / importance-grid（**`q3_k`** / **`q2_k`** / **`iq2_xs`**）へ変換、もしくは元精度のまま配信できます。
  - 下位ビットへの量子化は、モデルがメモリに収まる場合はメモリ上で、収まらない場合は SSD から読み込みながらオンザフライで実行し、`QUANTEEZE_QUANT_CACHE` でディスクに保存して再起動時の変換を省略できます。
- **配布済み GGUF（`UD-IQ*` 等）からのエキスパート直接配信（`--experts-gguf` / `QUANTEEZE_EXPERTS_GGUF`）**
  - モデル本体・Attention・PLE / Engram・Vision・MTP は元の MLX チェックポイントのまま保持しつつ、ルーティング・エキスパート重みだけを配布済みの GGUF（単一ファイル、分割シャード `*-00001-of-0000N.gguf`、またはフォルダ）から直接読み込みます。
  - Unsloth 等の量子化済み混合精度 GGUF（層・射影ごとに異なる `IQ1_S` / `IQ1_M` / `IQ2_XXS` / `IQ2_XS` / `IQ2_S` / `IQ3_XXS` / `IQ4_XS` / `IQ4_NL` / `MXFP4` / `Q2_K`〜`Q6_K` / `Q4_0`〜`Q8_0` 等が混在するビルド）にも対応し、MLX 形式へ変換せず ggml 生ブロックのままメモリ保持・SSD オンデマンド読み込み（`QUANTEEZE_FROM_SSD_GB`）・CUDA 環境でのホスト RAM キャッシュ（`QUANTEEZE_HOST_EXPERTS_GB`）に対応し、Metal / CUDA 融合カーネル（`gguf_qmv`）で直接実行できます。
- **メモリ保持とオンデマンド SSD / ホスト RAM ストリーミングの自動切り替え**
  - 利用可能な作業領域（`QUANTEEZE_MEMORY_RESERVE` を差し引いた余裕）にエキスパートが収まる場合はすべてメモリに保持します。
  - 収まらない場合は自動で **オンデマンド SSD ストリーミング**（`QUANTEEZE_FROM_SSD_GB`）へ切り替わり、メモリ内 LRU スロットバンク、省メモリな 1 レイヤーずつのプリフィル（`QUANTEEZE_PREFILL_LAYERS`、`QUANTEEZE_DEMAND_PROMPT_GB`、`QUANTEEZE_PROMPT_MOE_TOKENS`）、および CUDA 環境でのホスト RAM キャッシュ（`QUANTEEZE_HOST_EXPERTS_GB`）を組み合わせて実行します。
- **ロックフリー C++20 STM オンデマンド配信エンジンと並行リクエスト調停（[`kamestm`](https://github.com/northriv/KAME/tree/master/kamestm)、`QUANTEEZE_DEMAND_STM=auto`）**
  - [KAME](https://github.com/northriv/KAME) の TLA+ / GenMC（RC11）形式検証済みロックフリー Software Transactional Memory（`Transactional::Node` / `Snapshot` / `Transaction`、タグ付きポインタ CAS 参照カウント `atomic_shared_ptr` / `local_shared_ptr`、および macOS `__ulock_wait` / Linux `futex` による mutex レス待機 `XWaitCell` — [arXiv:2608.12024](https://arxiv.org/abs/2608.12024)）を搭載しています。
  - レイヤーごとのスロット配置表、GPU 実行中パスのスロット保護リース、およびロックフリー C++ `pread` プール（`native_reads`、`QUANTEEZE_DEMAND_READ_NATIVE=auto`）による非同期 SSD ランダーを、Python GIL を握らず **O(1) スナップショット読み取り＋CAS コミット** で制御します。
  - `QUANTEEZE_REQUEST_THREADS` > 1 で複数リクエストを専用ワーカー＋独立 MLX ストリームで並行実行する際は、STM `TurnNode`（`QUANTEEZE_PROMPT_DECODE_TURNS=on`）がプロンプト処理とデコード間の GPU ターンを最古到着順に調停しつつデコード同士は自由にオーバーラップさせ、競合時はプリフィルを `QUANTEEZE_PROMPT_CONTENDED_TOKENS`（既定 `512`）トークン窓へ分割します。
- **PLE・Engram・Vision・トークン埋め込みの SSD オフロード（`QUANTEEZE_COMPACT`）**
  - **PLE / Engram テーブル（`QUANTEEZE_NGRAM_SSD`）**：数十 GB 規模の Per-Layer Embedding や DeepSeek V4.1 Engram テーブルを `madvise(MADV_WILLNEED)`、マルチスレッド先読み、クロスレイヤー非同期プリフェッチ、または CUDA ピン留めホストメモリを用いて行単位で参照します。
  - **Vision tower・トークン埋め込み（`QUANTEEZE_VISION_SSD`、`QUANTEEZE_EMBED_SSD`）**：画像エンコード時だけ Vision ブロックを `mmap` で順次読み込み、非共有トークン埋め込みも行単位で `mmap` 参照することで、マルチモーダル機能を保ったまま 1〜1.5 GB のメモリをエキスパート用に空けます。
- **ビット単位で一致検証済みの Metal / CUDA 専用カーネル**
  - マルチトークン検証 QMV（`verify_qmv`）、単一パス・ルーティングエキスパートカーネル（`expert_qmv`（Affine / MXFP4）・`cuda_experts`（Affine / MXFP4））、GGUF 直接実行カーネル（`gguf_qmv`、Metal/CUDA 両対応）および Metal K-quant / IQ2_XS カーネル（`kquant_qmv`）を搭載。
  - Hyper-connection / Sinkhorn / FP8（`hyper`、`glm_hyper`、`cuda_v41`）、Linear Attention / Gated Delta Rule（`linear_attn`、`glm_linear_attn`）、ルーター（`sigmoid_router`、`cuda_router`）、疎アテンションインデクサ（`glm_indexer`、`cuda_attn`）、LM ヘッド（`cuda_head`）を各テンソル形状ごとに標準 MLX とビット一致検証してから適用します。
- **MTP & DSpark 投機的デコーディング（`QUANTEEZE_MTP`）**
  - チェックポイント付属の MTP ヘッドや DeepSeek V4.1 の信頼度ゲート付き DSpark ヘッドで複数トークンを先読みし、1 トークンずつのデコードとビット一致する `QUANTEEZE_MTP_EXACT=bits`、またはより高速な `distribution` モードで検証します。
- **マルチ GPU レイヤー並列（`QUANTEEZE_PARALLEL`）**
  - 単一 GPU に収まらないチェックポイント（`qwen4_exp`・`glm5_next`・`deepseek_v41`）を、`QUANTEEZE_BITS` や `--experts-gguf` のサイズ縮減および実際の GPU 空き VRAM に応じて複数 GPU へ自動でレイヤー分割します（`QUANTEEZE_PARALLEL=auto`）。
- **省 RSS・ロックフリー 4 階層ホストプールアロケータ（[`kamepoolalloc`](https://github.com/northriv/kamepoolalloc)、`QUANTEEZE_ALLOCATOR=auto`）**
  - 同梱の [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc)（v1.4.0 / `rss-zero-loss`、TLA+ / GenMC 形式検証済み）を macOS（Mach-O `__DATA,__interpose` + `DYLD_INSERT_LIBRARIES`）および Linux（ELF strong symbols + `LD_PRELOAD`）で自動ビルド・プリロードします。
  - **4 階層設計**：スレッドごとのバケット群（`1 B .. 32 KiB`）・専用チャンク（`32 KiB .. 4 MiB`）・L1/L2 リサイクルキャッシュ付き大規模 `mmap`（`4 .. 32 MiB`）・超大規模 `mmap`（`> 32 MiB`）を一つの解放パスで扱い、`kamestm` と共通の `atomic_smart_ptr.h` により終了スレッドの孤立チャンクもロックフリー回収します。
  - オンデマンド L1 バンド開通・ページ単位の遅延プレフィルにより未接触ページの RSS 上昇を抑え、SSD 読み込み用の一時スライスバッファを再利用して `mmap`/`munmap` のオーバーヘッドを削減し、モデル切り替えやメモリ回収時に自動でトリムします。
- **運用・実行ガード機構**
  - MLX コマンドバッファサイズ（`MLX_MAX_OPS_PER_BUFFER`、`MLX_MAX_MB_PER_BUFFER`）および NVIDIA GPU ごとの CUDA Graph / PTX キャッシュ（`MLX_PTX_CACHE_DIR`）の自動設定、macOS Metal ワイヤードメモリ固定（`QUANTEEZE_WIRED_GB`）、メモリ上限キーパー（`QUANTEEZE_KEEP_GB`）、CUDA 未使用プール解放（`QUANTEEZE_GPU_FREE_GB`）、メモリ残量を考慮したバッチ受入制御（`QUANTEEZE_ADMIT_SHARE`、`QUANTEEZE_ADMIT_AHEAD`）、AES-256-GCM 暗号化プレフィックスキャッシュ（`QUANTEEZE_APC_ENCRYPT`）、macOS 発熱スロットリング（`QUANTEEZE_THROTTLE_AT`）、繰り返しループ停止（`QUANTEEZE_LOOPS`）、DSML ツール呼び出し検証と自動再試行（`QUANTEEZE_REASK`）。

---

## 対応モデル（アーキテクチャ）

`mlx-vlm` で読み込める Dense / MoE モデル全般に対応しており、以下のモデルファミリーには専用の推論・カーネル最適化を備えています。

- **Qwen3.8 Flash-Next**（`qwen4_exp`）
- **Qwen3.5 / Qwen3.6 MoE**（`qwen3_5_moe`）、**Qwen3-Next**（`qwen3_next`）
- **DeepSeek V4.1**（`deepseek_v41` — FP4 エキスパート、FP8 射影、Engram テーブル、DSpark MTP）および **DeepSeek V4**（`deepseek_v4`）
- **GLM-5.3**（`glm5_next`）
- その他 `mlx-vlm` が対応する一般的な `SwitchGLU` 系 MoE モデル・Dense モデル

---

## 主な環境変数（設定リファレンス）

すべての `QUANTEEZE_*` 環境変数は起動時に名前チェックされ、タイプミスがあると警告が出ます。全設定の一覧、既定値、反映タイミング（`start` / `load` / `request`）、現在の適用値は次のコマンドで確認できます（Web UI 右上の **詳細設定 →** からも編集できます）。

```bash
uv run python -m quanteeze.variables
# Markdown の表として出力する場合：
uv run python -m quanteeze.variables --markdown
```

### エキスパート量子化・ストリーミング

| 変数 | 既定値 | 説明 |
|---|---|---|
| `QUANTEEZE_BITS` | 未設定 | 全ルーティング・エキスパートの均一精度：`8` / `4` / `3` / `2` / `1`（GPU 最小二乗）、`mxfp4`、`q3_k` / `q2_k` / `iq2_xs`、または `off` / 未設定（元の精度）。 |
| `QUANTEEZE_EXPERTS_GGUF` | 未設定 | 本体・Vision・MTP は MLX チェックポイントのまま、エキスパート重みだけを読み込む配布済み GGUF パス（単一・分割シャード・フォルダ、`--experts-gguf` と同義）。オンデマンド読み込みや CUDA `HostCopy` にも対応。 |
| `QUANTEEZE_QUANT_CACHE` | 未設定 | 量子化済みエキスパート重みを保存し、再起動時に再利用するディレクトリ。 |
| `QUANTEEZE_FROM_SSD_GB` | 未設定 | エキスパートを SSD からオンデマンド読み込みし、メモリに保持する容量 GB（`auto` は空き容量に合わせる）。 |
| `QUANTEEZE_LOAD_WHOLE` | 未設定 | `on` で空き容量に関わらず全重みをメモリへ読み込む。 |
| `QUANTEEZE_HOST_EXPERTS_GB` | `auto` | CUDA 環境でホスト RAM に保持するエキスパート容量 GB（MLX safetensors / `--experts-gguf` 両対応）。統合メモリ型 GPU や `/dev/shm` 上では自動無効化。 |
| `QUANTEEZE_CPU_EXPERTS` | `auto` | `HostCopy` 常駐のデコード（`B=1..8`）ミスを PCIe 転送せず CPU で並行計算する（`auto` / `on` / `off`、`cpuset` 内の他プロセス負荷に応じて自動譲歩）。 |
| `QUANTEEZE_PREFILL_LAYERS` | `16384` | SSD ストリーミング時、この長さ以下のプロンプトを 1 レイヤーずつプリフィル処理する。 |
| `QUANTEEZE_PROMPT_MOE_TOKENS` | `2048` | CUDA の 1 レイヤーずつプリフィル時、ビット一致検証済みの 512 トークン塊を MoE パスあたりこのトークン数までまとめて実行する（`off` または `0` で無効）。 |
| `QUANTEEZE_DEMAND_PROMPT_GB` | `2` | オンデマンドプリフィル時に一時的な GPU メモリへまとめて読むエキスパートの上限 GB。 |
| `QUANTEEZE_READ_THREADS` | `auto` | オンデマンド読込に使うスレッド数（`auto`: ローカル SSD では 4、Lustre/GPFS/NFS 等では 16）。 |
| `QUANTEEZE_DEMAND_READ_NATIVE` | `auto` | オンデマンド SSD 読込のロックフリー C++ `pread` プール（`auto`: 自動ビルド＋バイト一致照合 / `off`: Python プール）。 |
| `QUANTEEZE_DEMAND_STM` | `auto` | 同梱 [`kamestm`](src/quanteeze/vendor/kamepoolalloc/README.md) ベースの C++20 STM 配置・パスリーストラッキング・ランダーエンジン（`auto`: macOS / CUDA の対応環境で有効化し非対応時は静かに Python へフォールバック / `on` / `shadow` / `off`）。 |
| `QUANTEEZE_NATIVE` | `on` | C++20 nanobind ネイティブ拡張を自動コンパイル・読み込みする（`off` で純 Python フォールバック、キャッシュ先は `QUANTEEZE_CACHE_DIR` 既定 `~/.cache/quanteeze`）。 |

### メモリ・オフロード・ハードウェア保護

| 変数 | 既定値 | 説明 |
|---|---|---|
| `QUANTEEZE_ALLOCATOR` | `auto` | 同梱 [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc)（v1.4.0 / `rss-zero-loss`）の自動ビルド・プリロード（`auto` / `on` / `off`）。 |
| `QUANTEEZE_CACHE_GB` | `auto` | MLX が保持する解放済みバッファの上限 GB（`auto`: 最大 2 GB かつ GPU 作業領域の 3% 以下、CUDA では MTP 検証パス用に最低 1 GB を維持）。 |
| `QUANTEEZE_WIRED_GB` | `0` | モデル読み込み時から MLX メモリをワイヤード固定する上限 GB（`0` / `off`: 生成中のみ mlx-vlm が固定、`auto`: Metal 推奨作業領域をロード時から固定。Metal 環境専用）。 |
| `QUANTEEZE_MEMORY_RESERVE` | `0.3` | KV キャッシュや推論用バッファのために GPU/Metal 作業領域から差し引いておく割合。 |
| `QUANTEEZE_CPU_THREADS` | `auto` | CPU エキスパート計算のワーカースレッド数（`auto`: GPU 直結 NUMA ノードの物理コア数から予約コアを除き上限 24 で各物理コアに固定）。 |
| `QUANTEEZE_CPU_PRIORITY` | `auto` | CPU エキスパートワーカーの優先度（`auto`: SLURM/PBS 等バッチジョブ時は `normal`・それ以外は `idle`、`idle` / `nice` / `normal` / `wake`）。 |
| `QUANTEEZE_CPU_MIN_THREADS` | `4` | CPU エキスパートオフロードを実行する最小スレッド数（下回ると PCIe 転送側へフォールバック）。 |
| `QUANTEEZE_CPU_YIELD_AT` | `25` | `QUANTEEZE_CPU_EXPERTS=auto` 時に他プロセスの CPU 使用率（%）がこの値以上で即座にワーカーを停止。 |
| `QUANTEEZE_COMPACT` | `off` | `on` で `NGRAM_SSD=on`、`VISION_SSD=on`、`EMBED_SSD=on`、`V41_READS_IMAGES=ssd`、`MTP=off` を一括有効化しエキスパート用メモリを最大化（`QUANTEEZE_COMPACT_KEEP` で個別に除外可能）。 |
| `QUANTEEZE_NGRAM_SSD` | `auto` | PLE / Engram n-gram テーブルの SSD 行単位ストリーミング（`auto` / `on` / `off`）。 |
| `QUANTEEZE_HOST_NGRAM_GB` | `auto` | CUDA 環境でホスト RAM に常駐させる n-gram テーブル容量 GB。 |
| `QUANTEEZE_VISION_SSD` | `auto` | 画像エンコード時のみ Vision tower を `mmap` でブロック単位に読み込む（`auto`: 全エキスパートが入り切らない時のみ SSD ストリーミング、ネットワーク FS 上では保持 / `on` / `off`）。 |
| `QUANTEEZE_EMBED_SSD` | `off` | 非共有トークン埋め込みを `mmap` で行単位に読み込む。 |
| `QUANTEEZE_APC_ENCRYPT` | `on` | ディスク上のプレフィックスキャッシュ（APC）を AES-256-GCM で暗号化保護する（`on` / `off`）。 |
| `QUANTEEZE_APC_DISK_SECRET` | `~/.quanteeze/apc-secret` | APC ディスク階層の名義と鍵へ HMAC-SHA-256 で混合する 32 バイトのマシン固有シークレット（0600・macOS は Time Machine 除外、`off` で平文トークン由来ハッシュのみ）。 |
| `QUANTEEZE_APC_READ_BACK_PLAIN` | `on` | APC から復元した 1 行・パディング無しのキャッシュ（KV 量子化・ドラフター併用なし）を通常リクエストと同じ `KVCache` / `ArraysCache` へ戻し（`quanteeze.apc_plain`）、Qwen3.5 / Qwen3.8-27B のデコード毎トークンのバッチ配列抽出・コピーを省く（`off`: `mlx-vlm` のバッチキャッシュのまま）。 |
| `QUANTEEZE_KV_BITS` | 未設定 | フルアテンション層の KV キャッシュ精度（Web UI の「KV キャッシュ」からも切替可。未設定 / `off`: bfloat16、`8`: 64 要素ごとのアフィン 8 ビット量子化。Qwen3.8 Flash-Next で 24.6 → 13.1 kB/tok の約 0.53 倍、Qwen3.8-27B など通常 `KVCache` モデルで `mlx-vlm` 8 ビットキャッシュにより 65.5 → 36.7 kB/tok の約 0.56 倍に省メモリ化、APC 保存・バッチ対応）。 |
| `QUANTEEZE_KV_BITS_SELECTED` | `on` | `QUANTEEZE_KV_BITS=8` 時、インデクサ予算を超えたデコード（1〜2 トークン）で選択されたキーのみを集めてスパース計算する（`on` / `off`）。 |
| `QUANTEEZE_KV_BITS_UNPACKED` | `auto` | `QUANTEEZE_KV_BITS=8` 時、プリフィルでは各層の 8 ビット KV をクエリ型へ展開して MLX 高速アテンションおよび Metal の QSA スパースプリフィルカーネルへ渡し、デコード・MTP 検証（1〜2 トークン）では `mlx-vlm` の量子化アテンションを 1 回で呼び出す（`auto`: Metal で有効・CUDA で無効、`on` / `off`）。 |
| `QUANTEEZE_WARM_UP_BATCH` | `0` | CUDA でロード時のウォームアップ（`QUANTEEZE_WARM_UP`）時に追加で事前コンパイルする同時バッチ行数 `2..N`（`0` で無効）。 |
| `QUANTEEZE_NVRTC_PCH` | `on` | CUDA で MLX 読み込み前に `libnvrtc_pch` をロードし、`--pch` 注入によりカスタム CUDA カーネルの NVRTC コンパイルでヘッダ解析を再利用する（`on` / `off`）。 |
| `QUANTEEZE_KEEP_GB` | `auto` | メモリキーパーが維持するサーバー総メモリ上限 GB（`off` で無効）。 |
| `QUANTEEZE_GPU_FREE_GB` | `0.5` | CUDA 環境で空き VRAM がこの値（GB）を下回った際に未使用プールを解放する。 |
| `QUANTEEZE_ADMIT_SHARE` | `0.9` | 同時プリフィル時に受け入れるバッチの作業領域占有上限割合（`QUANTEEZE_ADMIT_AHEAD` で追加伸長分を確保）。 |
| `QUANTEEZE_THROTTLE_AT` | `2` | macOS で発熱に応じた GPU スロットリングを開始するサーマルレベル（`off` で無効）。 |

### 投機デコード・マルチ GPU・リクエスト制御

| 変数 | 既定値 | 説明 |
|---|---|---|
| `QUANTEEZE_MTP` | `auto` | MTP / DSpark 投機的デコーディング（`auto` / `on` / `off` / `dflash2`）。 |
| `QUANTEEZE_DRAFTER` | `unset` | 外部ドラフトモデル（`dflash2`、リポジトリ名、またはディレクトリパス）。未取得や非対応時はモデル同梱 MTP ヘッドへ自動フォールバック。 |
| `QUANTEEZE_MTP_BITS` | `auto` | MTP ドラフトヘッド内エキスパートの均一ビット幅（`auto`: `qwen4_exp` は 4・ほかは 2、数値指定、または `off` で元精度）。 |
| `QUANTEEZE_EXPERT_QMV_VERIFY_FUSED` | `on` | 均一 MoE 層の speculative verify パスをデコード用 2 カーネル融合パス（`expert_qmv.fused`）で実行（`on` / `off`）。 |
| `QUANTEEZE_MTP_EXACT` | `bits` | 検証モード：`bits`（1 トークンデコードとビット完全一致）または `distribution`（高速な一括検証）。 |
| `QUANTEEZE_VERIFY_LEAD_LAYERS` | `auto` | Flash-Next の MTP 検証パスでホストが GPU より先に組み進めてよい最大層数（`auto`: CUDA 単一 GPU では 4（一時バッファによるマネージドメモリ退避を抑制）、Apple Silicon / マルチ GPU では `0`、`off` または `0` で無制限）。 |
| `QUANTEEZE_PARALLEL` | `auto` | マルチ GPU レイヤー並列（`auto` / `layers` / `off`）。 |
| `QUANTEEZE_GPUS` | `auto` | `QUANTEEZE_PARALLEL` 使用時の GPU 数（`auto` ですべて）。 |
| `QUANTEEZE_LOOPS` | `on` | 繰り返しループに陥った生成を自動検出して停止する。 |
| `QUANTEEZE_REASK` | `2` | DSML ツール呼び出しが不正な場合に `temperature=1.0` で自動再試行する回数。 |
| `QUANTEEZE_CONTROL_TOKEN` | 自動生成 | Web UI / 管理 API 用トークン（既定は `~/.quanteeze/control-token` を使用、設定保存先は `QUANTEEZE_SETTINGS` の `~/.quanteeze/settings.json`）。 |
| `QUANTEEZE_REQUEST_THREADS` | `1` | リクエストごとに専用ワーカー＋独立 MLX ストリームで並行実行する最大スレッド数（`QUANTEEZE_REQUEST_THREADS_ANY` / `QUANTEEZE_REQUEST_THREADS_DRAFT_TOGETHER` で細部制御）。 |
| `QUANTEEZE_PROMPT_DECODE_TURNS` | `on` | `QUANTEEZE_REQUEST_THREADS` > 1 かつ C++ STM 稼働時、プロンプト処理とデコード間の GPU ターンを最古到着順に調停しつつデコード同士を重ねて実行する（`off` で調停なし）。 |
| `QUANTEEZE_PROMPT_CONTENDED_TOKENS` | `512` | プリフィルチャンクの最大トークン数（常時）および `QUANTEEZE_PROMPT_DECODE_TURNS` 有効時の競合 MoE 窓サイズ。 |

---

## テストの実行

```bash
# macOS
uv run pytest

# Linux（NVIDIA GPU）
uv run --extra cuda pytest
```

---

## ベンチマーク（llama.cpp 互換）

配信中のサーバーに対して llama.cpp の `llama-bench`（`pp` / `tg` / `pp+tg` / `@ d<depth>`）および `llama-batched-bench`（同時 `B` リクエスト）と同じ基準・列形式で速度を測定するスクリプトを用意しています。詳細は [bench/llamacpp/README.ja.md](bench/llamacpp/README.ja.md) を参照してください。

```bash
# llama-bench 互換（pp512 / tg128 / pp512+tg128）
uv run python bench/llamacpp/llama_bench.py --url 127.0.0.1:8765 --checkpoint <チェックポイント> --log serve.log -p 512 -n 128 -pg 512,128

# llama-batched-bench 互換（B = 1, 2, 4 同時実行）
uv run python bench/llamacpp/batched_bench.py --url 127.0.0.1:8765 --checkpoint <チェックポイント> --log serve.log -npp 128,512 -ntg 128 -npl 1,2,4
```

---

## 引用（Citing）

BibTeX 等は [CITATION.cff](CITATION.cff)（または GitHub の **Cite this repository**）を参照してください。

- **Quanteeze**：K. Kitagawa, [`northriv/Quanteeze`](https://github.com/northriv/Quanteeze) (2026)
- **`kamestm`**：K. Kitagawa, [arXiv:2608.12024](https://arxiv.org/abs/2608.12024) (2026)
- **`kamepoolalloc` / KAME**：[`northriv/kamepoolalloc`](https://github.com/northriv/kamepoolalloc) · [`northriv/KAME`](https://github.com/northriv/KAME)

---

## ライセンス（License）

Copyright (C) 2026 Kentaro Kitagawa.

Quanteeze は以下のいずれかを選択できるデュアルライセンスで提供されます：

- **GNU General Public License, version 2 only**（`GPL-2.0-only`） — [LICENSE-GPL-2.0](LICENSE-GPL-2.0) を参照、または
- **PolyForm Shield License 1.0.0**（`LicenseRef-PolyForm-Shield-1.0.0`） — [LICENSE-POLYFORM-SHIELD](LICENSE-POLYFORM-SHIELD) を参照。

詳細は [LICENSE](LICENSE) をご確認ください。`src/quanteeze/vendor/deepseek_v4/`・`src/quanteeze/vendor/deepseek_v41/`・`src/quanteeze/iq_grids.py`・`bench/llamacpp/` 配下のサードパーティコンポーネントは元の MIT License 表記を保持します。[`src/quanteeze/vendor/kamepoolalloc/`](src/quanteeze/vendor/kamepoolalloc/README.md) 配下の同梱 [`kamestm`](https://github.com/northriv/KAME/tree/master/kamestm) および [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc) ソース（Copyright (C) 2002–2026 Kentaro Kitagawa）は `Apache-2.0 OR GPL-2.0-or-later` のデュアルライセンスです（[`LICENSE-APACHE-2.0`](src/quanteeze/vendor/kamepoolalloc/LICENSE-APACHE-2.0) および [`LICENSE-GPL-2.0`](src/quanteeze/vendor/kamepoolalloc/LICENSE-GPL-2.0) を参照）。なお、[`src/quanteeze/vendor/nemotron_image/`](src/quanteeze/vendor/nemotron_image/NOTICE.md) 配下の任意サードパーティファイル（`l24sq.py`）のみは Quanteeze の `GPL-2.0-only OR LicenseRef-PolyForm-Shield-1.0.0` ではなく、同フォルダの NVIDIA License（[`LICENSE`](src/quanteeze/vendor/nemotron_image/LICENSE)：**研究・評価目的限定・非商用**）のみが適用されます。通常の Quanteeze 推論サーバーからは一切 import されず、上記の任意画像ツールを使用しない場合や、`GPL-2.0-only` 条件・商用目的で Quanteeze を再配布・利用する場合は `src/quanteeze/vendor/nemotron_image/` ディレクトリを除外してください。

Quanteeze はモデルの重みを配布しません。利用する各モデル固有のライセンス（`--host` で `127.0.0.1` 以外に公開する場合の MaaS / サービス提供条項を含む）については利用者の責任で遵守してください。

**製品組み込み・機能向上版について：** GPL-2.0 / PolyForm Shield の条件外での製品組み込み、共同研究、または追加の最適化機能を備えた機能向上版にご関心のある場合は、[東京大学物性研究所 北川研究室](https://kitag.issp.u-tokyo.ac.jp/) まで別途お問い合わせください。

---

## 謝辞（Acknowledgements）

東京大学物性研究所 [北川研究室（Kitagawa Laboratory, ISSP, University of Tokyo）](https://kitag.issp.u-tokyo.ac.jp/) にて開発されました。

本研究は文部科学省「AI 等の活用を推進する研究データエコシステム構築事業（AI for 1,000 Discovery challenges: SPReAD）」（課題番号 JPMXP1726275196）の支援を受けました。数値計算には東京大学物性研究所スーパーコンピュータセンター System C（Kugui）（課題番号 2026-A-0004）を利用しました。
