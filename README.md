# Quanteeze

Fast MLX inference server for large Mixture-of-Experts (MoE) and dense language/vision-language models on **Apple Silicon (Metal)** and **NVIDIA GPUs (CUDA)**.

Apple Silicon Mac や NVIDIA GPU 搭載の Linux マシンで、巨大な MoE（Mixture-of-Experts）や VLM/LLM モデルをそのまま、または手軽に省メモリ化して配信できる MLX 推論サーバーです。均一な低ビット量子化や自動 SSD / ホスト RAM ストリーミング、内蔵ブラウザ UI、さらに **OpenAI 互換 API** と **Anthropic Messages API（Claude Code 対応）** を標準で備えています。

---

## はじめて試す方へ（クイックスタート）

ここでは、インストールからブラウザ UI での確認、Claude Code などのアプリからの接続までを順番に説明します。

### 1. 必要なもの

- **Apple Silicon の Mac**（大きな MoE モデルを快適に動かすならメモリ 32〜48 GB 以上推奨。均一な 2〜3 ビット量子化や自動 SSD ストリーミングにより、メモリ容量を超えるモデルも動作します）、または **NVIDIA GPU 搭載の Linux 機**（手順は [Linux（NVIDIA GPU）のクイックスタート](#linuxnvidia-gpuのクイックスタート)）
- モデルを置く **SSD の空き容量**（モデルによって数十〜数百 GB。Thunderbolt / USB4 接続の高速な外付け NVMe SSD でも使えます）
- **[uv](https://docs.astral.sh/uv/)**（Python 3.12 以上の環境と依存パッケージを自動で用意します）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **補足（任意）**：C++17 コンパイラ（Mac の Xcode Command Line Tools の `clang++`、または Linux の `g++`）が入っている環境では、初回起動時に同梱の省メモリ・ロックフリーアロケータ [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc) を自動ビルドして有効にします。未導入の場合は標準の `malloc` に自動でフォールバックするため、事前インストールは必須ではありません。

### 2. 取得とインストール（Mac）

置き場所は `~/Documents` や `~/Desktop` の外（たとえば `~/Quanteeze`）をおすすめします。macOS では `~/Documents` などが OS の保護対象フォルダになっており、Claude Code などのエージェントツールから使う際にアクセス許可を求められたり止められたりすることがあるためです。

```bash
git clone https://github.com/northriv/Quanteeze.git ~/Quanteeze
cd ~/Quanteeze
uv sync
```

`pip` を使う場合は Python 3.12 以上の仮想環境で `pip install -e .` を実行します。Linux（NVIDIA GPU）の手順は少し異なるため、[Linux（NVIDIA GPU）のクイックスタート](#linuxnvidia-gpuのクイックスタート) を見てください。

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

1. **モデルと均一量子化精度**（Model & uniform precision）
   このマシンで見つかったローカルモデルの一覧から使いたいモデルを選びます。まだ手元にないモデルを使う場合は、`org/model-name` を入力して **サイズ確認**（Check size）を押し、表示された合計ファイル数と容量（GB）を確認してから **Download ... GB** で取得できます。
2. **ルーティング・エキスパートの精度**（Routed expert precision）
   MoE モデルを選ぶと、元の精度や均一量子化（**4・3・2・1 ビット（最小二乗）**、**MXFP4 / Q3_K / Q2_K / IQ2_XS**）ごとに、エキスパートが使う容量と **メモリに収まる**（fits in memory）/ **SSD からオンデマンド読込**（streams from SSD）が表示されます。
   - **迷ったときの選び方**：まずは **メモリに収まる**（fits in memory）と表示される選択肢の中で一番ビット数の高いものを選ぶのがおすすめです（元の精度のままでも収まる場合は **元の精度** でそのまま高速に動きます）。メモリに入りきらないモデルを試したい場合は **SSD からオンデマンド読込** でもそのまま動きます。
   - 選んだら **このモデルを配信**（Serve this model）を押します。読み込みの進捗や現在のメモリ・発熱・速度は、画面一番上の **いま配信中**（Serving now）にリアルタイムで表示されます。
3. **質問してみる**（Try a prompt）
   ブラウザからそのままモデルに質問を送って、応答・生成速度（`tok/s`）・**思考の内容**（Its thinking）を確認できます（`Cmd+Enter` / `Ctrl+Enter` でも送信できます）。
4. **メモリ・PLE・MTP・発熱**（Memory, PLE, MTP & heat）
   よく使う設定をワンクリックで切り替えられます。
   - **メモリ確保枠**（Memory reserve）：**ゆったり** / **標準** / **Most for experts**
   - **PLE n-gram 表**（PLE n-gram table）：**自動** / **SSD** / **Memory**
   - **MTP 投機デコード**（MTP speculative decoding）：**自動** / **オン（完全一致）** / **On (fast)** / **オフ**
   - **発熱による減速**（Thermal throttle、macOS のみ）：**静か** / **標準** / **全速**
5. **アプリをつなぐ**（Connect an app）
   OpenAI 互換 URL・Anthropic 互換 URL・Claude Code 用コマンド・`curl` 例が表示され、**コピー**（Copy）ボタンでそのままコピーできます。

### 7. Claude Code や他のアプリからつなぐ

サーバーは **OpenAI 互換 API** と **Anthropic 互換 API** の両方を同じポートで待ち受けています。

| API | 接続先（base URL） | 主な用途 |
|---|---|---|
| OpenAI Chat Completions（`/v1/chat/completions`） | `http://127.0.0.1:8765/v1` | OpenAI SDK、KAME、各種 OpenAI 互換アプリ |
| Anthropic Messages（`/v1/messages`） | `http://127.0.0.1:8765` | Claude Code、Anthropic SDK |
| OpenAI Responses（`/v1/responses`） | `http://127.0.0.1:8765/v1` | Responses API 対応クライアント |

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

> README のクイックスタートに従って Quanteeze を起動して。モデルは LM Studio や Hugging Face キャッシュにあるものを使い、無ければダウンロード前にサイズを教えて。起動したら、Web UI の `#token=` 付き URL を `#token=` まで含めてそのまま教えて。

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

**KAME（pydantic AI エージェント）**

`~/.kame_pyai.env` に次の 2 行を書くだけで接続できます（`local:` の後ろの名前は何でも構いません）。

```bash
KAME_PYAI_MODEL=local:quanteeze
KAME_PYAI_LOCAL_URL=http://127.0.0.1:8765/v1
```

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

Linux + NVIDIA GPU 環境でもそのまま動きます。MoE エキスパートが GPU の VRAM に入りきらない場合、Quanteeze は空いている**メインメモリ（ホスト RAM）**にエキスパートや PLE / Engram テーブル（PLE はピン留め対応）を自動キャッシュし、PCIe 経由で GPU に転送しながら推論できます。また複数の GPU がある場合は `QUANTEEZE_PARALLEL=auto` で自動的にレイヤー並列を行います。

**1. 必要なもの**

- **NVIDIA GPU** と CUDA 12 以上（CUDA 13 も選択可）に対応したドライバ（`nvidia-smi` 右上の CUDA Version で確認）
- モデルを置く **SSD の空き容量** と、なるべく余裕のある **メインメモリ**
- **[uv](https://docs.astral.sh/uv/)**：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- **（任意）`g++`**：初回起動時に同梱の [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc) を自動ビルドしたい場合（例：Ubuntu/Debian なら `sudo apt install g++`）

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

---

## 主な特徴・仕組み

- **均一エキスパート量子化（`QUANTEEZE_BITS`）**
  - すべてのルーティング・エキスパートを GPU 上の閉形式最小二乗フィッティングで均一に **8 / 4 / 3 / 2 / 1 ビット** へ量子化、または **MXFP4**（`mxfp4`）・GGML K-quant / importance-grid（**`q3_k`** / **`q2_k`** / **`iq2_xs`**）へ変換、もしくは元精度のまま配信できます。
  - 下位ビットへの量子化は、モデルがメモリに収まる場合はメモリ上で、収まらない場合は SSD から読み込みながらオンザフライで実行し、`QUANTEEZE_QUANT_CACHE` でディスクに保存して再起動時の変換を省略できます。
- **配布済み GGUF（`UD-IQ*` 等）からのエキスパート直接配信（`--experts-gguf` / `QUANTEEZE_EXPERTS_GGUF`）**
  - モデル本体・Attention・PLE / Engram・Vision・MTP は元の MLX チェックポイントのまま保持しつつ、ルーティング・エキスパート重みだけを配布済みの GGUF（単一ファイル、分割シャード `*-00001-of-0000N.gguf`、またはフォルダ）から直接読み込みます。
  - Unsloth 等の量子化済み混合精度 GGUF（層・射影ごとに異なる `IQ1_S` / `IQ1_M` / `IQ2_XXS` / `IQ2_XS` / `IQ2_S` / `IQ3_XXS` / `IQ4_XS` / `IQ4_NL` / `MXFP4` / `Q2_K`〜`Q6_K` / `Q4_0`〜`Q8_0` 等が混在するビルド）にも対応し、MLX 形式へ変換せず ggml 生ブロックのままメモリ保持・SSD オンデマンド読み込み（`QUANTEEZE_FROM_SSD_GB`）・CUDA 環境でのホスト RAM キャッシュ（`QUANTEEZE_HOST_EXPERTS_GB`）に対応し、Metal / CUDA 融合カーネル（`gguf_qmv`）で直接実行できます。
- **メモリ保持とオンデマンド SSD / ホスト RAM ストリーミングの自動切り替え**
  - 利用可能な作業領域（`QUANTEEZE_MEMORY_RESERVE` を差し引いた余裕）にエキスパートが収まる場合はすべてメモリに保持します。
  - 収まらない場合は自動で **オンデマンド SSD ストリーミング**（`QUANTEEZE_FROM_SSD_GB`）へ切り替わり、メモリ内 LRU スロットバンク、省メモリな 1 レイヤーずつのプリフィル（`QUANTEEZE_PREFILL_LAYERS`、`QUANTEEZE_DEMAND_PROMPT_GB`）、および CUDA 環境でのホスト RAM キャッシュ（`QUANTEEZE_HOST_EXPERTS_GB`）を組み合わせて実行します。
- **PLE・Engram・Vision・トークン埋め込みの SSD オフロード（`QUANTEEZE_COMPACT`）**
  - **PLE / Engram テーブル（`QUANTEEZE_NGRAM_SSD`）**：数十 GB 規模の Per-Layer Embedding や DeepSeek V4.1 Engram テーブルを `madvise(MADV_WILLNEED)`、マルチスレッド先読み、クロスレイヤー非同期プリフェッチ、または CUDA ピン留めホストメモリを用いて行単位で参照します。
  - **Vision tower・トークン埋め込み（`QUANTEEZE_VISION_SSD`、`QUANTEEZE_EMBED_SSD`）**：画像エンコード時だけ Vision ブロックを `mmap` で順次読み込み、非共有トークン埋め込みも行単位で `mmap` 参照することで、マルチモーダル機能を保ったまま 1〜1.5 GB のメモリをエキスパート用に空けます。
- **ビット単位で一致検証済みの Metal / CUDA 専用カーネル**
  - マルチトークン検証 QMV（`verify_qmv`）、単一パス・ルーティングエキスパートカーネル（`expert_qmv`（Affine / MXFP4）・`cuda_experts`（Affine / MXFP4））、GGUF 直接実行カーネル（`gguf_qmv`、Metal/CUDA 両対応）および Metal K-quant / IQ2_XS カーネル（`kquant_qmv`）を搭載。
  - Hyper-connection / Sinkhorn / FP8（`hyper`、`glm_hyper`、`cuda_v41`）、Linear Attention / Gated Delta Rule（`linear_attn`、`glm_linear_attn`）、ルーター（`sigmoid_router`、`cuda_router`）、疎アテンションインデクサ（`glm_indexer`、`cuda_attn`）、LM ヘッド（`cuda_head`）を各テンソル形状ごとに標準 MLX とビット一致検証してから適用します。
- **MTP & DSpark 投機的デコーディング（`QUANTEEZE_MTP`）**
  - チェックポイント付属の MTP ヘッドや DeepSeek V4.1 の信頼度ゲート付き DSpark ヘッドで複数トークンを先読みし、1 トークンずつのデコードとビット一致する `QUANTEEZE_MTP_EXACT=bits`、またはより高速な `distribution` モードで検証します。
- **マルチ GPU レイヤー並列（`QUANTEEZE_PARALLEL`）**
  - 単一 GPU に収まらないチェックポイントを認識済みの複数 GPU に自動でレイヤー分割します（`QUANTEEZE_PARALLEL=auto`）。
- **省 RSS・ロックフリーのホストプールアロケータ（[kamepoolalloc](https://github.com/northriv/kamepoolalloc)、`QUANTEEZE_ALLOCATOR`）**
  - 同梱の [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc)（`rss-zero-loss` ブランチ）を macOS（`DYLD_INSERT_LIBRARIES`）および Linux（`LD_PRELOAD`）で自動ビルド・プリロードします。
  - 小〜中サイズのホスト割り当て（`16 B .. 32 KiB`）をロックフリーなスレッドごとのフリーリストから遅延 1 ページ確保で供給して未接触ページの RSS 上昇を抑え、SSD 読み込み用の一時スライスバッファ（`32 KiB .. 32 MiB`）を再利用して `mmap`/`munmap` のオーバーヘッドを減らし、モデル切り替えやメモリ回収時に自動でトリムします。
- **運用向けガード機構**
  - メモリ上限キーパー（`QUANTEEZE_KEEP_GB`）、CUDA 未使用プール解放（`QUANTEEZE_GPU_FREE_GB`）、メモリ残量を考慮したバッチ受入制御（`QUANTEEZE_ADMIT_SHARE`、`QUANTEEZE_ADMIT_AHEAD`）、AES-256-GCM 暗号化プレフィックスキャッシュ（`QUANTEEZE_APC_ENCRYPT`）、macOS 発熱スロットリング（`QUANTEEZE_THROTTLE_AT`）、繰り返しループ停止（`QUANTEEZE_LOOPS`）、DSML ツール呼び出し検証と自動再試行（`QUANTEEZE_REASK`）。

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
| `QUANTEEZE_HOST_EXPERTS_GB` | `auto` | CUDA 環境でホスト RAM に保持するエキスパート容量 GB（MLX safetensors / `--experts-gguf` 両対応）。 |
| `QUANTEEZE_PREFILL_LAYERS` | `16384` | SSD ストリーミング時、この長さ以下のプロンプトを 1 レイヤーずつプリフィル処理する。 |
| `QUANTEEZE_DEMAND_PROMPT_GB` | `2` | オンデマンドプリフィル時に一時的な GPU メモリへまとめて読むエキスパートの上限 GB。 |
| `QUANTEEZE_READ_THREADS` | `4` | オンデマンド SSD 読み込みに使うスレッド数。 |

### メモリ・オフロード・ハードウェア保護

| 変数 | 既定値 | 説明 |
|---|---|---|
| `QUANTEEZE_ALLOCATOR` | `auto` | 同梱 [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc)（`rss-zero-loss`）の自動ビルド・プリロード（`auto` / `on` / `off`）。 |
| `QUANTEEZE_MEMORY_RESERVE` | `0.3` | KV キャッシュや推論用バッファのために GPU/Metal 作業領域から差し引いておく割合。 |
| `QUANTEEZE_COMPACT` | `off` | `on` で `NGRAM_SSD=on`、`VISION_SSD=on`、`EMBED_SSD=on`、`V41_READS_IMAGES=ssd`、`MTP=off` を一括有効化しエキスパート用メモリを最大化（`QUANTEEZE_COMPACT_KEEP` で個別に除外可能）。 |
| `QUANTEEZE_NGRAM_SSD` | `auto` | PLE / Engram n-gram テーブルの SSD 行単位ストリーミング（`auto` / `on` / `off`）。 |
| `QUANTEEZE_HOST_NGRAM_GB` | `auto` | CUDA 環境でホスト RAM に常駐させる n-gram テーブル容量 GB。 |
| `QUANTEEZE_VISION_SSD` | `off` | 画像エンコード時のみ Vision tower を `mmap` でブロック単位に読み込む。 |
| `QUANTEEZE_EMBED_SSD` | `off` | 非共有トークン埋め込みを `mmap` で行単位に読み込む。 |
| `QUANTEEZE_APC_ENCRYPT` | `on` | ディスク上のプレフィックスキャッシュ（APC）を AES-256-GCM で暗号化保護する（`on` / `off`）。 |
| `QUANTEEZE_KEEP_GB` | `auto` | メモリキーパーが維持するサーバー総メモリ上限 GB（`off` で無効）。 |
| `QUANTEEZE_GPU_FREE_GB` | `0.5` | CUDA 環境で空き VRAM がこの値（GB）を下回った際に未使用プールを解放する。 |
| `QUANTEEZE_ADMIT_SHARE` | `0.9` | 同時プリフィル時に受け入れるバッチの作業領域占有上限割合（`QUANTEEZE_ADMIT_AHEAD` で追加伸長分を確保）。 |
| `QUANTEEZE_THROTTLE_AT` | `2` | macOS で発熱に応じた GPU スロットリングを開始するサーマルレベル（`off` で無効）。 |

### 投機デコード・マルチ GPU・リクエスト制御

| 変数 | 既定値 | 説明 |
|---|---|---|
| `QUANTEEZE_MTP` | `auto` | MTP / DSpark 投機的デコーディング（`auto` / `on` / `off`）。 |
| `QUANTEEZE_MTP_BITS` | `2` | MTP ドラフトヘッド内エキスパートの均一ビット幅。 |
| `QUANTEEZE_MTP_EXACT` | `bits` | 検証モード：`bits`（1 トークンデコードとビット完全一致）または `distribution`（高速な一括検証）。 |
| `QUANTEEZE_PARALLEL` | `auto` | マルチ GPU レイヤー並列（`auto` / `layers` / `off`）。 |
| `QUANTEEZE_GPUS` | `auto` | `QUANTEEZE_PARALLEL` 使用時の GPU 数（`auto` ですべて）。 |
| `QUANTEEZE_LOOPS` | `on` | 繰り返しループに陥った生成を自動検出して停止する。 |
| `QUANTEEZE_REASK` | `2` | DSML ツール呼び出しが不正な場合に `temperature=1.0` で自動再試行する回数。 |
| `QUANTEEZE_CONTROL_TOKEN` | 自動生成 | Web UI / 管理 API 用トークン（既定は `~/.quanteeze/control-token` を使用、設定保存先は `QUANTEEZE_SETTINGS` の `~/.quanteeze/settings.json`）。 |

---

## テストの実行

```bash
# macOS
uv run pytest

# Linux（NVIDIA GPU）
uv run --extra cuda pytest
```

---

## ライセンス（License）

Copyright (C) 2026 Kentaro Kitagawa.

Quanteeze is dual-licensed under your choice of either:

- **GNU General Public License, version 2 only** (`GPL-2.0-only`) — see [LICENSE-GPL-2.0](LICENSE-GPL-2.0), OR
- **PolyForm Shield License 1.0.0** (`LicenseRef-PolyForm-Shield-1.0.0`) — see [LICENSE-POLYFORM-SHIELD](LICENSE-POLYFORM-SHIELD).

See [LICENSE](LICENSE) for full details. Third-party components under `src/quanteeze/vendor/deepseek_v4/`, `src/quanteeze/vendor/deepseek_v41/`, and `src/quanteeze/iq_grids.py` retain their original MIT License notices. Bundled [`kamepoolalloc`](https://github.com/northriv/kamepoolalloc) sources under `src/quanteeze/vendor/kamepoolalloc/` (Copyright (C) 2002–2026 Kentaro Kitagawa) are dual-licensed under `Apache-2.0 OR GPL-2.0-or-later`.

---

## 謝辞（Acknowledgements）

Developed at [Kitagawa Laboratory, ISSP, University of Tokyo](https://kitag.issp.u-tokyo.ac.jp/).

This work was supported by the MEXT Supporting Pioneering Research through
AI for 1,000 Discovery challenges Program (SPReAD), Japan, Grant Number
JPMXP1726275196. Computations used System C (Kugui) of the Supercomputer
Center, Institute for Solid State Physics, the University of Tokyo
(2026-A-0004).
