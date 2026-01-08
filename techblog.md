# NeMo-RLで始める大規模言語モデルのSFT（Supervised Fine-Tuning）完全ガイド

## はじめに

大規模言語モデル（LLM）の性能を特定のタスクに最適化するために、**SFT（Supervised Fine-Tuning）** は欠かせない技術です。しかし、数十億パラメータを持つモデルのファインチューニングには、分散学習の知識やGPUメモリ管理など、多くの技術的課題があります。

本記事では、NVIDIAが開発したオープンソースライブラリ **NeMo-RL** を使って、Slurm環境でカスタムデータセットを用いたSFTを実行する方法を、実際のセットアップ経験をもとに詳しく解説します。

### この記事で学べること

- NeMo-RLの環境構築方法
- HuggingFaceデータセットの準備と変換
- 分散学習の設定（DTensor、Tensor Parallel）
- Slurmでのジョブ実行方法
- よくあるエラーとその解決策

### 対象読者

- LLMのファインチューニングを始めたい方
- Slurm環境でGPUクラスタを使用している方
- PyTorchの分散学習に興味がある方

---

## NeMo-RLとは

**NeMo-RL** は、NVIDIA NeMo Frameworkの一部として開発された、LLMのポストトレーニング（後処理学習）に特化したライブラリです。

### 主な特徴

| 特徴 | 説明 |
|------|------|
| **複数のアルゴリズム対応** | SFT、GRPO、DPO、On-policy Distillationなど |
| **スケーラブル** | 単一GPUからマルチノード・マルチGPUまで対応 |
| **Rayベースの分散処理** | 効率的なリソース管理と柔軟なデプロイメント |
| **HuggingFace統合** | 豊富な事前学習モデルをそのまま使用可能 |
| **高性能バックエンド** | DTensor（PyTorch native）とMegatron Coreの2つの学習バックエンドを選択可能 |

---

## 全体の流れ

SFTを実行するまでの全体像を把握しておきましょう。

```
┌─────────────────────────────────────────────────────────────┐
│  1. 環境構築                                                  │
│     ├── リポジトリのクローン                                    │
│     ├── サブモジュールの初期化                                  │
│     ├── 仮想環境の作成（uv）                                    │
│     └── 環境変数の設定                                         │
├─────────────────────────────────────────────────────────────┤
│  2. データ準備                                                 │
│     ├── HuggingFaceからデータセットをダウンロード                 │
│     └── JSONL形式に変換                                        │
├─────────────────────────────────────────────────────────────┤
│  3. 設定ファイル作成                                            │
│     ├── モデル設定                                             │
│     ├── データ設定                                             │
│     ├── 分散学習設定                                           │
│     └── ロギング設定                                           │
├─────────────────────────────────────────────────────────────┤
│  4. 学習実行                                                   │
│     ├── Slurmジョブスクリプト作成                                │
│     └── ジョブ投入・監視                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 環境構築

### 1.1 リポジトリのクローン

まず、NeMo-RLのリポジトリをクローンします。**重要なのは `--recursive` オプション**です。これにより、依存するサブモジュール（Megatron-LM等）も同時にクローンされます。

```bash
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl
```

#### すでにクローン済みの場合

`--recursive`を付けずにクローンしてしまった場合は、以下のコマンドでサブモジュールを初期化できます。

```bash
git submodule update --init --recursive
```

#### 確認方法

サブモジュールが正しく初期化されているか確認するには：

```bash
git submodule status
```

正常な場合、各サブモジュールのコミットハッシュの前にスペースが表示されます。`-`が表示されている場合は未初期化です。

```
# 正常な状態
 1d42deb98... 3rdparty/Automodel-workspace/Automodel

# 未初期化の状態（-が付いている）
-1d42deb98... 3rdparty/Automodel-workspace/Automodel
```

### 1.2 uvのインストール

NeMo-RLは、高速なPythonパッケージマネージャ **uv** を使用して依存関係を管理しています。uvはpipより10〜100倍高速で、環境の再現性も高いのが特徴です。

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# インストール確認
uv --version
```

### 1.3 仮想環境の作成

プロジェクトルートで以下を実行します。

```bash
uv venv
```

> **注意点**
> - `-p`や`--python`オプションは使用しないでください
> - NeMo-RLは`.python-version`ファイルでPythonバージョンを指定しています
> - uvがこのファイルを自動的に読み取り、適切なバージョンを使用します

### 1.4 環境変数の設定

学習を実行する前に、いくつかの環境変数を設定する必要があります。`~/.bashrc`に以下を追加してください。

```bash
# ==========================================
# NeMo-RL用環境変数
# ==========================================

# HuggingFaceのモデル・トークナイザーのキャッシュディレクトリ
# 大容量のストレージを指定することを推奨
export HF_HOME=/path/to/your/hf_cache

# HuggingFaceデータセットのキャッシュディレクトリ（任意）
export HF_DATASETS_CACHE=/path/to/your/datasets_cache

# WandB APIキー（学習曲線の可視化に使用）
export WANDB_API_KEY=your_wandb_api_key
```

設定を反映します。

```bash
source ~/.bashrc
```

### 1.5 HuggingFaceへのログイン（必要な場合）

Llama等のゲート付きモデルを使用する場合は、HuggingFaceにログインが必要です。

```bash
uv run huggingface-cli login
```

ブラウザでHuggingFaceにログインし、トークンを入力してください。

---

## 2. データセットの準備

### 2.1 NeMo-RLが期待するデータ形式

NeMo-RLのSFTは、**OpenAI形式のチャットフォーマット**を使用します。これは以下のような構造です。

```json
{
  "messages": [
    {"role": "system", "content": "あなたは数学の問題を解くアシスタントです。"},
    {"role": "user", "content": "2 + 3 を計算してください。"},
    {"role": "assistant", "content": "2 + 3 = 5 です。"}
  ]
}
```

#### 各ロールの説明

| ロール | 説明 |
|--------|------|
| `system` | モデルの振る舞いを指定するシステムプロンプト（任意） |
| `user` | ユーザーからの入力・質問 |
| `assistant` | モデルが生成すべき応答 |

### 2.2 HuggingFaceデータセットの変換

HuggingFaceにあるデータセットを直接使うことはできません。`openai_format`データセットタイプはローカルのJSONLファイルのみを受け付けるため、事前に変換が必要です。

#### 変換スクリプトの作成

`scripts/download_dataset.py`を作成します。

```python
#!/usr/bin/env python3
"""
HuggingFaceデータセットをNeMo-RL用のJSONL形式に変換するスクリプト

使い方:
    uv run python scripts/download_dataset.py
"""

import json
import os
from datasets import load_dataset


def main():
    # ============================================
    # 設定（ここを自分のデータセットに合わせて変更）
    # ============================================

    # HuggingFaceのデータセット名
    dataset_name = "your-organization/your-dataset-name"

    # 出力ディレクトリ
    output_dir = "data/your_dataset"

    # データセット内のメッセージカラム名（通常は"messages"）
    messages_column = "messages"

    # ============================================
    # 変換処理
    # ============================================

    print(f"データセットをダウンロード中: {dataset_name}")
    dataset = load_dataset(dataset_name)

    # 利用可能なスプリットを表示
    print(f"利用可能なスプリット: {list(dataset.keys())}")

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # trainスプリットを変換
    if "train" in dataset:
        train_path = os.path.join(output_dir, "train.jsonl")
        print(f"\ntrainスプリットを保存中: {train_path}")

        with open(train_path, "w", encoding="utf-8") as f:
            for item in dataset["train"]:
                # messagesカラムのみを抽出
                output = {"messages": item[messages_column]}
                f.write(json.dumps(output, ensure_ascii=False) + "\n")

        print(f"  -> {len(dataset['train']):,} サンプルを保存しました")

    # testまたはvalidationスプリットを変換
    val_split = "test" if "test" in dataset else "validation" if "validation" in dataset else None

    if val_split:
        val_path = os.path.join(output_dir, "test.jsonl")
        print(f"\n{val_split}スプリットを保存中: {val_path}")

        with open(val_path, "w", encoding="utf-8") as f:
            for item in dataset[val_split]:
                output = {"messages": item[messages_column]}
                f.write(json.dumps(output, ensure_ascii=False) + "\n")

        print(f"  -> {len(dataset[val_split]):,} サンプルを保存しました")

    print(f"\n完了！ファイルは {output_dir}/ に保存されました")


if __name__ == "__main__":
    main()
```

#### スクリプトの実行

```bash
# ディレクトリを作成
mkdir -p scripts

# スクリプトを実行（ログインノードで実行可能）
uv run python scripts/download_dataset.py
```

実行後、`data/your_dataset/`に以下のファイルが作成されます。

```
data/your_dataset/
├── train.jsonl    # 学習データ
└── test.jsonl     # 検証データ
```

---

## 3. 設定ファイルの作成

NeMo-RLはYAML形式の設定ファイルで学習パラメータを管理します。ここでは、各セクションを詳しく解説します。

### 3.1 設定ファイルの全体構造

`examples/configs/sft_custom.yaml`を作成します。

```yaml
# ============================================
# SFT（Supervised Fine-Tuning）設定ファイル
# ============================================

# --------------------------------------------
# 1. 学習アルゴリズムの設定
# --------------------------------------------
sft:
  # 学習エポック数
  # 1エポック = 全学習データを1回学習
  max_num_epochs: 1

  # 最大ステップ数
  # エポック数とステップ数の小さい方で学習が終了
  # フルエポック学習したい場合は大きな値を設定
  max_num_steps: 999999

  # 検証（validation）の設定
  val_period: 100          # 何ステップごとに検証を行うか
  val_batches: 16          # 検証時に使用するバッチ数
  val_global_batch_size: 32
  val_micro_batch_size: 1
  val_at_start: true       # 学習開始前に検証を行うか

  # 再現性のためのシード値
  seed: 42

# --------------------------------------------
# 2. チェックポイント（モデル保存）の設定
# --------------------------------------------
checkpointing:
  enabled: true

  # チェックポイントの保存先ディレクトリ
  checkpoint_dir: "results/sft_custom"

  # ベストモデル選択の基準
  metric_name: "val:val_loss"  # 検証ロスを使用
  higher_is_better: false      # ロスは低い方が良い

  # 保持するチェックポイント数（古いものから削除）
  keep_top_k: 3

  # 何ステップごとに保存するか
  save_period: 1000

# --------------------------------------------
# 3. モデル（Policy）の設定
# --------------------------------------------
policy:
  # HuggingFaceのモデル名またはローカルパス
  model_name: "your-organization/your-model"

  # トークナイザーの設定
  tokenizer:
    name: ${policy.model_name}  # モデルと同じトークナイザーを使用
    chat_template: "default"    # モデルのデフォルトテンプレートを使用
    chat_template_kwargs: null

  # バッチサイズの設定
  # global_batch_size = micro_batch_size × GPU数 × gradient_accumulation
  train_global_batch_size: 32
  train_micro_batch_size: 1    # GPU1枚あたりのバッチサイズ

  # シーケンス長（トークン数）
  # モデルの最大コンテキスト長以下に設定
  max_total_sequence_length: 8192

  # 計算精度
  # bfloat16: メモリ効率と精度のバランスが良い
  precision: "bfloat16"

  # オプティマイザのオフロード（通常はfalse）
  offload_optimizer_for_logprob: false

  # --------------------------------------------
  # 3.1 分散学習（DTensor）の設定
  # --------------------------------------------
  dtensor_cfg:
    # DTensorのバージョン
    # false: v1（安定版）、true: v2（新機能あり）
    _v2: false

    enabled: true

    # 追加の環境変数
    env_vars: {}

    # CPUオフロード（メモリ不足時に有効化）
    cpu_offload: false

    # Sequence Parallel
    # 注意: PyTorch 2.8/2.9ではTP>1で問題があるため無効化推奨
    sequence_parallel: false

    # Activation Checkpointing
    # メモリ使用量を削減（計算時間は増加）
    # 大きなモデルでは必須
    activation_checkpointing: true

    # Tensor Parallel Size
    # モデルをGPU間で分割する数
    # 通常はGPU数と同じに設定
    tensor_parallel_size: 8

    # Context Parallel Size（長文対応）
    context_parallel_size: 1

    custom_parallel_plan: null

    # LoRA設定（Parameter-Efficient Fine-Tuning）
    lora_cfg:
      enabled: false  # フルパラメータSFTの場合はfalse

  # --------------------------------------------
  # 3.2 Sequence Packing（効率化）
  # --------------------------------------------
  sequence_packing:
    # 複数の短いサンプルを1つのシーケンスにまとめる
    # スループットが大幅に向上
    enabled: true
    train_mb_tokens: ${mul:${policy.max_total_sequence_length}, ${policy.train_micro_batch_size}}
    algorithm: "modified_first_fit_decreasing"
    sequence_length_round: 64

  # Dynamic Batching（通常は無効）
  dynamic_batching:
    enabled: false

  # シーケンス長をTPサイズで割り切れるようにする
  make_sequence_length_divisible_by: ${policy.dtensor_cfg.tensor_parallel_size}

  # 勾配クリッピング
  max_grad_norm: 1.0

  # --------------------------------------------
  # 3.3 オプティマイザの設定
  # --------------------------------------------
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 2.0e-5           # 学習率
      weight_decay: 0.1    # 重み減衰（正則化）
      betas: [0.9, 0.98]   # Adam のモーメンタム係数
      eps: 1e-5
      foreach: false       # DTensorでは false に設定
      fused: false         # DTensorでは false に設定

  # Megatron Core（別の分散学習バックエンド）
  # DTensorを使う場合は無効化
  megatron_cfg:
    enabled: false
    env_vars: {}

# --------------------------------------------
# 4. データの設定
# --------------------------------------------
data:
  # 最大入力シーケンス長
  max_input_seq_length: ${policy.max_total_sequence_length}

  # 特殊トークンの追加
  add_bos: true   # Beginning of Sequence
  add_eos: true   # End of Sequence
  add_generation_prompt: false

  # データのシャッフル
  shuffle: true

  # データローダーのワーカー数
  num_workers: 4

  # データセットタイプ
  # "openai_format": OpenAI形式のJSONLファイル
  dataset_name: "openai_format"

  # データファイルのパス
  train_data_path: "data/your_dataset/train.jsonl"
  val_data_path: "data/your_dataset/test.jsonl"

  # JSONLファイル内のキー名
  chat_key: "messages"
  system_key: null
  system_prompt: null
  tool_key: null
  use_preserving_dataset: false

# --------------------------------------------
# 5. ロギングの設定
# --------------------------------------------
logger:
  log_dir: "logs"

  # WandB（Weights & Biases）
  wandb_enabled: true
  wandb:
    entity: "your-wandb-entity"   # WandBの組織名
    project: "your-project-name"  # プロジェクト名
    name: "sft-experiment-1"      # 実験名

  # TensorBoard
  tensorboard_enabled: true
  tensorboard:
    log_dir: "tb_logs"

  # その他のロガー
  mlflow_enabled: false
  swanlab_enabled: false

  # GPUモニタリング
  monitor_gpus: true
  gpu_monitoring:
    collection_interval: 10
    flush_interval: 10

# --------------------------------------------
# 6. クラスタの設定
# --------------------------------------------
cluster:
  gpus_per_node: 8   # ノードあたりのGPU数
  num_nodes: 1       # 使用するノード数
```

### 3.2 重要な設定項目の詳細解説

#### Tensor Parallel Size について

`tensor_parallel_size`は、モデルをGPU間で分割する数を指定します。

```
例: 9Bモデル、8GPU、tensor_parallel_size=8 の場合

GPU 0: モデルの1/8を担当
GPU 1: モデルの1/8を担当
...
GPU 7: モデルの1/8を担当
```

**設定の目安:**
- GPU数以下に設定
- 大きなモデルでは高い値を推奨
- 通常はGPU数と同じ値

#### Activation Checkpointing について

学習中の中間結果（Activation）を保存せず、必要時に再計算する技術です。

| 設定 | メモリ使用量 | 計算時間 |
|------|-------------|---------|
| `false` | 多い | 短い |
| `true` | 少ない | 長い（約20-30%増） |

大きなモデル（7B以上）では、メモリ不足を防ぐために`true`を推奨します。

#### Sequence Packing について

複数の短いサンプルを1つのシーケンスにまとめる技術です。

```
Packing無効の場合:
[サンプルA][PAD][PAD][PAD]  -> GPUの計算が無駄
[サンプルB][PAD][PAD][PAD]

Packing有効の場合:
[サンプルA][サンプルB][サンプルC]  -> 効率的
```

スループットが大幅に向上するため、有効化を推奨します。

---

## 4. Slurmジョブスクリプトの作成

### 4.1 基本的なジョブスクリプト

`run_sft_slurm.sh`を作成します。

```bash
#!/bin/bash

# ============================================
# Slurm ジョブ設定
# ============================================
#SBATCH --job-name=sft-custom          # ジョブ名
#SBATCH --partition=your_partition      # パーティション名（環境に合わせて変更）
#SBATCH --nodes=1                       # ノード数
#SBATCH --ntasks=1                      # タスク数
#SBATCH --cpus-per-task=200             # CPUコア数（データローダー用）
#SBATCH --gpus=8                        # GPU数
#SBATCH --output=logs/sft-%j.out        # 標準出力ログ（%jはジョブID）
#SBATCH --error=logs/sft-%j.err         # エラーログ

# エラー時にスクリプトを停止
set -e

# ============================================
# ジョブ情報の表示
# ============================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# ============================================
# 環境設定
# ============================================

# プロジェクトディレクトリに移動
cd /path/to/nemo-rl

# 仮想環境をアクティベート
source .venv/bin/activate

# 重要: 既存のPython環境を使用する設定
# この環境変数がないと、NeMo-RLが各コンポーネント用に
# 新しい仮想環境を作成しようとして時間がかかる
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# ============================================
# GPU状態の確認
# ============================================
echo ""
echo "GPU Status:"
nvidia-smi
echo ""

# ============================================
# 学習の実行
# ============================================
echo "Starting SFT training..."
python examples/run_sft.py \
    --config examples/configs/sft_custom.yaml

# ============================================
# 完了メッセージ
# ============================================
echo "=========================================="
echo "End time: $(date)"
echo "Training complete!"
echo "=========================================="
```

### 4.2 重要な環境変数

| 環境変数 | 説明 |
|----------|------|
| `NEMO_RL_PY_EXECUTABLES_SYSTEM=1` | 既存のPython環境を使用。設定しないとセットアップに時間がかかる |
| `CUDA_VISIBLE_DEVICES` | Slurmが自動設定。手動で設定しないこと |

### 4.3 コマンドラインでの設定オーバーライド

設定ファイルを変更せずに、コマンドラインでパラメータを変更できます。

```bash
python examples/run_sft.py \
    --config examples/configs/sft_custom.yaml \
    policy.train_global_batch_size=16 \
    sft.max_num_epochs=3 \
    logger.wandb.name="experiment-v2"
```

---

## 5. 学習の実行

### 5.1 事前準備

```bash
# ログディレクトリを作成
mkdir -p logs

# データが存在することを確認
ls -la data/your_dataset/
```

### 5.2 ジョブの投入

```bash
sbatch run_sft_slurm.sh
```

投入後、ジョブIDが表示されます。

```
Submitted batch job 12345
```

### 5.3 ジョブの監視

```bash
# ジョブの状態確認
squeue -u $USER

# 標準出力のリアルタイム監視
tail -f logs/sft-12345.out

# エラーログの監視
tail -f logs/sft-12345.err

# ジョブのキャンセル（必要な場合）
scancel 12345
```

### 5.4 WandBでの可視化

WandBを有効にしている場合、以下のURLでリアルタイムに学習曲線を確認できます。

```
https://wandb.ai/your-entity/your-project
```

確認できる主なメトリクス:
- `train/loss`: 学習ロス
- `val/val_loss`: 検証ロス
- `train/learning_rate`: 学習率
- `gpu/memory_used`: GPUメモリ使用量

---

## 6. トラブルシューティング

実際の環境構築で遭遇した問題とその解決策をまとめました。

### 6.1 Rayワーカーの登録失敗

**エラーメッセージ:**
```
Failed to register worker to Raylet: IOError: Failed to read data from the socket: End of file
```

**原因:**
NeMo-RLは分散処理にRayを使用していますが、Rayletが正常に起動できていない。

**解決策:**

1. **環境変数の設定**
```bash
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
```

2. **古いRayプロセスのクリーンアップ**
```bash
pkill -9 -f "ray::"
pkill -9 -f "raylet"
rm -rf /tmp/ray/*
```

3. **共有メモリの確認**
```bash
df -h /dev/shm
```
共有メモリが不足している場合は、システム管理者に相談してください。

### 6.2 データセットのパスエラー

**エラーメッセージ:**
```
FileNotFoundError: Unable to find 'your-org/your-dataset'
```

**原因:**
`openai_format`データセットはローカルファイルのみ対応。HuggingFaceのデータセット名を直接指定することはできない。

**解決策:**
HuggingFaceデータセットを事前にJSONLファイルに変換してください（2章参照）。

### 6.3 Sequence Parallelのエラー

**エラーメッセージ:**
PyTorch 2.8/2.9で`tensor_parallel_size > 1`の場合に発生

**解決策:**
設定ファイルで無効化:
```yaml
dtensor_cfg:
  sequence_parallel: false
```

### 6.4 Out of Memory (OOM)

**エラーメッセージ:**
```
CUDA out of memory
```

**解決策（優先度順）:**

1. **バッチサイズを減らす**
```yaml
policy:
  train_global_batch_size: 16  # 32から減らす
```

2. **Activation Checkpointingを有効化**
```yaml
dtensor_cfg:
  activation_checkpointing: true
```

3. **シーケンス長を減らす**
```yaml
policy:
  max_total_sequence_length: 4096  # 8192から減らす
```

4. **CPUオフロードを有効化**（最後の手段）
```yaml
dtensor_cfg:
  cpu_offload: true
```

### 6.5 DTensor v2のエラー

**症状:**
`_v2: true`で不安定な動作

**解決策:**
v1に戻す:
```yaml
dtensor_cfg:
  _v2: false
```

---

## 7. 学習完了後の操作

### 7.1 チェックポイントの確認

学習が完了すると、`checkpoint_dir`にチェックポイントが保存されます。

```bash
ls -la results/sft_custom/
```

```
results/sft_custom/
├── step_1000/
│   ├── config.yaml
│   └── policy/
│       └── weights/
├── step_2000/
│   └── ...
└── best/
    └── ...
```

### 7.2 HuggingFace形式への変換

評価や推論のために、チェックポイントをHuggingFace形式に変換できます。

```bash
uv run python examples/converters/convert_dcp_to_hf.py \
    --config results/sft_custom/step_2000/config.yaml \
    --dcp-ckpt-path results/sft_custom/step_2000/policy/weights/ \
    --hf-ckpt-path results/sft_custom/hf_model
```

### 7.3 モデルの評価

変換したモデルを評価:

```bash
uv run python examples/run_eval.py \
    generation.model_name=results/sft_custom/hf_model
```

---

## まとめ

本記事では、NeMo-RLを使ったSFTの環境構築から実行までを解説しました。

### 重要なポイントのおさらい

1. **環境構築**
   - `--recursive`でクローン
   - `uv venv`で仮想環境作成
   - 環境変数の設定（`HF_HOME`、`WANDB_API_KEY`）

2. **データ準備**
   - HuggingFaceデータセットをJSONLに変換
   - OpenAI形式（`messages`キー）で保存

3. **設定ファイル**
   - `tensor_parallel_size`: GPU数に合わせる
   - `activation_checkpointing`: 大きなモデルでは必須
   - `sequence_packing`: スループット向上

4. **Slurmジョブ**
   - `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`を設定
   - ログを監視して問題を早期発見

NeMo-RLは活発に開発が進んでいるプロジェクトです。最新の情報は公式ドキュメントやGitHubを参照してください。

---

## 参考リンク

- [NeMo-RL GitHub](https://github.com/NVIDIA-NeMo/RL)
- [NeMo-RL Documentation](https://docs.nvidia.com/nemo/rl/latest/index.html)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Weights & Biases](https://wandb.ai/)
- [HuggingFace Hub](https://huggingface.co/)

---

## 付録: 設定ファイルテンプレート

以下に、すぐに使える設定ファイルのテンプレートを用意しました。コピーして使用してください。

### 小規模モデル用（1-3B、1GPU）

```yaml
sft:
  max_num_epochs: 3
  max_num_steps: 999999
  val_period: 100
  seed: 42

policy:
  model_name: "your-model"
  train_global_batch_size: 8
  train_micro_batch_size: 8
  max_total_sequence_length: 4096
  precision: "bfloat16"

  dtensor_cfg:
    _v2: false
    enabled: true
    activation_checkpointing: false
    tensor_parallel_size: 1

cluster:
  gpus_per_node: 1
  num_nodes: 1
```

### 中規模モデル用（7-9B、8GPU）

```yaml
sft:
  max_num_epochs: 1
  max_num_steps: 999999
  val_period: 100
  seed: 42

policy:
  model_name: "your-model"
  train_global_batch_size: 32
  train_micro_batch_size: 1
  max_total_sequence_length: 8192
  precision: "bfloat16"

  dtensor_cfg:
    _v2: false
    enabled: true
    sequence_parallel: false
    activation_checkpointing: true
    tensor_parallel_size: 8

  sequence_packing:
    enabled: true

cluster:
  gpus_per_node: 8
  num_nodes: 1
```

### 大規模モデル用（30B+、マルチノード）

```yaml
sft:
  max_num_epochs: 1
  max_num_steps: 999999
  val_period: 50
  seed: 42

policy:
  model_name: "your-model"
  train_global_batch_size: 64
  train_micro_batch_size: 1
  max_total_sequence_length: 8192
  precision: "bfloat16"

  dtensor_cfg:
    _v2: false
    enabled: true
    sequence_parallel: false
    activation_checkpointing: true
    tensor_parallel_size: 8
    cpu_offload: false

  sequence_packing:
    enabled: true

cluster:
  gpus_per_node: 8
  num_nodes: 4  # 32 GPUs total
```
