# Struct Eval コンペティション ルール詳細解説

## 1. コンペティションの概要

本コンペティションは，**Struct Eval** というベンチマークを一部改変した独自ベンチマークを用いて，LLM の**構造化された出力を生成する能力**を評価するものである．

Struct Eval は，LLM に対して「これを json 形式で出力して」「json 形式を yaml 形式に変換して」といった指示を与え，その出力が仕様通りの構造を持っているかどうかを検証するベンチマークである．

Struct Eval ベンチマークは大きく2つの評価セットで構成されている：

- **Struct Eval - T**：テキスト生成を測る
- **Struct Eval - V**：上記に加えて，レンダリングしたものの視覚的な評価も行う

本コンペでは **Struct Eval - T のみ**を対象として評価を行う．すなわち，LLM がコンピュータシステムやソフトウェア開発の AI エージェントワークフローにおいて，**エラーなく動作する正確な構造データを生成できるか**を試すものである．

---

## 2. タスクの種類と形式

### 2.1 タスクのパターン

タスクの形式は以下の2パターンである：

1. **TEXT to ○○（生成タスク）**：自然言語文の指示から，特定の形式を出力させる
2. **○○ to ××（変換タスク）**：特定の形式から，特定の形式へ変換させる

### 2.2 扱う形式

本コンペで扱う出力形式は以下の5種類である：

- JSON
- YAML
- TOML
- XML
- CSV

---

## 3. タスクの具体例：json 生成タスク

### 3.1 プロンプト例

以下のようなプロンプトが LLM に与えられる：

```
Please output JSON code.

Task:
Summarize metadata about a fictional
scientific article.

Feature Requirements:
1. Top-level field "title" is a string
2. Field "authors" is a list of exactly two items
3. Each author has "name" and "affiliation"
4. Field "publication.year" is an integer
5. Field "keywords" is a list of strings
```

### 3.2 期待される出力例

上記プロンプトを受けたモデルは，以下のような JSON 出力を生成することが期待される：

```json
{
  "title": "Temporal Anomalies in Sub-Atomic Particle Acceleration: A Case Study",
  "authors": [
    {
      "name": "Dr. Aris Thorne",
      "affiliation": "Institute of Theoretical Chronophysics, Zurich"
    },
    {
      "name": "Prof. Elena Voshkova",
      "affiliation": "Department of High Energy Physics, MIT"
    }
  ],
  "publication": {
    "year": 2028
  },
  "keywords": [
    "Chronophysics",
    "Particle Acceleration",
    "Temporal Dilution",
    "Quantum Mechanics"
  ]
}
```

---

## 4. 採点基準（raw_output_metric）

### 4.1 採点基準の構造

各タスクに対して，JSON の仕様に沿っているかを判定するための**採点基準がタスク毎に定義**されている．上記の例では，以下のような `raw_output_metric` が設定される：

```
"raw_output_metric": {
    "title",
    "authors[0].name",
    "authors[1].affiliation",
    "publication.year",
    "keywords[2]"
}
```

### 4.2 各基準の意味

| 基準キー | 意味 |
|---|---|
| `title` | トップレベルに `title` キーが存在するか |
| `authors[0].name` | `authors` 配列の 0 番目に `name` キーが存在するか |
| `authors[1].affiliation` | `authors` 配列の 1 番目に `affiliation` キーが存在するか |
| `publication.year` | ネストされた `publication` の中に `year` キーが存在するか |
| `keywords[2]` | `keywords` 配列に 3 つ目の要素が存在するか |

**重要：入っている値の内容は何でも良い．**  
採点は「指定されたキーパスが存在するかどうか」のみで行われ，値の正確さは問われない．

---

## 5. 評価データと推論結果の形式

### 5.1 評価データの一例

実際の評価データは以下のような JSON 形式で提供される：

```json
[
  {
    "task_id": "000500",
    "query": "Please output JSON code:\n\nTask:\n...",
    "feature_requirements": "",
    "task_name": "Text to JSON",
    "input_type": "Text",
    "output_type": "JSON",
    "query_example": "",
    "VQA": [],
    "raw_output_metric": [
      "novel.title",
      "novel.author.name",
      "novel.characters[0].name"
    ],
    "rendering": false
  }
]
```

ここで `raw_output_metric` が採点基準の部分にあたる．

### 5.2 推論結果の一例

LLM による推論結果は `"generation"` フィールドとして付与される：

```json
[
  {
    "task_id": "000500",
    "query": "Please output JSON code:\n\nTask:\n...",
    "feature_requirements": "",
    "task_name": "Text to JSON",
    "input_type": "Text",
    "output_type": "JSON",
    "query_example": "",
    "VQA": [],
    "raw_output_metric": [...],
    "rendering": false,
    "generation": "```json\n{\n  \"novel\": {\n    \"title\": \"The Obsidian Labyrinth\",\n  \"author..."
  }
]
```

### 5.3 評価方法

この生成結果に対して，以下の2点で評価が行われる：

1. **パース可否**：生成結果が指定された形式（この例では JSON）として正しくパースできるか
2. **基準の充足度**：`raw_output_metric` で指定された各キーパスが，パースされた構造内にどれだけ存在するか

---

## 6. 本コンペ独自の工夫（オリジナルとの相違点）

本コンペでは StructEval-T のオリジナルの評価データや評価方式をそのまま使用せず，以下の工夫を加えている：

### 6.1 採点基準の秘匿

- `raw_output_metric` を**評価データから分からないようにしている**
- 評価システム（Omni キャンバス）に `task_id` と紐づけて埋め込んでいる

### 6.2 採点基準の変更

- 各タスクの `raw_output_metric` 自体を**オリジナルから変更**している（採点基準を変えている）

### 6.3 評価方法の変更

- 評価方法・アルゴリズムも概ねオリジナルと同じだが，**コンペ用に一部変更**を加えている

### 6.4 総合点の算出

- コンペで使用する総合点については，**形式毎の評価点を平均**し，**独自の重みづけ**を行って算出している

### 6.5 目的

これらの工夫により，コンペ参加者が評価データの採点基準に特化した学習を行うことができないようになっている．

> ※このため，オリジナルの Struct Eval を回した場合と採点結果に多少差異が出る．  
> ※評価の詳細は非公開．

---

## 7. 運営からの配布物（6.5）

配布時期はメインコンペスタート時となる．

### 7.1 評価用データ

- ファイル名：`public_150.json`
- **学習に使用してはならない**
- **改変も行ってはならない．人手，ツール，LLM，一切の手入れを禁じる**

### 7.2 標準コード

#### モデルの学習コード

- ファイル名：
  - `2026最終課題メインコンペ_標準コード1（SFT）.ipynb`
  - `2026最終課題メインコンペ_標準コード3（DPO）.ipynb`
- **改変自由**
- HuggingFace の WRITE 権限が必要
- HuggingFace README のサンプルを含んでいる（参考にして記載すること）

#### モデルの推論コード

- ファイル名：`2026最終課題メインコンペ_標準コード2（提出JSON生成）.ipynb`
- **モデルの情報以外を一切改変せず利用すること**
- **推論時に動く RAG や ToolUse や外部サービスとの連携の類はすべて禁止**
- このコードで提出物 json ファイル（`inference.json`）が作成される

### 7.3 学習用データ

- 運営作成の合成データ
- その他，ライセンス OK なデータセットを紹介予定
- ※これらはスコアが確実に上がることを保証するものではない

---

## 8. モデル関連ルール（6.6）

### 8.1 学習指定モデル

本コンペにおける学習指定モデルは以下の2つ**のみ**である：

- `Qwen/Qwen3-4B-Instruct-2507`
- `unsloth/Qwen3-4B-Instruct-2507`

**これ以外のモデルの使用は認めない．派生モデルも禁止．**

### 8.2 モデルの制約

| ルール | 詳細 |
|---|---|
| アーキテクチャ変更 | 不可 |
| 何かしらの変更 | 必須（SFT，RLHF，DPO，量子化など） |
| アップロード先 | Hugging Face にモデルとしてアップロード可能であること（WRITE 権限必須） |

### 8.3 禁止事項

- **StructEval のデータをモデル開発のあらゆる段階で使用することを禁止**
  - 運営が提供する評価データはもちろん，**オリジナルの StructEval のデータを用いることも禁止**
- **リーダーボードを利用したチューニングを禁止**

---

## 9. 学習データ関連ルール（6.6）

### 9.1 使用可能なデータ

**運営が提供，または紹介する学習データ以外の使用は禁止．** 運営が提供する，または紹介するデータの範囲で学習を行うこと．

### 9.2 LLM を用いたデータ作成は禁止

- 運営が提供，または紹介するデータを **LLM を用いて改変したり，合成することは禁止**

### 9.3 LLM を用いないデータ作成は可

- 運営が提供，または紹介するデータを **LLM を用いずに改変することは可能**
- 詳細は補足資料「LLM モデル開発におけるお約束」および「LLM によるデータ作成」を参照

---

## 10. 学習のヒント

本コンペでは，「指定された出力形式」で「正しい構造を安定して生成する能力」を評価する．したがって，以下の点を意識して取り組むことが推奨される：

1. **構文を壊さない**：JSON，YAML，TOML，XML，CSV の各形式の文法を正確に守る
2. **要求されたキーを落とさない**：プロンプトで指定された構造要素を漏れなく含める
3. **余計な文章を出さない**：コードだけを出力するのが安全（自然言語の説明文などを混ぜるとパースに失敗する可能性がある）

---

## 11. ルールまとめ（一覧）

| カテゴリ | ルール | 詳細 |
|---|---|---|
| モデル | 使用モデル | Qwen3-4B-Instruct-2507（Qwen 公式 or unsloth 版）のみ |
| モデル | アーキテクチャ | 変更不可 |
| モデル | パラメータ変更 | 必須（SFT / RLHF / DPO / 量子化等） |
| モデル | 提出先 | Hugging Face（WRITE 権限必要） |
| データ | 学習データ | 運営提供 or 紹介のもののみ |
| データ | LLM によるデータ改変・合成 | 禁止 |
| データ | 手作業によるデータ改変 | 可 |
| データ | 評価データの学習利用 | 禁止 |
| データ | 評価データの改変 | 禁止 |
| データ | StructEval データの利用 | 全面禁止（オリジナル含む） |
| 推論 | 推論コードの改変 | モデル情報以外は禁止 |
| 推論 | RAG / ToolUse / 外部連携 | 禁止 |
| その他 | リーダーボード利用チューニング | 禁止 |