# dataset_vlm_captions

Ollama の VLM（Vision Language Model）を使って、画像からキャプションを生成し、同名の `.txt` ファイルとして保存するツールです。

- 単一画像ファイル、または画像フォルダをまとめて処理できます
- `--target` で指定した対象語（例: `face`）を、プロンプト上で除外対象として扱います

----

## サンプル

### 入力画像

<img src="sample/sample.png" width="320" alt="sample image">

### 実行例

```bash
python caption_tool.py -d "sample/sample.png" -t "face"
```

### 出力キャプション例

A person with long, dark, loose hair is seated at an outdoor wooden table, angled slightly away from the camera but with their head turned to face forward. They are wearing a fitted, rust-colored or terracotta ribbed knit long-sleeved top. Their left elbow is resting on the edge of the wooden table in the immediate foreground, and their right hand is positioned near their lap or the edge of their seat. In the background, there is a narrow cobblestone street lined with beige, European-style buildings featuring arched windows and balconies. Several large terracotta planters containing green leafy bushes are placed along the walkway. Other individuals are visible in the background, seated at various tables or walking along the street, appearing out of focus due to the depth of field. The lighting is bright and diffuse, suggesting a daytime setting.


---

## セットアップ

1. 依存ライブラリをインストール

```bash
pip install requests
```

2. Ollama を起動（デフォルト: `http://localhost:11434`）

3. 必要に応じて環境変数を設定

- `OLLAMA_BASE_URL`（例: `http://localhost:11434`）
- `OLLAMA_VLM_MODEL`（例: `qwen3.5:9b`）

---

## 使い方

### 単一画像を処理

```bash
python caption_tool.py -d "C:\path\to\image.png" -t "face"
```

### フォルダ内の画像を一括処理

```bash
python caption_tool.py -d "C:\path\to\images" -t "face"
```

### 主なオプション

- `-d, --dir` : 画像ファイルまたは画像フォルダのパス（必須）
- `-t, --target` : キャプションから除外したい学習対象（必須）
- `-m, --model` : 使用する Ollama VLM モデル名（省略時は設定値）

生成されたキャプションは、入力画像と同じ場所に同名 `.txt` で保存されます。

---


## 補足

- 対応拡張子: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- モデル応答が空の場合は `.txt` を保存せずスキップします
