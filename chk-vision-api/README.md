# NSFW Image Checker Pro

Google Cloud Vision API の SafeSearch Detection を使用した画像NSFWチェックツールです。

## 特徴

- �️ **GUI インターフェース**: 直感的なウィンドウ操作で画像やフォルダを選択
- 📊 **客観的スコアリング**: カテゴリ別重み付けによる総合スコア（0-100）
- 🔍 **詳細レポート**: Adult, Racy, Violence 等の全5カテゴリの詳細値を表示
- 🏷️ **5段階判定**: SAFE / LOW_RISK / MODERATE / HIGH_RISK / UNSAFE
- 💾 **エクスポート**: 結果を CSV または JSON 形式で保存可能
- 🔑 **APIキー埋め込み**: コード内にキーを保持して簡単に起動可能

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. Google Cloud Vision API キーの設定

1. [Google Cloud Console](https://console.cloud.google.com/) で APIキーを作成（Cloud Vision API を有効化してください）
2. [config.py](file:///e:/Sync_Connect_Plus/sd.webui/system/mosaic_20230513/chk-vision-api/config.py) を開き、`API_KEY` 変数に取得したキーを貼り付けます。

```python
# config.py
API_KEY = "あなたのAPIキーをここに貼り付け"
```

## 使用方法

### GUI モードで起動（推奨）

引数なしで実行すると GUI が立ち上がります。

```bash
python main.py
```

1. **Select File(s)** または **Select Folder** で画像を追加します。
2. **Start Analysis** をクリックして分析を開始します。
3. 結果がテーブルに表示されます。
4. **Export Results** で結果を保存できます。
5. **Reference Sheet** でスコアリングの基準を確認できます。

### CLI モードで起動

特定のファイルやフォルダをコマンドラインから即座にチェックしたい場合に使用します。

```bash
# フォルダ内の全画像をチェック
python main.py ./images --recursive

# スコア40以上のみ抽出して保存
python main.py ./images --threshold 40 --output risky.json
```

## スコアリング詳細

詳しい判定基準や重み付けについては [REFERENCE_SHEET.md](file:///e:/Sync_Connect_Plus/sd.webui/system/mosaic_20230513/chk-vision-api/REFERENCE_SHEET.md) を参照してください。

## 注意事項

- Cloud Vision API は有料サービスです（月1000リクエストまで無料枠があります）。
- APIキーが記述されたファイルを共有したり、公開リポジトリにアップロードしないよう注意してください。
