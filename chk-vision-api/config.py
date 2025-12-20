# -*- coding: utf-8 -*-
"""
NSFW Image Checker - Configuration
設定ファイル
"""

# Google Cloud Vision API キー (ここに貼り付けてください)
API_KEY = ""

# 対応する画像拡張子
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

# SafeSearch likelihood 値のスコアマッピング
LIKELIHOOD_SCORES = {
    'UNKNOWN': 0,
    'VERY_UNLIKELY': 1,
    'UNLIKELY': 2,
    'POSSIBLE': 3,
    'LIKELY': 4,
    'VERY_LIKELY': 5
}

# カテゴリ別重み係数
CATEGORY_WEIGHTS = {
    'adult': 1.5,      # 成人向けコンテンツ
    'racy': 1.0,       # きわどい/挑発的
    'violence': 1.2,   # 暴力的コンテンツ
    'medical': 0.5,    # 医療的画像
    'spoof': 0.3       # パロディ/なりすまし
}

# 判定閾値
VERDICT_THRESHOLDS = {
    'SAFE': (0.0, 20.0),
    'LOW_RISK': (20.01, 40.0),
    'MODERATE': (40.01, 60.0),
    'HIGH_RISK': (60.01, 80.0),
    'UNSAFE': (80.01, 100.0)
}

# 判定結果の表示アイコン
VERDICT_ICONS = {
    'SAFE': '✅',
    'LOW_RISK': '⚠️',
    'MODERATE': '⚠️',
    'HIGH_RISK': '🔶',
    'UNSAFE': '🔴'
}

# Vision API エンドポイント
VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate'

# API リクエストのタイムアウト（秒）
API_TIMEOUT = 30

# バッチ処理時の待機時間（秒）- レート制限対策
BATCH_DELAY = 0.1
