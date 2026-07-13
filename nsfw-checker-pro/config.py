# -*- coding: utf-8 -*-
"""
nsfw-checker-pro - Configuration
統合型 NSFW チェッカー 設定ファイル
"""

# ============================================================
# Model URLs
# ============================================================
# 注: NudeNet 640m はGitHubの直接URLがログイン要求されるため、
# プロジェクトルートの download_models.bat (API経由) で取得する。
# エンジンは ../models/nudenet_640m.onnx を自動検出し、無ければ同梱320nを使う。
ANIME_MODEL_URL = "https://huggingface.co/deepghs/anime_real_cls/resolve/main/mobilenetv3_v1.4_dist/model.onnx?download=true"
WD14_TAGGER_URL = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx?download=true"
WD14_TAGS_URL = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv?download=true"

# 実写(グラビア/アイドル系)タガー。WD14 v3系と同一の前処理・カテゴリ体系
# (SmilingWolf作、deepghs org掲載) — アニメ調に偏るWD14の弱点である実写画像を補完する。
# https://huggingface.co/deepghs/idolsankaku-eva02-large-tagger-v1
PHOTO_TAGGER_URL = "https://huggingface.co/deepghs/idolsankaku-eva02-large-tagger-v1/resolve/main/model.onnx?download=true"
PHOTO_TAGGER_TAGS_URL = "https://huggingface.co/deepghs/idolsankaku-eva02-large-tagger-v1/resolve/main/selected_tags.csv?download=true"

# WD14 レーティングの寄与係数 (スコア = max(explicit*100, questionable*60, 裸タグ*100))
WD14_RATING_QUESTIONABLE_WEIGHT = 0.60

# ============================================================
# Google Cloud Vision API
# ============================================================
VISION_API_KEY = "YOUR_GOOGLE_CLOUD_VISION_API_KEY_HERE" # 機密情報のためプレースホルダーに置き換えました
VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"
VISION_API_TIMEOUT = 30

# SafeSearch likelihood mapping
LIKELIHOOD_SCORES = {
    'UNKNOWN': 0, 'VERY_UNLIKELY': 1, 'UNLIKELY': 2,
    'POSSIBLE': 3, 'LIKELY': 4, 'VERY_LIKELY': 5
}

VISION_CATEGORY_WEIGHTS = {
    'adult': 1.5, 'racy': 1.0, 'violence': 1.2,
    'medical': 0.5, 'spoof': 0.3
}

# ============================================================
# NudeNet Category Mappings
# ============================================================
CATEGORY_MAP = {
    'FEMALE_BREAST': ['FEMALE_BREAST_EXPOSED'],
    'GENITALIA': [
        'FEMALE_GENITALIA_EXPOSED', 'FEMALE_GENITALIA_COVERED',
        'MALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_COVERED'
    ],
    'BUTTOCKS': ['BUTTOCKS_EXPOSED', 'BUTTOCKS_COVERED'],
    'ANUS': ['ANUS_EXPOSED', 'ANUS_COVERED'],
    'OTHER_REGIONS': [
        'BELLY_EXPOSED', 'BELLY_COVERED',
        'FEET_EXPOSED', 'FEET_COVERED',
        'ARMPITS_EXPOSED', 'ARMPITS_COVERED'
    ],
    'FACE': ['FACE_FEMALE', 'FACE_MALE']
}

# ============================================================
# Scoring Thresholds
# ============================================================
THRESHOLDS = {
    'UNSAFE': 0.8, 'HIGH_RISK': 0.6,
    'MODERATE': 0.4, 'LOW_RISK': 0.2
}

VERDICT_ICONS = {
    'SAFE': '✅', 'LOW_RISK': '⚠️', 'MODERATE': '⚠️',
    'HIGH_RISK': '🔶', 'UNSAFE': '🔴', 'ERROR': '❌'
}

# ============================================================
# Style / Clothing Tags (WD14)
# ============================================================
STYLE_TAG_MAP = {
    '水着': [
        'swimsuit', 'bikini', 'one-piece swimsuit', 'school swimsuit',
        'competition swimsuit', 'sling bikini', 'micro bikini', 'front-tie bikini',
        'side-tie bikini', 'monokini', 'sukumizu', 'maillot', 'tankini',
        'bottomless swimsuit', 'collared swimsuit', 'striped swimsuit'
    ],
    '下着': [
        'underwear', 'bra', 'panties', 'lingerie', 'thong', 'undressing',
        'panties under leotard', 'bra visible', 'panties visible', 'lace-trimmed legwear'
    ],
    '制服': [
        'uniform', 'school uniform', 'serafuku', 'japanese school uniform', 'sailor uniform',
        'police uniform', 'nurse uniform', 'military uniform',
        'necktie', 'vest', 'blouse', 'shirt', 'ribbon', 'cardigan',
        'demon slayer uniform', 'haori'
    ],
    'メイド': ['maid', 'maid outfit', 'maid apron', 'maid uniform', 'maid headdress', 'apron'],
    'ドレス/ワンピ': ['dress', 'wedding dress', 'sundress', 'nightgown', 'evening dress', 'prom dress'],
    '和服': ['kimono', 'short kimono', 'yukata', 'haori', 'japanese clothes', 'obi', 'sash'],
    'スカート': ['skirt', 'miniskirt', 'micro skirt', 'pleated skirt', 'pencil skirt', 'high-waist skirt'],
    'ショートパンツ': ['shorts', 'short shorts', 'denim shorts', 'buruma', 'gym shorts'],
    'シャツ/トップス': [
        'shirt', 't-shirt', 'top', 'blouse', 'sweater', 'hoodie', 'tank top', 'camisole',
        'off-shoulder shirt', 'halter top'
    ],
    'ズボン/パンツ': ['pants', 'jeans', 'trousers', 'leggings', 'slacks'],
    '裸': [
        'nude', 'naked', 'topless', 'pussy', 'pubic hair', 'sex', 'hetero',
        'nipples', 'sex toy', 'dildo', 'bdsm', 'futanari', 'penis', 'uncensored', 'clitoris',
        'cum', 'cumdrip', 'bondage', 'masturbation', 'orgasm', 'ejaculation'
    ]
}

# ============================================================
# Engine Weights (for consensus scoring)
# ============================================================
ENGINE_WEIGHTS = {
    'nudenet': 0.25,
    'wd14': 0.20,
    'photo_tagger': 0.15,
    'vision_api': 0.15,
    'vit_nsfw': 0.10,
    'lfm_vl': 0.25,
    'anime_cls': 0.05
}

# ============================================================
# Supported Extensions
# ============================================================
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

# ============================================================
# UI Settings
# ============================================================
UI_THEME = "Dark"
UI_COLOR_THEME = "blue"

CATEGORY_SCORE_COLORS = {
    'BREAST': {'SAFE': '#2ecc71', 'LOW_RISK': '#f1c40f', 'MODERATE': '#f39c12', 'HIGH_RISK': '#e67e22', 'UNSAFE': '#e74c3c', 'ERROR': 'gray'},
    'GENITALIA': {'SAFE': '#2ecc71', 'LOW_RISK': '#f1c40f', 'MODERATE': '#f39c12', 'HIGH_RISK': '#e67e22', 'UNSAFE': '#e74c3c', 'ERROR': 'gray'},
    'ANUS': {'SAFE': '#2ecc71', 'LOW_RISK': '#f1c40f', 'MODERATE': '#f39c12', 'HIGH_RISK': '#e67e22', 'UNSAFE': '#e74c3c', 'ERROR': 'gray'},
    'BUTTOCKS': {'SAFE': '#2ecc71', 'LOW_RISK': '#f1c40f', 'MODERATE': '#f39c12', 'HIGH_RISK': '#e67e22', 'UNSAFE': '#e74c3c', 'ERROR': 'gray'}
}

STYLE_COLORS = {
    '裸': '#e74c3c', '下着': '#e67e22', '水着': '#f1c40f', 'その他': '#2ecc71'
}

ANIME_TAGS = {'anime', 'comic', 'manga', 'illustration', 'painting', 'sketch', 'drawing', '2d'}
REAL_TAGS = {'photorealistic', 'realistic', 'photo', 'real life', '3d'}
