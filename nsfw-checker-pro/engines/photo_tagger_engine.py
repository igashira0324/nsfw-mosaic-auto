# -*- coding: utf-8 -*-
"""
Photo Tagger Engine (idolsankaku-eva02-large-tagger-v1) - Real-photo tag classification

WD14-Tagger V3 はDanbooru学習でアニメ調画像に強いが実写(グラビア/アイドル系)の
タグ精度が落ちる。idolsankakuタガーは同じ作者(SmilingWolf)によるWD v3系と
互換の前処理・カテゴリ体系を持つ実写特化モデルのため、WD14Engineをそのまま
継承しモデル/タグファイルだけ差し替える。
https://huggingface.co/deepghs/idolsankaku-eva02-large-tagger-v1
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PHOTO_TAGGER_URL, PHOTO_TAGGER_TAGS_URL
from engines.wd14_engine import WD14Engine


class PhotoTaggerEngine(WD14Engine):
    """実写(グラビア/アイドル系)タグ分類エンジン"""

    NAME = "photo_tagger"
    DISPLAY_NAME = "Photo Tagger (idolsankaku)"
    MODEL_URL = PHOTO_TAGGER_URL
    TAGS_URL = PHOTO_TAGGER_TAGS_URL
    MODEL_FILENAME = "idolsankaku_eva02_large_v1.onnx"
    TAGS_FILENAME = "idolsankaku_eva02_large_v1_tags.csv"
    # eva02_large_patch14_448 アーキテクチャだが実際の学習解像度は480px
    # (config.json: model_args.img_size=480 / pretrained_cfg.input_size=[3,480,480])
    INPUT_SIZE = 480
