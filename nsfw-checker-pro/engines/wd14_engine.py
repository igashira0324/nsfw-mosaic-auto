# -*- coding: utf-8 -*-
"""
WD14-Tagger V3 (EVA02-Large) Engine - High-precision clothing/tag classification
"""

import os
import cv2
import numpy as np
import urllib.request
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, Any
import onnxruntime as ort

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WD14_TAGGER_URL, WD14_TAGS_URL


class WD14Engine:
    """WD14-Tagger V3 タグ分類エンジン (Danbooru/アニメ調に強い)

    MODEL_URL/TAGS_URL/MODEL_FILENAME/TAGS_FILENAME はサブクラスで差し替え可能
    (例: PhotoTaggerEngine は同じ前処理・カテゴリ体系で実写向けモデルを使う)。
    """

    NAME = "wd14"
    DISPLAY_NAME = "WD14-Tagger V3"
    MODEL_URL = WD14_TAGGER_URL
    TAGS_URL = WD14_TAGS_URL
    MODEL_FILENAME = "wd_eva02_large_v3.onnx"
    TAGS_FILENAME = "wd_eva02_large_v3_tags.csv"
    INPUT_SIZE = 448

    def __init__(self):
        self.available = False
        self.model_path = Path.home() / ".gemini" / "models" / self.MODEL_FILENAME
        self.tags_path = Path.home() / ".gemini" / "models" / self.TAGS_FILENAME
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._ensure_model()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            except Exception:
                self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])

            self.input_name = self.session.get_inputs()[0].name
            self.tags_df = pd.read_csv(self.tags_path)
            self.tags = self.tags_df[self.tags_df['category'] == 0]['name'].tolist()
            self.tag_indices = self.tags_df[self.tags_df['category'] == 0].index.tolist()
            # category 9 = レーティング (general/sensitive/questionable/explicit)
            # 旧実装はこの最重要NSFW指標を捨てていた
            self.rating_indices = self.tags_df[self.tags_df['category'] == 9].index.tolist()
            self.available = True
            print(f"[OK] {self.DISPLAY_NAME} initialized.")
        except Exception as e:
            print(f"[WARN] Failed to initialize {self.DISPLAY_NAME}: {e}")

    def _ensure_model(self):
        if not self.model_path.exists():
            print(f"Downloading {self.DISPLAY_NAME} model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
        if not self.tags_path.exists():
            print(f"Downloading {self.DISPLAY_NAME} tags data...")
            urllib.request.urlretrieve(self.TAGS_URL, self.tags_path)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """WD14 preprocessing: aspect-ratio-preserving padding to a square, BGR order."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        w, h = pil_img.size
        size = self.INPUT_SIZE
        if w > h:
            new_w, new_h = size, int(h * (size / w))
        else:
            new_h, new_w = size, int(w * (size / h))

        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (size, size), (255, 255, 255))
        new_img.paste(pil_img, ((size - new_w) // 2, (size - new_h) // 2))

        # SmilingWolf WDタガー系は BGR 入力で学習されている (旧実装はRGBのまま渡していた)
        img_array = np.array(new_img)[:, :, ::-1].astype(np.float32)
        return np.expand_dims(img_array, axis=0)

    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image and return clothing/tag predictions + content rating.
        Returns: {'tags': {tag: score, ...}, 'rating': {general/sensitive/questionable/explicit: score}, 'engine': NAME}
        """
        if not self.available:
            return {'tags': {}, 'rating': {}, 'engine': self.NAME, 'error': 'Not available'}

        try:
            input_data = self._preprocess(image_array)
            outputs = self.session.run(None, {self.input_name: input_data})
            probs = outputs[0][0]

            result = {}
            for idx in self.tag_indices:
                if idx >= len(probs):
                    continue
                score = float(probs[idx])
                if score > 0.1:
                    tag = self.tags_df.iloc[idx]['name']
                    result[tag.replace('_', ' ')] = score

            rating = {}
            for idx in self.rating_indices:
                if idx >= len(probs):
                    continue
                rating[self.tags_df.iloc[idx]['name']] = float(probs[idx])

            return {'tags': result, 'rating': rating, 'engine': self.NAME}
        except Exception as e:
            return {'tags': {}, 'rating': {}, 'engine': self.NAME, 'error': str(e)}
