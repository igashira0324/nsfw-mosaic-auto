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
    """WD14-Tagger V3 タグ分類エンジン"""

    NAME = "wd14"
    DISPLAY_NAME = "WD14-Tagger V3"

    def __init__(self):
        self.available = False
        self.model_path = Path.home() / ".gemini" / "models" / "wd_eva02_large_v3.onnx"
        self.tags_path = Path.home() / ".gemini" / "models" / "wd_eva02_large_v3_tags.csv"
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
            self.available = True
            print(f"[OK] {self.DISPLAY_NAME} initialized.")
        except Exception as e:
            print(f"[WARN] Failed to initialize {self.DISPLAY_NAME}: {e}")

    def _ensure_model(self):
        if not self.model_path.exists():
            print(f"Downloading WD14-Tagger V3 model (~1.3GB)...")
            urllib.request.urlretrieve(WD14_TAGGER_URL, self.model_path)
        if not self.tags_path.exists():
            print(f"Downloading WD14 tags data...")
            urllib.request.urlretrieve(WD14_TAGS_URL, self.tags_path)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """WD14 preprocessing: 448x448 with aspect-ratio-preserving padding."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        w, h = pil_img.size
        size = 448
        if w > h:
            new_w, new_h = size, int(h * (size / w))
        else:
            new_h, new_w = size, int(w * (size / h))

        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (size, size), (255, 255, 255))
        new_img.paste(pil_img, ((size - new_w) // 2, (size - new_h) // 2))

        img_array = np.array(new_img).astype(np.float32)
        return np.expand_dims(img_array, axis=0)

    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image and return clothing/tag predictions.
        Returns: {'tags': {tag: score, ...}, 'engine': 'wd14'}
        """
        if not self.available:
            return {'tags': {}, 'engine': self.NAME, 'error': 'Not available'}

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

            return {'tags': result, 'engine': self.NAME}
        except Exception as e:
            return {'tags': {}, 'engine': self.NAME, 'error': str(e)}
