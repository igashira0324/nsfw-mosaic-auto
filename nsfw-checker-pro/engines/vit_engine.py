# -*- coding: utf-8 -*-
"""
NSFW ViT Classifier Engine - EVA02-based 4-level NSFW severity classifier
Uses Freepik/nsfw_image_detector (2025, MIT license), which reports higher
accuracy than the previous Falconsai/nsfw_image_detection (2023, binary-only)
across all four severity levels (neutral/low/medium/high) — see model card:
https://huggingface.co/Freepik/nsfw_image_detector
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2
from PIL import Image

# Try to use transformers pipeline (best quality)
_HAS_TRANSFORMERS = False
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

# 4段階の重大度→0-1 NSFWスコアへの写像 (モデルカードのラベル順に対応)
SEVERITY_WEIGHTS = {'neutral': 0.0, 'low': 0.33, 'medium': 0.66, 'high': 1.0}


class ViTNSFWEngine:
    """EVA02ベース 4段階NSFW重大度分類エンジン"""

    NAME = "vit_nsfw"
    DISPLAY_NAME = "ViT NSFW Classifier (Freepik EVA02)"
    MODEL_ID = "Freepik/nsfw_image_detector"

    def __init__(self):
        self.available = False
        self.classifier = None

        if _HAS_TRANSFORMERS:
            try:
                # CUDAがあればGPUで実行 (旧実装はCPU固定だった)
                device = -1
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 0
                except ImportError:
                    pass
                print(f"[INFO] Loading {self.DISPLAY_NAME} ({self.MODEL_ID}) on {'GPU' if device == 0 else 'CPU'}...")
                self.classifier = pipeline(
                    "image-classification",
                    model=self.MODEL_ID,
                    device=device,
                )
                self.available = True
                print(f"[OK] {self.DISPLAY_NAME} initialized (transformers pipeline).")
            except Exception as e:
                print(f"[WARN] Failed to initialize {self.DISPLAY_NAME}: {e}")
        else:
            print(f"[WARN] {self.DISPLAY_NAME}: transformers package not installed. Skipping.")

    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Classify image NSFW severity (4-level).
        Returns: {'nsfw_score': float, 'label': str, 'severity': {neutral/low/medium/high: float}, 'engine': 'vit_nsfw'}

        nsfw_score は各重大度クラスの確率と SEVERITY_WEIGHTS の期待値
        (Σ p(class) * weight(class)) — 単純な argmax より滑らかで、
        「medium/highが僅差で割れている」ような曖昧な画像でも妥当な連続値になる。
        """
        if not self.available or self.classifier is None:
            return {'nsfw_score': 0.0, 'label': 'unknown', 'severity': {},
                    'engine': self.NAME, 'error': 'Not available'}

        try:
            # Convert BGR numpy array to RGB PIL Image
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            results = self.classifier(pil_img)
            severity = {r['label'].lower(): float(r['score']) for r in results}

            nsfw_score = sum(severity.get(k, 0.0) * w for k, w in SEVERITY_WEIGHTS.items())
            label = max(severity, key=severity.get) if severity else 'unknown'

            return {
                'nsfw_score': round(nsfw_score, 4),
                'label': label,
                'severity': {k: round(v, 4) for k, v in severity.items()},
                'engine': self.NAME
            }
        except Exception as e:
            return {'nsfw_score': 0.0, 'label': 'error', 'severity': {},
                    'engine': self.NAME, 'error': str(e)}
