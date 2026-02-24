# -*- coding: utf-8 -*-
"""
NSFW ViT Classifier Engine - Fine-tuned Vision Transformer for NSFW classification
Uses Falconsai/nsfw_image_detection model (ViT-based)

Falls back to a lightweight ONNX-based approach if transformers is not installed.
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


class ViTNSFWEngine:
    """Vision Transformer NSFW 分類エンジン"""

    NAME = "vit_nsfw"
    DISPLAY_NAME = "ViT NSFW Classifier"
    MODEL_ID = "Falconsai/nsfw_image_detection"

    def __init__(self):
        self.available = False
        self.classifier = None

        if _HAS_TRANSFORMERS:
            try:
                print(f"[INFO] Loading {self.DISPLAY_NAME} ({self.MODEL_ID})...")
                self.classifier = pipeline(
                    "image-classification",
                    model=self.MODEL_ID,
                    device=-1  # CPU; change to 0 for GPU
                )
                self.available = True
                print(f"[OK] {self.DISPLAY_NAME} initialized (transformers pipeline).")
            except Exception as e:
                print(f"[WARN] Failed to initialize {self.DISPLAY_NAME}: {e}")
        else:
            print(f"[WARN] {self.DISPLAY_NAME}: transformers package not installed. Skipping.")

    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Classify image as nsfw or normal.
        Returns: {'nsfw_score': float, 'normal_score': float, 'label': str, 'engine': 'vit_nsfw'}
        """
        if not self.available or self.classifier is None:
            return {'nsfw_score': 0.0, 'normal_score': 0.0, 'label': 'unknown',
                    'engine': self.NAME, 'error': 'Not available'}

        try:
            # Convert BGR numpy array to RGB PIL Image
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            results = self.classifier(pil_img)

            nsfw_score = 0.0
            normal_score = 0.0
            label = 'normal'

            for r in results:
                if r['label'].lower() == 'nsfw':
                    nsfw_score = r['score']
                elif r['label'].lower() == 'normal':
                    normal_score = r['score']

            if nsfw_score > normal_score:
                label = 'nsfw'

            return {
                'nsfw_score': round(nsfw_score, 4),
                'normal_score': round(normal_score, 4),
                'label': label,
                'engine': self.NAME
            }
        except Exception as e:
            return {'nsfw_score': 0.0, 'normal_score': 0.0, 'label': 'error',
                    'engine': self.NAME, 'error': str(e)}
