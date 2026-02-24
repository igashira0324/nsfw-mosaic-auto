# -*- coding: utf-8 -*-
"""
Anime/Real Classifier Engine
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2
import onnxruntime as ort

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANIME_MODEL_URL


class AnimeEngine:
    """Anime/Real 分類エンジン"""

    NAME = "anime_cls"
    DISPLAY_NAME = "Anime/Real Classifier"

    def __init__(self):
        self.available = False
        model_dir = os.path.join(os.path.expanduser("~"), ".nudenet_classifier")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "anime_real_cls.onnx")

        try:
            if not os.path.exists(self.model_path):
                print(f"Downloading anime classifier model...")
                urllib.request.urlretrieve(ANIME_MODEL_URL, self.model_path)

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
            except Exception:
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

            self.input_name = self.session.get_inputs()[0].name
            self.available = True
            print(f"[OK] {self.DISPLAY_NAME} initialized.")
        except Exception as e:
            print(f"[WARN] Failed to initialize {self.DISPLAY_NAME}: {e}")

    def analyze(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Classify image as anime or real.
        Returns: {'style': {'anime': float, 'real': float}, 'engine': 'anime_cls'}
        """
        if not self.available:
            return {'style': {'anime': 0.0, 'real': 0.0}, 'engine': self.NAME, 'error': 'Not available'}

        try:
            img = cv2.resize(image_array, (384, 384))
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            outputs = self.session.run(None, {self.input_name: img})
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            return {
                'style': {'anime': float(probs[0]), 'real': float(probs[1])},
                'engine': self.NAME
            }
        except Exception as e:
            return {'style': {'anime': 0.0, 'real': 0.0}, 'engine': self.NAME, 'error': str(e)}
