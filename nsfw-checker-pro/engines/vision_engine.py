# -*- coding: utf-8 -*-
"""
Google Cloud Vision API Engine - SafeSearch + Label Detection
"""

import base64
import requests
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VISION_API_KEY, VISION_API_URL, VISION_API_TIMEOUT, LIKELIHOOD_SCORES, VISION_CATEGORY_WEIGHTS


class VisionEngine:
    """Google Cloud Vision API SafeSearch エンジン"""

    NAME = "vision_api"
    DISPLAY_NAME = "Google Vision API"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or VISION_API_KEY
        self.available = bool(self.api_key)
        if self.available:
            print(f"[OK] {self.DISPLAY_NAME} configured (API key present).")
        else:
            print(f"[WARN] {self.DISPLAY_NAME}: No API key. Skipping.")

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _build_request(self, image_base64: str) -> dict:
        return {
            "requests": [{
                "image": {"content": image_base64},
                "features": [
                    {"type": "SAFE_SEARCH_DETECTION"},
                    {"type": "LABEL_DETECTION", "maxResults": 10}
                ]
            }]
        }

    def analyze_from_path(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze image via Vision API using file path.
        Returns: {'safe_search': {...}, 'labels': {...}, 'score': float, 'engine': 'vision_api'}
        """
        if not self.available:
            return {'safe_search': {}, 'labels': {}, 'score': 0.0, 'engine': self.NAME, 'error': 'No API key'}

        try:
            image_base64 = self._encode_image(image_path)
            body = self._build_request(image_base64)
            url = f"{VISION_API_URL}?key={self.api_key}"

            response = requests.post(url, json=body, timeout=VISION_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            result_block = data.get('responses', [{}])[0]

            # SafeSearch
            ss = result_block.get('safeSearchAnnotation', {})
            safe_search = {
                'adult': ss.get('adult', 'UNKNOWN'),
                'racy': ss.get('racy', 'UNKNOWN'),
                'violence': ss.get('violence', 'UNKNOWN'),
                'medical': ss.get('medical', 'UNKNOWN'),
                'spoof': ss.get('spoof', 'UNKNOWN')
            }

            # Score calculation
            total_weight = sum(VISION_CATEGORY_WEIGHTS.values())
            weighted_sum = 0
            for cat, rating_str in safe_search.items():
                raw = LIKELIHOOD_SCORES.get(rating_str, 0)
                w = VISION_CATEGORY_WEIGHTS.get(cat, 1.0)
                weighted_sum += (raw / 5.0) * w
            score = (weighted_sum / total_weight) * 100.0

            # Labels
            labels = {}
            for label_ann in result_block.get('labelAnnotations', []):
                labels[label_ann.get('description', '')] = label_ann.get('score', 0.0)

            return {
                'safe_search': safe_search,
                'labels': labels,
                'score': round(score, 2),
                'engine': self.NAME
            }
        except Exception as e:
            return {'safe_search': {}, 'labels': {}, 'score': 0.0, 'engine': self.NAME, 'error': str(e)}

    def analyze(self, image_array=None, image_path: Path = None) -> Dict[str, Any]:
        """Unified interface. Vision API needs file path for base64 encoding."""
        if image_path:
            return self.analyze_from_path(image_path)
        return {'safe_search': {}, 'labels': {}, 'score': 0.0, 'engine': self.NAME, 'error': 'Path required'}
