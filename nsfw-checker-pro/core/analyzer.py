# -*- coding: utf-8 -*-
"""
nsfw-checker-pro - Multi-Engine Analyzer
Orchestrates all detection engines and produces unified analysis results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.nudenet_engine import NudeNetEngine
from engines.wd14_engine import WD14Engine
from engines.anime_engine import AnimeEngine
from engines.vision_engine import VisionEngine
from engines.vit_engine import ViTNSFWEngine
from engines.lfm_engine import LFMEngine


class MultiEngineAnalyzer:
    """マルチエンジン統合アナライザ"""

    def __init__(self, enable_vision: bool = True, enable_vit: bool = True, enable_lfm: bool = True):
        """
        Initialize all available engines.

        Args:
            enable_vision: Enable Google Cloud Vision API engine
            enable_vit: Enable ViT NSFW Classifier engine
            enable_lfm: Enable LFM2.5-VL Vision Language Model engine
        """
        print("=" * 60)
        print("nsfw-checker-pro: Initializing engines...")
        print("=" * 60)

        self.engines = {}

        # Always try to initialize core engines
        try:
            self.engines['nudenet'] = NudeNetEngine()
        except Exception as e:
            print(f"[ERROR] NudeNet init failed: {e}")

        try:
            self.engines['wd14'] = WD14Engine()
        except Exception as e:
            print(f"[ERROR] WD14 init failed: {e}")

        try:
            self.engines['anime_cls'] = AnimeEngine()
        except Exception as e:
            print(f"[ERROR] AnimeEngine init failed: {e}")

        # Optional engines
        if enable_vision:
            try:
                self.engines['vision_api'] = VisionEngine()
            except Exception as e:
                print(f"[ERROR] VisionEngine init failed: {e}")

        if enable_vit:
            try:
                self.engines['vit_nsfw'] = ViTNSFWEngine()
            except Exception as e:
                print(f"[ERROR] ViTNSFWEngine init failed: {e}")

        if enable_lfm:
            try:
                self.engines['lfm_vl'] = LFMEngine()
            except Exception as e:
                print(f"[ERROR] LFMEngine init failed: {e}")

        # Summary
        available = [name for name, eng in self.engines.items() if eng.available]
        print(f"\n[INFO] Available engines: {', '.join(available)} ({len(available)}/{len(self.engines)})")
        print("=" * 60)

    def get_available_engines(self) -> List[str]:
        """Return list of available engine names."""
        return [name for name, eng in self.engines.items() if eng.available]

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analyze a single image with all available engines.

        Returns:
            {
                'path': str,
                'nudenet': {...},
                'wd14': {...},
                'anime_cls': {...},
                'vision_api': {...},
                'vit_nsfw': {...}
            }
        """
        result = {'path': str(image_path)}

        # Read image once (shared by all local engines)
        image_array = None
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            result['error'] = f"Failed to read image: {e}"
            return result

        if image_array is None:
            result['error'] = f"Failed to decode image: {image_path}"
            return result

        # Run each engine
        for name, engine in self.engines.items():
            if not engine.available:
                result[name] = {'error': 'Not available'}
                continue

            try:
                if name in ('vision_api', 'nudenet', 'lfm_vl'):
                    # These engines benefit from file path access
                    result[name] = engine.analyze(image_array, image_path=image_path)
                else:
                    result[name] = engine.analyze(image_array)
            except Exception as e:
                result[name] = {'error': str(e)}

        return result
