# -*- coding: utf-8 -*-
"""
Integrated verification script for nsfw-checker-pro.
Tests all 6 engines and the scorer in a single flow.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.analyzer import MultiEngineAnalyzer
from core.scorer import Scorer
from config import THRESHOLDS

def verify():
    print("=" * 60)
    print("nsfw-checker-pro: Integrated Verification")
    print("=" * 60)

    # 1. Initialize Analyzer
    # Note: enable_vision depends on API key. enable_lfm will attempt download.
    try:
        analyzer = MultiEngineAnalyzer(enable_vision=True, enable_vit=True, enable_lfm=True)
    except Exception as e:
        print(f"[FAIL] Analyzer initialization error: {e}")
        return

    # 2. Check available engines
    available = analyzer.get_available_engines()
    print(f"\n[INFO] Engines found: {available}")
    expected = ['nudenet', 'wd14', 'anime_cls', 'vision_api', 'vit_nsfw', 'lfm_vl']
    for exp in expected:
        if exp not in available:
            print(f"[WARN] Expected engine '{exp}' is NOT available.")

    # 3. Load test image
    test_image_name = "..\\Gemini_Generated_Image_h3nz3oh3nz3oh3nz.png"
    test_path = Path(test_image_name)
    if not test_path.exists():
        print(f"[ERROR] Test image not found at {test_path}")
        return
    
    # 4. Run Analysis
    print(f"\n[TASK] Running full analysis on: {test_path.name}")
    try:
        raw_results = analyzer.analyze_image(test_path)
        print("[OK] Raw analysis completed.")
        
        # Check individual engine health
        for engine_name in expected:
            res = raw_results.get(engine_name, {})
            if 'error' in res:
                print(f"  [-] {engine_name:10s}: ERROR -> {res['error']}")
            elif 'detections' in res or 'tags' in res or 'score' in res or 'nsfw_score' in res:
                print(f"  [+] {engine_name:10s}: OK")
            else:
                print(f"  [?] {engine_name:10s}: Unexpected response format: {res}")
    except Exception as e:
        print(f"[FAIL] Analysis execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Run Scoring
    print("\n[TASK] Running scoring consensus...")
    try:
        scorer = Scorer()
        score_result = scorer.score(raw_results)
        
        print(f"\n[RESULT] Final Verdict: {score_result.verdict_icon} {score_result.verdict}")
        print(f"[RESULT] Total Score  : {score_result.total_score:.2f}")
        print(f"[RESULT] Primary Style: {score_result.primary_style}")
        print(f"[RESULT] Gender       : {score_result.gender}")
        print(f"[RESULT] Art Style    : {score_result.image_style}")
        print("\n[RESULT] Per-Engine Scores:")
        for eng, score in score_result.engine_scores.items():
            print(f"  - {eng:10s}: {score:6.2f}")
            
        if score_result.lfm_safety_level:
            print(f"\n[RESULT] LFM2.5 Summary: {score_result.lfm_safety_level} ({score_result.lfm_nsfw_score:.4f})")
            print(f"  Description: {score_result.lfm_description}")

    except Exception as e:
        print(f"[FAIL] Scoring failed: {e}")
        return

    print("\n" + "=" * 60)
    print("Verification Completed.")
    print("=" * 60)

if __name__ == "__main__":
    verify()
