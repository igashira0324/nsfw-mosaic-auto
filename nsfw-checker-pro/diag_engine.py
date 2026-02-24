# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.nudenet_engine import NudeNetEngine

def test_engine_on_file(file_path):
    print(f"Testing NudeNetEngine on: {file_path}")
    engine = NudeNetEngine()
    if not engine.available:
        print("Engine not available!")
        return

    # Read image as the app does
    with open(file_path, 'rb') as f:
        data = f.read()
    image_array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    
    if image_array is None:
        print("Failed to decode image.")
        return

    print("Running analysis...")
    result = engine.analyze(image_array)
    print("Result:", result)

if __name__ == "__main__":
    # Use any image file in the parent dir for testing
    import glob
    images = glob.glob("../*.jpg") + glob.glob("../*.png")
    if images:
        test_engine_on_file(images[0])
    else:
        print("No test images found in parent directory.")
