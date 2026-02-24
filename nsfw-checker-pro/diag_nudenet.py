# -*- coding: utf-8 -*-
import cv2
import numpy as np
import nudenet
import os

def test_nudenet():
    print(f"NudeNet version: {nudenet.__version__ if hasattr(nudenet, '__version__') else 'unknown'}")
    try:
        detector = nudenet.NudeDetector()
        print("Detector initialized.")
        
        # Test with 3-channel black image (BGR)
        img3 = np.zeros((640, 640, 3), dtype=np.uint8)
        print("Testing with 3-channel array...")
        try:
            res3 = detector.detect(img3)
            print(f"3-channel detection successful: {res3}")
        except Exception as e:
            print(f"3-channel detection FAILED: {e}")
            
        # Test with 4-channel black image (RGBA)
        img4 = np.zeros((640, 640, 4), dtype=np.uint8)
        print("Testing with 4-channel array...")
        try:
            res4 = detector.detect(img4)
            print(f"4-channel detection successful: {res4}")
        except Exception as e:
            print(f"4-channel detection FAILED: {e}")
            
        # Test with a path (if we can find one)
        # For now, let's just create a dummy file
        dummy_path = "dummy_test.jpg"
        cv2.imwrite(dummy_path, img3)
        print(f"Testing with image path: {dummy_path}")
        try:
            res_path = detector.detect(dummy_path)
            print(f"Path detection successful: {res_path}")
        except Exception as e:
            print(f"Path detection FAILED: {e}")
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
                
    except Exception as e:
        print(f"Initialization/Test error: {e}")

if __name__ == "__main__":
    test_nudenet()
