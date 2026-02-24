# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = np.ones((10, 10, 3), dtype=np.uint8) * 255
print(f"Original shape: {img.shape}")

try:
    res = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    print(f"Converted shape: {res.shape}")
    print(f"Value at [0,0]: {res[0,0]}")
except Exception as e:
    print(f"Error: {e}")

# Also test with 4 channels
img4 = np.ones((10, 10, 4), dtype=np.uint8) * 255
try:
    res4 = cv2.cvtColor(img4, cv2.COLOR_RGBA2BGR)
    print(f"RGBA -> BGR (4ch input) successful. Shape: {res4.shape}")
except Exception as e:
    print(f"RGBA -> BGR (4ch input) FAILED: {e}")
