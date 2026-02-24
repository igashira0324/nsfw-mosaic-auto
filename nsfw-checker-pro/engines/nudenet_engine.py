# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

# NudeNet v3 labels
NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED"
]

def _add_cuda_to_path():
    """Add CUDA DLL directories for Windows environments."""
    if sys.platform != "win32":
        return
    possible = [
        Path(__file__).parent.parent.parent.parent / "bin",
        Path(__file__).parent.parent.parent.parent / "cudnn" / "bin",
        Path("E:\\Sync_Connect_Plus\\sd.webui\\bin"),
        Path("E:\\Sync_Connect_Plus\\sd.webui\\cudnn\\bin"),
    ]
    venv_torch = Path(os.path.dirname(sys.executable)) / "Lib" / "site-packages" / "torch" / "lib"
    if venv_torch.exists():
        possible.append(venv_torch)
    for p in possible:
        if p.exists():
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass

class NudeNetEngine:
    """NudeNet v3 検出エンジン (Custom ONNX Implementation)"""

    NAME = "nudenet"
    DISPLAY_NAME = "NudeNet v3"

    def __init__(self):
        self.session = None
        self.available = False
        _add_cuda_to_path()
        
        # Try to find the default model from the installed package
        self.model_path = self._find_package_model()
        if not self.model_path:
            self.model_path = Path.home() / ".gemini" / "models" / "320n.onnx"

        self._init_session()

    def _find_package_model(self) -> Optional[Path]:
        try:
            import nudenet
            pkg_path = Path(nudenet.__file__).parent / "320n.onnx"
            if pkg_path.exists():
                return pkg_path
        except:
            pass
        return None

    def _init_session(self):
        if onnxruntime is None or not self.model_path or not self.model_path.exists():
            print(f"[WARN] {self.DISPLAY_NAME}: Model or onnxruntime missing.")
            return

        try:
            # Enable GPU if possible
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(str(self.model_path), providers=providers)
            self.available = True
            print(f"[OK] {self.DISPLAY_NAME} initialized (Custom ONNX).")
        except Exception as e:
            print(f"[WARN] Failed to init {self.DISPLAY_NAME} ONNX: {e}")

    def _preprocess(self, img_bgr: np.ndarray, target_size: int = 320):
        h, w = img_bgr.shape[:2]
        max_size = max(h, w)
        
        # Padding to square (YOLOv8 style)
        pad_h = max_size - h
        pad_w = max_size - w
        img_pad = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        # BGR -> RGB and Resize
        img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_size, target_size))
        
        # Normalize to 0-1 and NCHW
        img_float = img_resized.astype(np.float32) / 255.0
        img_nchw = np.transpose(img_float, (2, 0, 1))
        img_nchw = np.expand_dims(img_nchw, axis=0)
        
        return img_nchw, max_size

    def analyze(self, image_array: np.ndarray, image_path: Path = None) -> Dict[str, Any]:
        if not self.available or self.session is None:
            return {'detections': [], 'engine': self.NAME}

        try:
            # Prefer passed array, but read if missing
            if image_array is None and image_path:
                # Handle Japanese paths
                with open(image_path, 'rb') as f:
                    data = f.read()
                image_array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            
            if image_array is None:
                return {'detections': [], 'engine': self.NAME}

            # NudeNet v3 (320n) expects 320x320
            target_res = 320
            blob, original_max_size = self._preprocess(image_array, target_res)
            
            # Inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: blob})
            
            # Post-processing
            detections = self._postprocess(outputs[0], original_max_size, target_res)
            return {'detections': detections, 'engine': self.NAME}
            
        except Exception as e:
            return {'detections': [], 'engine': self.NAME, 'error': str(e)}

    def _postprocess(self, output, original_max_size, model_res):
        # output shape: (1, 22, 2100) or similar
        out = np.squeeze(output)
        if out.ndim == 2:
            out = np.transpose(out) # (2100, 22)
        
        boxes = []
        confidences = []
        class_ids = []
        
        # YOLOv8 format: x, y, w, h, class0, class1, ...
        for i in range(out.shape[0]):
            classes_scores = out[i][4:]
            max_score = np.max(classes_scores)
            if max_score > 0.3: # Increased threshold for reliability
                class_id = np.argmax(classes_scores)
                cx, cy, w, h = out[i][:4]
                
                # Scale back to original coordinates (accounting for padding)
                scale = original_max_size / model_res
                abs_w = w * scale
                abs_h = h * scale
                abs_x = (cx - w/2) * scale
                abs_y = (cy - h/2) * scale
                
                boxes.append([int(abs_x), int(abs_y), int(abs_w), int(abs_h)])
                confidences.append(float(max_score))
                class_ids.append(class_id)
        
        if not boxes:
            return []

        # NMS to remove duplicates
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.45)
        
        results = []
        if len(indices) > 0:
            # Flatten indices if it's a list of lists (depending on OpenCV version)
            idx_list = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in idx_list:
                results.append({
                    'box': boxes[i],
                    'score': round(confidences[i], 3),
                    'label': NUDENET_LABELS[class_ids[i]]
                })
        
        return results
