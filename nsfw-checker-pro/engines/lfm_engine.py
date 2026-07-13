# -*- coding: utf-8 -*-
"""
LFM2.5-VL Engine - Vision Language Model NSFW Analyzer
Uses LiquidAI/LFM2.5-VL-1.6B for contextual NSFW content analysis.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    HAS_TRANSFORMERS = True
except ImportError as e:
    import traceback
    print(f"[DEBUG] ImportError details: {e}")
    traceback.print_exc()
    HAS_TRANSFORMERS = False


# Structured prompt for NSFW analysis
NSFW_ANALYSIS_PROMPT = """Analyze this image for content safety. Respond ONLY in the exact JSON format below, with no other text:
{"safety_level": "<SAFE|LOW_RISK|MODERATE|HIGH_RISK|UNSAFE>", "nsfw_score": <0.0-1.0>, "description": "<brief description>", "detected_elements": ["<element1>", "<element2>"]}

Classification guide:
- SAFE (0.0-0.2): Fully clothed, no suggestive content
- LOW_RISK (0.2-0.4): Slightly revealing clothing (swimwear, sports)
- MODERATE (0.4-0.6): Suggestive poses, lingerie, partial nudity
- HIGH_RISK (0.6-0.8): Significant nudity, explicit poses
- UNSAFE (0.8-1.0): Full nudity, explicit sexual content"""


class LFMEngine:
    """LFM2.5-VL-1.6B Vision Language Model Engine"""

    NAME = "lfm_vl"
    DISPLAY_NAME = "LFM2.5-VL"
    MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"

    def __init__(self):
        self.model = None
        self.processor = None
        self.available = False

        if not HAS_TRANSFORMERS:
            print(f"[WARN] {self.DISPLAY_NAME}: transformers package not installed.")
            return

        try:
            print(f"[INFO] Loading {self.DISPLAY_NAME} ({self.MODEL_ID})...")

            # Determine device and dtype
            if torch.cuda.is_available():
                self.device = "cuda"
                self.dtype = torch.bfloat16
                print(f"[INFO] {self.DISPLAY_NAME}: Using CUDA with bfloat16")
            else:
                self.device = "cpu"
                self.dtype = torch.float32
                print(f"[INFO] {self.DISPLAY_NAME}: Using CPU with float32")

            # device_map="auto" を使いつつ、低リソースロードを試みる
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID,
                device_map="auto",
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Processor のロードを追加
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True
            )
            
            self.available = True
            print(f"[OK] {self.DISPLAY_NAME} initialized.")
        except Exception as e:
            import traceback
            print(f"[WARN] Failed to initialize {self.DISPLAY_NAME}: {e}")
            traceback.print_exc()

    def analyze(self, image_array: np.ndarray = None, image_path: Path = None, custom_prompt: str = None) -> Dict[str, Any]:
        """
        Analyze image using LFM2.5-VL vision-language model.
        
        Returns:
            {
                'safety_level': str,
                'nsfw_score': float,
                'description': str,
                'detected_elements': list,
                'raw_response': str,
                'engine': str
            }
        """
        if not self.available:
            return {'error': 'Not available', 'engine': self.NAME}

        try:
            # Load image
            if image_path:
                pil_image = Image.open(image_path).convert("RGB")
            elif image_array is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                from PIL import Image as PILImage
                import cv2
                rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(rgb)
            else:
                return {'error': 'No image data', 'engine': self.NAME}

            # Build conversation
            prompt_to_use = custom_prompt if custom_prompt else NSFW_ANALYSIS_PROMPT
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt_to_use},
                    ],
                },
            ]

            # Generate
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model.device)

            # 生成パラメータ: 安全性グレーディング(構造化JSON)は公式モデルカード推奨値
            # (temperature=0.1, min_p=0.15, repetition_penalty=1.05) を使用。
            # repetition_penalty を高くするとJSON構造トークン("・:・,)まで抑制され
            # フォーマット破綻の原因になるため低めに保つ。
            # SNS創作文生成 (custom_prompt) は文末の単調な繰り返し防止が要件のため
            # repetition_penalty=1.2 の既存チューニングを維持する。
            if custom_prompt:
                gen_kwargs = dict(temperature=0.1, do_sample=True, repetition_penalty=1.2)
            else:
                gen_kwargs = dict(temperature=0.1, min_p=0.15, do_sample=True, repetition_penalty=1.05)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    **gen_kwargs,
                )

            # 重要: 生成トークンのみをデコードする。
            # 旧実装は入力プロンプトごとデコードしていたため、プロンプト内の
            # JSONテンプレートや "UNSAFE" 等のガイド語を回答として誤パースしていた。
            input_len = inputs["input_ids"].shape[1]
            gen_only = outputs[:, input_len:]
            raw_response = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

            # Parse the response
            if custom_prompt:
                parsed = self._extract_json(raw_response)
                result = parsed if parsed is not None else {'raw_text': raw_response}
            else:
                result = self._parse_response(raw_response)

            result['raw_response'] = raw_response
            result['engine'] = self.NAME
            return result

        except Exception as e:
            return {'error': str(e), 'engine': self.NAME}

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """テキスト中のJSONブロックを探し、最初に正しくパースできたものを返す。"""
        import json
        for m in re.finditer(r'\{.*?\}', text, re.DOTALL):
            try:
                data = json.loads(m.group())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract structured safety data."""
        default = {
            'safety_level': 'UNKNOWN',
            'nsfw_score': 0.0,
            'description': '',
            'detected_elements': []
        }

        data = self._extract_json(response)
        if data is not None and ('safety_level' in data or 'nsfw_score' in data):
            try:
                return {
                    'safety_level': str(data.get('safety_level', 'UNKNOWN')),
                    'nsfw_score': float(data.get('nsfw_score', 0.0)),
                    'description': str(data.get('description', '')),
                    'detected_elements': list(data.get('detected_elements', []) or [])
                }
            except (TypeError, ValueError):
                pass

        # Fallback: keyword-based parsing
        response_lower = response.lower()

        # Score estimation from keywords
        if any(w in response_lower for w in ['unsafe', 'explicit', 'full nudity', 'sexual']):
            default['nsfw_score'] = 0.9
            default['safety_level'] = 'UNSAFE'
        elif any(w in response_lower for w in ['high_risk', 'significant nudity', 'nude']):
            default['nsfw_score'] = 0.7
            default['safety_level'] = 'HIGH_RISK'
        elif any(w in response_lower for w in ['moderate', 'suggestive', 'lingerie', 'partial']):
            default['nsfw_score'] = 0.5
            default['safety_level'] = 'MODERATE'
        elif any(w in response_lower for w in ['low_risk', 'swimwear', 'bikini', 'revealing']):
            default['nsfw_score'] = 0.3
            default['safety_level'] = 'LOW_RISK'
        elif any(w in response_lower for w in ['safe', 'clothed', 'appropriate']):
            default['nsfw_score'] = 0.1
            default['safety_level'] = 'SAFE'

        default['description'] = response[:200]
        return default
