# -*- coding: utf-8 -*-
"""
NSFW Image Checker - Vision API Client
Google Cloud Vision API クライアント
"""

import base64
import requests
from pathlib import Path
from typing import Dict, Optional, Any

from config import VISION_API_URL, API_TIMEOUT, API_KEY


class VisionAPIError(Exception):
    """Vision API関連のエラー"""
    pass


class VisionClient:
    """Google Cloud Vision API クライアント"""
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: Google Cloud Vision API キー (省略時は config.API_KEY を使用)
        """
        self.api_key = api_key or API_KEY
        self.endpoint = f"{VISION_API_URL}?key={self.api_key}"
    
    def _encode_image(self, image_path: Path) -> str:
        """画像をBase64エンコード"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _build_request(self, image_base64: str) -> Dict[str, Any]:
        """APIリクエストボディを構築"""
        return {
            "requests": [
                {
                    "image": {
                        "content": image_base64
                    },
                    "features": [
                        {
                            "type": "SAFE_SEARCH_DETECTION"
                        },
                        {
                            "type": "LABEL_DETECTION",
                            "maxResults": 10
                        }
                    ]
                }
            ]
        }
    
    def detect_safe_search(self, image_path: Path) -> Dict[str, str]:
        """
        画像のSafeSearch判定を実行
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            SafeSearch結果の辞書
            {
                'adult': 'UNLIKELY',
                'racy': 'POSSIBLE',
                'violence': 'VERY_UNLIKELY',
                'medical': 'UNLIKELY',
                'spoof': 'VERY_UNLIKELY'
            }
            
        Raises:
            VisionAPIError: API呼び出しに失敗した場合
        """
        try:
            # 画像をBase64エンコード
            image_base64 = self._encode_image(image_path)
            
            # リクエスト送信
            response = requests.post(
                self.endpoint,
                json=self._build_request(image_base64),
                timeout=API_TIMEOUT
            )
            
            # レスポンスチェック
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise VisionAPIError(f"API Error ({response.status_code}): {error_msg}")
            
            # 結果を解析
            result = response.json()
            
            if 'responses' not in result or len(result['responses']) == 0:
                raise VisionAPIError("Empty response from API")
            
            safe_search = result['responses'][0].get('safeSearchAnnotation', {})
            
            if not safe_search:
                raise VisionAPIError("No SafeSearch annotation in response")
            
            return {
                'adult': safe_search.get('adult', 'UNKNOWN'),
                'racy': safe_search.get('racy', 'UNKNOWN'),
                'violence': safe_search.get('violence', 'UNKNOWN'),
                'medical': safe_search.get('medical', 'UNKNOWN'),
                'spoof': safe_search.get('spoof', 'UNKNOWN')
            }
            
        except requests.exceptions.Timeout:
            raise VisionAPIError(f"API request timed out after {API_TIMEOUT} seconds")
        except requests.exceptions.RequestException as e:
            raise VisionAPIError(f"Network error: {str(e)}")
        except Exception as e:
            if isinstance(e, VisionAPIError):
                raise
            raise VisionAPIError(f"Unexpected error: {str(e)}")

    def detect_labels(self, image_path: Path) -> Dict[str, float]:
        """
        画像のラベル検出を実行
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            ラベルの辞書（スコア付き）
            {
                'label1': confidence_score,
                'label2': confidence_score,
                ...
            }
            
        Raises:
            VisionAPIError: API呼び出しに失敗した場合
        """
        try:
            # 画像をBase64エンコード
            image_base64 = self._encode_image(image_path)
            
            # リクエスト送信
            response = requests.post(
                self.endpoint,
                json=self._build_request(image_base64),
                timeout=API_TIMEOUT
            )
            
            # レスポンスチェック
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise VisionAPIError(f"API Error ({response.status_code}): {error_msg}")
            
            # 結果を解析
            result = response.json()
            
            if 'responses' not in result or len(result['responses']) == 0:
                raise VisionAPIError("Empty response from API")
            
            labels = result['responses'][0].get('labelAnnotations', [])
            
            if not labels:
                return {}
            
            # ラベルを辞書形式で返す
            return {
                label['description']: label['score']
                for label in labels
            }
            
        except requests.exceptions.Timeout:
            raise VisionAPIError(f"API request timed out after {API_TIMEOUT} seconds")
        except requests.exceptions.RequestException as e:
            raise VisionAPIError(f"Network error: {str(e)}")
        except Exception as e:
            if isinstance(e, VisionAPIError):
                raise
            raise VisionAPIError(f"Unexpected error: {str(e)}")

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        画像を包括的に分析し、SafeSearch判定とラベル検出をまとめて実行する
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            画像分析結果の辞書
            {
                'safe_search': {
                    'adult': 'UNLIKELY',
                    'racy': 'POSSIBLE',
                    'violence': 'VERY_UNLIKELY',
                    'medical': 'UNLIKELY',
                    'spoof': 'VERY_UNLIKELY'
                },
                'description': 'cat, animal, pet',
                'labels': {
                    'cat': 0.95,
                    'animal': 0.90,
                    'pet': 0.85
                }
            }
            
        Raises:
            VisionAPIError: API呼び出しに失敗した場合
        """
        try:
            # 画像をBase64エンコード
            image_base64 = self._encode_image(image_path)
            
            # リクエスト送信
            response = requests.post(
                self.endpoint,
                json=self._build_request(image_base64),
                timeout=API_TIMEOUT
            )
            
            # レスポンスチェック
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise VisionAPIError(f"API Error ({response.status_code}): {error_msg}")
            
            # 結果を解析
            result = response.json()
            
            if 'responses' not in result or len(result['responses']) == 0:
                raise VisionAPIError("Empty response from API")
            
            response_data = result['responses'][0]
            
            # SafeSearch結果を処理
            safe_search_annotation = response_data.get('safeSearchAnnotation', {})
            safe_search = {
                'adult': safe_search_annotation.get('adult', 'UNKNOWN'),
                'racy': safe_search_annotation.get('racy', 'UNKNOWN'),
                'violence': safe_search_annotation.get('violence', 'UNKNOWN'),
                'medical': safe_search_annotation.get('medical', 'UNKNOWN'),
                'spoof': safe_search_annotation.get('spoof', 'UNKNOWN')
            }
            
            # ラベル検出結果を処理
            labels_annotation = response_data.get('labelAnnotations', [])
            labels = {}
            for label in labels_annotation:
                labels[label['description']] = label['score']
            
            # 上位5つのラベルを説明文として結合
            top_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:5]
            description = ', '.join([label for label, score in top_labels])
            
            return {
                'safe_search': safe_search,
                'description': description,
                'labels': labels
            }
            
        except requests.exceptions.Timeout:
            raise VisionAPIError(f"API request timed out after {API_TIMEOUT} seconds")
        except requests.exceptions.RequestException as e:
            raise VisionAPIError(f"Network error: {str(e)}")
        except Exception as e:
            if isinstance(e, VisionAPIError):
                raise
            raise VisionAPIError(f"Unexpected error: {str(e)}")
