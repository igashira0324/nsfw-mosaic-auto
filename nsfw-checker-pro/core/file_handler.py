# -*- coding: utf-8 -*-
"""
nsfw-checker-pro - File Handler
Handles file selection, filtering, and path management with Japanese path support.
"""

import os
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SUPPORTED_EXTENSIONS


class FileHandler:
    """ファイル操作ユーティリティ"""

    @staticmethod
    def get_images_from_folder(folder_path: str, recursive: bool = True) -> List[Path]:
        """Get all supported image files from a folder."""
        images = []
        folder = Path(folder_path)
        if not folder.exists():
            return images

        pattern = '**/*' if recursive else '*'
        for f in folder.glob(pattern):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(f)

        return sorted(images)

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if a file is a supported image format."""
        return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS

    @staticmethod
    def get_safe_filename(path: Path) -> str:
        """Get filename safe for display (handles Japanese characters)."""
        return path.name
