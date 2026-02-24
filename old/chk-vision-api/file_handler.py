# -*- coding: utf-8 -*-
"""
NSFW Image Checker - File Handler
ファイル/フォルダ処理
"""

from pathlib import Path
from typing import List, Generator

from config import SUPPORTED_EXTENSIONS


class FileHandler:
    """画像ファイルの検索・処理"""
    
    def __init__(self, extensions: set = None):
        """
        Args:
            extensions: 対応する画像拡張子のセット
        """
        self.extensions = extensions or SUPPORTED_EXTENSIONS
    
    def is_image(self, path: Path) -> bool:
        """画像ファイルかどうかを判定"""
        return path.suffix.lower() in self.extensions
    
    def get_images(self, path: Path, recursive: bool = False) -> Generator[Path, None, None]:
        """
        指定パスから画像ファイルを取得
        
        Args:
            path: ファイルまたはディレクトリのパス
            recursive: サブディレクトリも検索するか
            
        Yields:
            画像ファイルのPath
        """
        if path.is_file():
            if self.is_image(path):
                yield path
        elif path.is_dir():
            pattern = '**/*' if recursive else '*'
            for file_path in path.glob(pattern):
                if file_path.is_file() and self.is_image(file_path):
                    yield file_path
    
    def collect_images(self, path: Path, recursive: bool = False) -> List[Path]:
        """
        指定パスから画像ファイルをリストとして取得
        
        Args:
            path: ファイルまたはディレクトリのパス
            recursive: サブディレクトリも検索するか
            
        Returns:
            画像ファイルのPathリスト（ソート済み）
        """
        images = list(self.get_images(path, recursive))
        return sorted(images)
    
    def validate_path(self, path_str: str) -> Path:
        """
        パス文字列を検証してPathオブジェクトを返す
        
        Args:
            path_str: パス文字列
            
        Returns:
            検証済みのPath
            
        Raises:
            FileNotFoundError: パスが存在しない場合
        """
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path_str}")
        return path
