# -*- coding: utf-8 -*-
"""
download_models.py — 高品質化用の追加AIモデルを一括取得する。

取得対象 (models/ 配下):
  1. erax-anti-nsfw-yolo11m-v1.1.pt  … EraX Anti-NSFW V1.1 (検出アンサンブル用, HuggingFace)
  2. nudenet_640m.onnx               … NudeNet 640m 高精度版 (GitHub Releases)
                                        ※直接URLはログイン要求されるため API 経由で取得
  3. mobile_sam.pt                   … MobileSAM (静止画のマスク精密モザイク用)

いずれも取得失敗してもツールは動作する (該当機能が自動的に無効/代替になる)。
"""

import json
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _http_get(url: str, headers: dict = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "mosaic-tool"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download_erax_v11() -> None:
    dst = os.path.join(MODELS_DIR, "erax-anti-nsfw-yolo11m-v1.1.pt")
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
        print(f"[SKIP] {os.path.basename(dst)} は取得済み")
        return
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="erax-ai/EraX-Anti-NSFW-V1.1",
            filename="erax-anti-nsfw-yolo11m-v1.1.pt",
            local_dir=MODELS_DIR,
        )
        print("[OK] EraX Anti-NSFW V1.1 取得完了")
    except Exception as e:
        print(f"[FAIL] EraX V1.1: {e}")


def download_nudenet_640m() -> None:
    dst = os.path.join(MODELS_DIR, "nudenet_640m.onnx")
    if os.path.exists(dst) and os.path.getsize(dst) > 50_000_000:
        print(f"[SKIP] {os.path.basename(dst)} は取得済み")
        return
    try:
        # 直接URLはGitHubログインへリダイレクトされるため、API asset endpoint を使用
        rel = json.loads(_http_get(
            "https://api.github.com/repos/notAI-tech/NudeNet/releases/tags/v3.4-weights",
            headers={"User-Agent": "mosaic-tool", "Accept": "application/vnd.github+json"},
        ))
        asset_id = next(a["id"] for a in rel["assets"] if a["name"] == "640m.onnx")
        data = _http_get(
            f"https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/{asset_id}",
            headers={"User-Agent": "mosaic-tool", "Accept": "application/octet-stream"},
        )
        if len(data) < 50_000_000:
            raise RuntimeError(f"サイズ異常 ({len(data)}B) — ダウンロード失敗の可能性")
        with open(dst, "wb") as f:
            f.write(data)
        print(f"[OK] NudeNet 640m 取得完了 ({len(data) // 1_000_000}MB)")
    except Exception as e:
        print(f"[FAIL] NudeNet 640m (同梱320nで動作継続可能): {e}")


def download_mobile_sam() -> None:
    dst = os.path.join(MODELS_DIR, "mobile_sam.pt")
    if os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
        print(f"[SKIP] {os.path.basename(dst)} は取得済み")
        return
    try:
        cwd = os.getcwd()
        os.chdir(MODELS_DIR)
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset("mobile_sam.pt")
        finally:
            os.chdir(cwd)
        print("[OK] MobileSAM 取得完了")
    except Exception as e:
        print(f"[FAIL] MobileSAM (矩形モザイクで動作継続可能): {e}")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=== 追加モデル一括ダウンロード ===")
    download_erax_v11()
    download_nudenet_640m()
    download_mobile_sam()
    print("=== 完了 ===")
