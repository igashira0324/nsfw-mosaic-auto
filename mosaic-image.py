# -*- coding: utf-8 -*-
"""
mosaic-image.py — NSFW自動モザイク (画像) 2026-07 刷新版

改善点:
  * 審査基準準拠モザイク (セル = max(4px, 長辺/100)) — 旧版の固定8/16/32pxはFHD以上で基準不足だった
  * 検出アンサンブル: EraX V1.0 + Anti-NSFW V1.1 + NudeNet 640m (取りこぼし削減)
  * TTA (Test-Time Augmentation) + 高解像度画像のタイル推論 (小さな対象の検出向上)
  * MobileSAM による箱→マスク精密モザイク (過剰な矩形モザイクを回避)
  * アニメGIF全フレーム処理 / EXIF回転反映 / ICC・透過保持 / WebP・BMP対応
  * 一時JPEG書き出し廃止 (ndarray直接推論)
使い方:
  python mosaic-image.py [画像フォルダ]   # 引数省略時はフォルダ選択ダイアログ
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import mosaic_core as mc
import mosaic_gui as mg


def main() -> None:
    # --- 入力フォルダ ---
    if len(sys.argv) == 1:
        folder = mg.pick_folder("画像フォルダを選択してください")
        if not folder:
            print("フォルダが選択されませんでした。処理を中止します。")
            sys.exit(1)
    elif len(sys.argv) == 2:
        folder = sys.argv[1]
    else:
        print("使い方: python mosaic-image.py <画像フォルダ>")
        sys.exit(1)

    if not os.path.isdir(folder):
        print("指定されたパスはフォルダではありません")
        sys.exit(1)

    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(mc.IMAGE_EXTS)]
    if not files:
        mg.show_info("画像なし", "対応画像ファイルが見つかりません。\n対応形式: " + " ".join(mc.IMAGE_EXTS))
        sys.exit(1)

    # --- パターン選択 ---
    pattern = mg.ask_mosaic_pattern()
    if pattern is None:
        print("キャンセルされました。処理を中止します。")
        return

    # --- 検出エンジン初期化 ---
    cfg = mc.load_config()
    print("[INFO] 検出エンジンを初期化しています...")
    try:
        detector = mc.NsfwDetector(cfg)
    except Exception as e:
        mg.show_error("エラー", f"検出モデルの読み込みに失敗しました。\n{e}")
        return

    out_folder = folder + "_mc"
    os.makedirs(out_folder, exist_ok=True)

    pw = mg.ProgressWindow("画像モザイク処理")
    t0 = time.time()
    n_done = 0
    n_regions_total = 0
    errors = []

    for idx, fname in enumerate(files, 1):
        if pw.cancelled:
            break
        pw.update("画像モザイク処理", idx - 1, len(files), extra=fname)
        in_path = os.path.join(folder, fname)
        out_path = os.path.join(out_folder, fname)
        try:
            n = mc.process_image_file(in_path, out_path, pattern, cfg, detector)
            n_regions_total += n
            n_done += 1
            print(f"[{idx}/{len(files)}] {fname}: {n}領域")
        except Exception as e:
            errors.append(f"{fname}: {e}")
            print(f"[ERROR] {fname}: {e}")
        pw.update("画像モザイク処理", idx, len(files), extra=fname)

    cancelled = pw.cancelled
    pw.close()

    elapsed = time.time() - t0
    msg = (f"処理完了: {n_done}/{len(files)} ファイル\n"
           f"モザイク適用領域: {n_regions_total}箇所\n"
           f"所要時間: {elapsed:.1f}秒\n"
           f"出力先: {out_folder}")
    if cancelled:
        msg = "キャンセルされました。\n" + msg
    if errors:
        msg += "\n\nエラー:\n" + "\n".join(errors[:10])
        if len(errors) > 10:
            msg += f"\n...他{len(errors) - 10}件"
    mg.show_info("完了", msg)


if __name__ == "__main__":
    main()
