# -*- coding: utf-8 -*-
"""
mosaic-video.py — NSFW自動モザイク (動画) 2026-07 刷新版

改善点:
  * 2パス方式による時間安定化:
      Pass1: BoT-SORTトラッキング検出 (track_buffer=60 / 全検出即トラック化) + シーンカット検出
      後処理: 検出ギャップ線形補間 / リードイン・アウト延長 / 移動平均平滑化 (被覆は和集合保証)
      Pass2: モザイク適用 + ffmpegパイプ一発エンコード
    → 旧版の「同一モデル二重推論 + 15フレーム保持ハック」を置き換え、モザイクの
      ちらつき・出現遅れ・消え残りを解消しつつ推論回数を半減
  * 審査基準準拠モザイク (セル = max(4px, 長辺/100)、グリッドは画像原点に整列)
  * エンコード品質: 旧版 mp4v→H.264 の二重再エンコードを廃止し、libx264 CRF18 一発出力
  * NudeNetクロスチェックの box形式バグ (xywh→xyxy誤解釈) を修正、ndarray直接入力で高速化
  * 再スキャン検証: モザイク漏れフレームの検出レポート + ワンクリック修正
使い方:
  python mosaic-video.py            # GUIで選択
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def collect_videos() -> list:
    """モード選択→動画パス一覧を返す。"""
    mode = mg.ask_video_mode()
    if mode == "file":
        path = mg.pick_file(
            "動画ファイルを選択してください",
            [("動画ファイル", "*.mp4;*.avi;*.mov;*.mkv;*.webm"), ("All files", "*.*")],
        )
        return [path] if path else []
    if mode == "folder":
        folder = mg.pick_folder("動画フォルダを選択してください")
        if not folder:
            return []
        vids = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(VIDEO_EXTS) and "_mc." not in f.lower()
        ]
        if not vids:
            mg.show_info("動画なし", "選択フォルダに対応動画がありません。")
        return vids
    return []


def process_all(video_paths: list, pattern: str, cfg: dict, detector) -> list:
    """各動画を2パス処理する。戻り値: [(out_path, stats), ...]"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for i, video_path in enumerate(video_paths, 1):
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(OUTPUT_DIR, name + "_mc" + mc.out_container_for(video_path))

        pw = mg.ProgressWindow(f"動画モザイク処理 ({i}/{len(video_paths)}): {os.path.basename(video_path)}")
        stage_names = {"detect": "Pass1: AI検出パス", "encode": "Pass2: モザイク適用+エンコード"}

        def on_progress(stage, cur, total):
            pw.update(stage_names.get(stage, stage), cur, total,
                      extra=os.path.basename(video_path))

        try:
            stats = mc.process_video(
                video_path, out_path, pattern, cfg, detector,
                keep_audio=True, progress=on_progress,
                cancel=lambda: pw.cancelled,
            )
        except Exception as e:
            pw.close()
            mg.show_error("エラー", f"{os.path.basename(video_path)} の処理に失敗しました。\n{e}")
            continue
        pw.close()
        if stats.get("cancelled"):
            print(f"[INFO] キャンセル: {video_path}")
            break
        results.append((out_path, stats))
        print(f"[INFO] 完了: {out_path} ({stats['elapsed_sec']}秒, "
              f"検出{stats['detected_frames']}f/適用{stats['covered_frames']}f/セル{stats['cell']}px)")
    return results


def rescan_and_fix(results: list, pattern: str, cfg: dict, detector) -> str:
    """出力を再スキャンし、漏れがあれば再処理で修正する。戻り値: レポート文字列"""
    report_lines = []
    for out_path, _ in results:
        if not os.path.exists(out_path):
            continue
        pw = mg.ProgressWindow(f"再スキャン検証: {os.path.basename(out_path)}", accent="#ff9500")

        def on_progress(stage, cur, total):
            pw.update("再スキャン検証", cur, total, extra=os.path.basename(out_path))

        leftover = mc.verify_video(out_path, cfg, detector,
                                   progress=on_progress, cancel=lambda: pw.cancelled)
        cancelled = pw.cancelled
        pw.close()
        if cancelled:
            report_lines.append(f"{os.path.basename(out_path)}: 検証キャンセル")
            break
        if leftover <= 0:
            report_lines.append(f"{os.path.basename(out_path)}: 漏れなし ✓")
            continue

        # 漏れあり → 出力をもう一度処理して差し替え (再エンコード1回ぶんの劣化は CRF18 で最小限)
        report_lines.append(f"{os.path.basename(out_path)}: {leftover}フレームに検出残り → 修正実行")
        fix_path = out_path + ".fix.mp4"
        pw = mg.ProgressWindow(f"漏れ修正: {os.path.basename(out_path)}", accent="#ff9500")

        def on_fix(stage, cur, total):
            pw.update("漏れ修正 " + ("検出" if stage == "detect" else "適用"), cur, total)

        try:
            stats = mc.process_video(out_path, fix_path, pattern, cfg, detector,
                                     keep_audio=True, progress=on_fix,
                                     cancel=lambda: pw.cancelled)
            pw.close()
            if not stats.get("cancelled") and os.path.exists(fix_path):
                os.replace(fix_path, out_path)
                report_lines.append(f"  → 修正完了 (適用{stats['covered_frames']}フレーム)")
            elif os.path.exists(fix_path):
                os.remove(fix_path)
        except Exception as e:
            pw.close()
            if os.path.exists(fix_path):
                os.remove(fix_path)
            report_lines.append(f"  → 修正失敗: {e}")
    return "\n".join(report_lines)


def main() -> None:
    video_paths = collect_videos()
    if not video_paths:
        print("キャンセルされました。処理を中止します。")
        return

    pattern = mg.ask_mosaic_pattern()
    if pattern is None:
        print("キャンセルされました。処理を中止します。")
        return

    cfg = mc.load_config()
    print("[INFO] 検出エンジンを初期化しています...")
    try:
        detector = mc.NsfwDetector(cfg)
    except Exception as e:
        mg.show_error("エラー", f"検出モデルの読み込みに失敗しました。\n{e}")
        return

    t0 = time.time()
    results = process_all(video_paths, pattern, cfg, detector)
    if not results:
        mg.show_info("終了", "処理された動画はありません。")
        return

    total_sec = time.time() - t0
    outs = "\n".join(p for p, _ in results)
    msg = (f"全ての動画の処理が完了しました。\n"
           f"出力数: {len(results)} / 所要時間: {total_sec:.0f}秒\n{outs}")

    if mg.ask_yesno("完了", msg + "\n\n再スキャン検証を行いますか？\n（出力動画を再チェックしてモザイク漏れを検出・修正します）"):
        report = rescan_and_fix(results, pattern, cfg, detector)
        mg.show_info("再スキャン完了", report or "検証対象がありません。")

    sys.exit(0)


if __name__ == "__main__":
    main()
