# -*- coding: utf-8 -*-
"""
mosaic-video-speek.py — NSFW自動モザイク (動画+音声調整) 2026-07 刷新版

mosaic-video.py の全改善 (2パス時間安定化 / 審査基準セル / 高品質一発エンコード) に加え、
任意の音声ファイルを動画長に自動調整 (トリミング / 速度調整 / ループ) して合成できる。

音声合成時の改善:
  * 旧版は合成時に映像を再エンコードしていた (mp4v→H.264→H.264 の三重劣化) が、
    刷新版はモザイクエンコード済み映像を -c:v copy で無劣化mux する
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
TEMP_DIR = os.path.join(BASE_DIR, "tmp")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def ask_audio_add() -> bool:
    return mg.ask_yesno("音声追加", "音声ファイルを追加しますか？\n（いいえ: 元動画の音声をそのまま使用）")


def ask_audio_file() -> str:
    return mg.pick_file(
        "音声ファイルを選択してください",
        [("音声ファイル", "*.mp3;*.wav;*.aac;*.m4a;*.flac;*.ogg"), ("All files", "*.*")],
    )


def adjust_audio_and_mux(video_path: str, audio_path: str, out_path: str,
                         log=print) -> None:
    """音声を動画長に合わせて調整し、映像は無劣化 (-c:v copy) で合成する。

    調整ルール (旧版と同一):
      音声が長い   → トリミング
      音声が短い   → 0.5〜2.0倍の範囲なら atempo 速度調整、それ以外はループ+トリミング
    """
    import ffmpeg

    os.makedirs(TEMP_DIR, exist_ok=True)
    v_dur = float(ffmpeg.probe(video_path)["format"]["duration"])
    a_dur = float(ffmpeg.probe(audio_path)["format"]["duration"])
    log(f"[INFO] 動画 {v_dur:.2f}s / 音声 {a_dur:.2f}s")

    temp_audio = os.path.join(TEMP_DIR, "_adjusted_audio.m4a")
    if abs(v_dur - a_dur) < 0.01:
        stream = ffmpeg.input(audio_path)
    elif a_dur > v_dur:
        log("[INFO] 音声が長いためトリミングします")
        stream = ffmpeg.input(audio_path).filter("atrim", duration=v_dur).filter("asetpts", "N/SR/TB")
    else:
        ratio = v_dur / a_dur
        if 0.5 <= ratio <= 2.0:
            log(f"[INFO] 音声を {1.0 / ratio:.2f}x 速度調整して合わせます")
            stream = ffmpeg.input(audio_path).filter("atempo", 1.0 / ratio).filter("asetpts", "N/SR/TB")
        else:
            log("[INFO] 音声が大幅に短いためループ+トリミングします")
            stream = (
                ffmpeg.input(audio_path)
                .filter_("aloop", loop=-1, size=2 ** 24)
                .filter_("atrim", duration=v_dur)
                .filter_("asetpts", "N/SR/TB")
            )
    (
        stream.output(temp_audio, acodec="aac", audio_bitrate="192k", format="ipod")
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    try:
        iv = ffmpeg.input(video_path)
        ia = ffmpeg.input(temp_audio)
        (
            ffmpeg.output(
                iv["v"], ia["a"], out_path,
                vcodec="copy",           # 映像は無劣化コピー (旧版は再エンコードしていた)
                acodec="copy",
                shortest=None,
                movflags="+faststart",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    finally:
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass


def collect_videos() -> list:
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


def main() -> None:
    video_paths = collect_videos()
    if not video_paths:
        print("キャンセルされました。処理を中止します。")
        return

    pattern = mg.ask_mosaic_pattern()
    if pattern is None:
        print("キャンセルされました。処理を中止します。")
        return

    # 音声設定 (全動画に共通適用)
    audio_path = None
    if ask_audio_add():
        audio_path = ask_audio_file()
        if not audio_path:
            print("音声ファイルが選択されませんでした。元音声のまま処理します。")
            audio_path = None

    cfg = mc.load_config()
    print("[INFO] 検出エンジンを初期化しています...")
    try:
        detector = mc.NsfwDetector(cfg)
    except Exception as e:
        mg.show_error("エラー", f"検出モデルの読み込みに失敗しました。\n{e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    t0 = time.time()
    results = []
    for i, video_path in enumerate(video_paths, 1):
        name = os.path.splitext(os.path.basename(video_path))[0]
        container = mc.out_container_for(video_path)
        out_path = os.path.join(OUTPUT_DIR, name + "_mc" + container)

        pw = mg.ProgressWindow(f"動画モザイク処理 ({i}/{len(video_paths)}): {os.path.basename(video_path)}")
        stage_names = {"detect": "Pass1: AI検出パス", "encode": "Pass2: モザイク適用+エンコード"}

        def on_progress(stage, cur, total):
            pw.update(stage_names.get(stage, stage), cur, total,
                      extra=os.path.basename(video_path))

        try:
            if audio_path:
                # 音声差し替え: モザイク工程は映像のみ → 調整済み音声を無劣化mux
                temp_video = os.path.join(TEMP_DIR, name + "_video_only.mp4")
                stats = mc.process_video(
                    video_path, temp_video, pattern, cfg, detector,
                    keep_audio=False, progress=on_progress,
                    cancel=lambda: pw.cancelled,
                )
                if not stats.get("cancelled"):
                    pw.update("音声調整+合成", 0, 0, extra=os.path.basename(audio_path))
                    adjust_audio_and_mux(temp_video, audio_path, out_path)
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            else:
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

    if not results:
        mg.show_info("終了", "処理された動画はありません。")
        return

    total_sec = time.time() - t0
    outs = "\n".join(p for p, _ in results)
    msg = (f"全ての動画の処理が完了しました。\n"
           f"出力数: {len(results)} / 所要時間: {total_sec:.0f}秒\n{outs}")

    if mg.ask_yesno("完了", msg + "\n\n再スキャン検証を行いますか？\n（出力動画を再チェックしてモザイク漏れを検出・修正します）"):
        report_lines = []
        for out_path, _ in results:
            if not os.path.exists(out_path):
                continue
            pw = mg.ProgressWindow(f"再スキャン検証: {os.path.basename(out_path)}", accent="#ff9500")

            def on_verify(stage, cur, total):
                pw.update("再スキャン検証", cur, total, extra=os.path.basename(out_path))

            leftover = mc.verify_video(out_path, cfg, detector,
                                       progress=on_verify, cancel=lambda: pw.cancelled)
            cancelled = pw.cancelled
            pw.close()
            if cancelled:
                report_lines.append(f"{os.path.basename(out_path)}: 検証キャンセル")
                break
            if leftover <= 0:
                report_lines.append(f"{os.path.basename(out_path)}: 漏れなし ✓")
                continue
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
        mg.show_info("再スキャン完了", "\n".join(report_lines) or "検証対象がありません。")

    sys.exit(0)


if __name__ == "__main__":
    main()
