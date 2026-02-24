import os
import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
from tkinter import ttk
import tempfile
import ffmpeg
import shutil
from collections import deque

# NudeNet (Layer 4) - optional
try:
    from nudenet import NudeDetector
except ImportError:
    NudeDetector = None

# NudeNet NSFW labels that require mosaic
NUDENET_NSFW_LABELS = {
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED', 
    'ANUS_EXPOSED',
}

# --- Constants for Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'tmp')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cleanup_tmp_dir():
    """Clears all files in the temporary directory."""
    if os.path.exists(TEMP_DIR):
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"[WARNING] Failed to delete {file_path}. Reason: {e}")

def ask_mosaic_pattern():
    # ...existing code from mosaic-video.py...
    import tkinter as tk
    from tkinter import ttk
    patterns = ["モザイク小", "モザイク中", "モザイク大", "ぼかし", "黒塗り"]
    selected = [patterns[0]]
    cancelled = [False]
    def on_select(event=None):
        idx = listbox.curselection()
        if idx:
            selected[0] = patterns[idx[0]]
            root.quit()
    def on_ok():
        idx = listbox.curselection()
        if idx:
            selected[0] = patterns[idx[0]]
        root.quit()
    def on_cancel():
        cancelled[0] = True
        root.quit()
    root = tk.Tk()
    root.title("モザイクパターン選択")
    root.geometry("420x360")
    root.configure(bg="#23272e")
    title = tk.Label(root, text="モザイクパターンを選択してください", font=("Segoe UI", 17, "bold"), bg="#23272e", fg="#fff")
    title.pack(padx=10, pady=18)
    listbox_frame = tk.Frame(root, bg="#23272e")
    listbox_frame.pack(padx=24, pady=8, fill=tk.BOTH, expand=True)
    listbox = tk.Listbox(listbox_frame, height=len(patterns), font=("Segoe UI", 15), bg="#181a20", fg="#fff", selectbackground="#00bfff", selectforeground="#fff", relief="flat", highlightthickness=0, bd=0)
    for p in patterns:
        listbox.insert(tk.END, p)
    listbox.selection_set(0)
    listbox.pack(fill=tk.BOTH, expand=True)
    listbox.bind('<Double-1>', on_select)
    btn_frame = tk.Frame(root, bg="#23272e")
    btn_frame.pack(pady=18)
    btn_ok = tk.Button(btn_frame, text="OK", font=("Segoe UI", 14), width=10, height=2, bg="#23272e", fg="#fff", relief="raised", bd=3, activebackground="#00bfff", activeforeground="#fff", command=on_ok)
    btn_ok.pack(side=tk.LEFT, padx=16)
    btn_ok.bind("<Enter>", lambda e: e.widget.config(bg="#00bfff", fg="#fff", relief="raised", bd=3))
    btn_ok.bind("<Leave>", lambda e: e.widget.config(bg="#23272e", fg="#fff", relief="raised", bd=3))
    btn_cancel = tk.Button(btn_frame, text="キャンセル", font=("Segoe UI", 14), width=10, height=2, bg="#23272e", fg="#fff", relief="raised", bd=3, activebackground="#ff5555", activeforeground="#fff", command=on_cancel)
    btn_cancel.pack(side=tk.LEFT, padx=16)
    btn_cancel.bind("<Enter>", lambda e: e.widget.config(bg="#ff5555", fg="#fff", relief="raised", bd=3))
    btn_cancel.bind("<Leave>", lambda e: e.widget.config(bg="#23272e", fg="#fff", relief="raised", bd=3))
    root.mainloop()
    root.destroy()
    if cancelled[0]:
        return None
    return selected[0]

def ask_audio_add():
    root = tk.Tk()
    root.withdraw()
    result = tkMessageBox.askyesno("音声追加", "音声ファイルを追加しますか？")
    root.destroy()
    return result

def ask_audio_file():
    root = tk.Tk()
    root.withdraw()
    path = tkFileDialog.askopenfilename(
        title="音声ファイルを選択してください",
        filetypes=[("音声ファイル", "*.mp3;*.wav;*.aac;*.m4a"), ("All files", "*.*")]
    )
    root.destroy()
    return path

def apply_pattern(region, pattern):
    
    w, h = region.size
    if pattern == "モザイク大":
        small = region.resize((max(1, w // 32), max(1, h // 32)), Image.Resampling.BICUBIC)
        return small.resize((w, h), Image.Resampling.NEAREST)
    elif pattern == "モザイク中":
        small = region.resize((max(1, w // 16), max(1, h // 16)), Image.Resampling.BICUBIC)
        return small.resize((w, h), Image.Resampling.NEAREST)
    elif pattern == "モザイク小":
        small = region.resize((max(1, w // 8), max(1, h // 8)), Image.Resampling.BICUBIC)
        return small.resize((w, h), Image.Resampling.NEAREST)
    elif pattern == "ぼかし":
        from PIL import ImageFilter
        blur_radius = max(8, min(w, h) // 10)
        return region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    elif pattern == "黒塗り":
        return Image.new("RGB", (w, h), (0, 0, 0))
    else:
        return region

# --- Class-based shrink ratios for optimal mosaic coverage ---
SHRINK_RATIOS = {
    'penis':  (0.70, 0.55),
    'vagina': (0.70, 0.55),
    'anus':   (0.65, 0.65),
}
DEFAULT_SHRINK = (0.60, 0.50)

def shrink_box(x1, y1, x2, y2, cls_name=''):
    """Shrink detection box to mosaic area based on detected class."""
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return None
    ratio_w, ratio_h = SHRINK_RATIOS.get(cls_name, DEFAULT_SHRINK)
    dx = int(w * ratio_w / 2)
    dy = int(h * ratio_h / 2)
    sx1 = x1 + dx; sy1 = y1 + dy
    sx2 = x2 - dx; sy2 = y2 - dy
    if sx2 > sx1 and sy2 > sy1:
        return (sx1, sy1, sx2, sy2)
    return None

def merge_boxes(all_boxes, iou_threshold=0.3):
    """De-duplicate overlapping boxes using IoU. Keeps larger box on overlap."""
    merged = []
    for box in all_boxes:
        is_duplicate = False
        for i, existing in enumerate(merged):
            ix1 = max(box[0], existing[0]); iy1 = max(box[1], existing[1])
            ix2 = min(box[2], existing[2]); iy2 = min(box[3], existing[3])
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                existing_area = (existing[2] - existing[0]) * (existing[3] - existing[1])
                union_area = box_area + existing_area - inter_area
                if union_area > 0 and inter_area / union_area > iou_threshold:
                    is_duplicate = True
                    if box_area > existing_area:
                        merged[i] = box
                    break
        if not is_duplicate:
            merged.append(box)
    return merged


def ask_video_mode():
    import tkinter as tk
    mode = {'value': None}
    def select_file():
        mode['value'] = 'file'
        root.quit()
    def select_folder():
        mode['value'] = 'folder'
        root.quit()
    root = tk.Tk()
    root.title("動画処理モード選択")
    root.geometry("480x260")
    root.configure(bg="#23272e")
    tk.Label(root, text="処理方法を選択してください", font=("Segoe UI", 18, "bold"), bg="#23272e", fg="#fff").pack(pady=32)
    btn_file = tk.Button(root, text="動画ファイルを選択", font=("Segoe UI", 15), width=22, height=2, bg="#23272e", fg="#fff", relief="raised", bd=3, activebackground="#00bfff", activeforeground="#fff", command=select_file)
    btn_file.pack(pady=12)
    btn_folder = tk.Button(root, text="フォルダ内の全動画を一括処理", font=("Segoe UI", 15), width=28, height=2, bg="#23272e", fg="#fff", relief="raised", bd=3, activebackground="#00bfff", activeforeground="#fff", command=select_folder)
    btn_folder.pack(pady=12)
    root.mainloop()
    root.destroy()
    return mode['value']

def adjust_audio_to_video(audio_path, video_path, output_path):
    import ffmpeg # 関数内で再度インポートしていますが、グローバルでもOK
    import tempfile
    import os

    probe_v = ffmpeg.probe(video_path)
    probe_a = ffmpeg.probe(audio_path)
    v_duration = float(probe_v['format']['duration'])
    a_duration = float(probe_a['format']['duration'])

    temp_audio_path = "" # 初期化
    try:
        # 一時音声ファイルを作成 (削除は finally で行う)
        with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False, dir=TEMP_DIR) as tmp_audio_file:
            temp_audio_path = tmp_audio_file.name
        
        print(f"[DEBUG] temp_audio_path: {temp_audio_path}")

        # 音声の長さを調整して一時ファイルに保存
        if abs(v_duration - a_duration) < 0.01: # ほぼ同じ長さ
            print("[DEBUG] 音声長さほぼ同じ: AAC変換のみ (m4a)")
            stream = ffmpeg.input(audio_path)
        elif a_duration > v_duration: # 音声が長い場合、トリミング
            print("[DEBUG] 音声が長い: トリミングしてAAC変換 (m4a)")
            stream = ffmpeg.input(audio_path).filter('atrim', duration=v_duration).filter('asetpts', 'N/SR/TB')
        else: # 音声が短い場合
            tempo_ratio = v_duration / a_duration # 目標の長さ / 元の長さ
            if 0.5 <= tempo_ratio <= 2.0: # atempoが対応可能な範囲 (0.5倍速から2倍速)
                print(f"[DEBUG] 音声が短い: atempo ({tempo_ratio:.2f}) でAAC変換 (m4a)")
                stream = ffmpeg.input(audio_path).filter('atempo', tempo_ratio).filter('asetpts', 'N/SR/TB')
            else: # atempoの範囲外ならループしてトリミング
                print(f"[DEBUG] 音声が短い (atempo範囲外): ループしてトリミング、AAC変換 (m4a)")
                stream = (
                    ffmpeg
                    .input(audio_path)
                    .filter_('aloop', loop=-1, size=2**24) # 無限ループ、sizeは十分な値を指定
                    .filter_('atrim', duration=v_duration)
                    .filter_('asetpts', 'N/SR/TB')
                )
        
        # 共通の出力設定で一時音声ファイルに書き出し
        (
            stream
            .output(temp_audio_path, acodec='aac', audio_bitrate='192k', format='ipod')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[DEBUG] 一時音声ファイル {temp_audio_path} の生成完了。")

        # 映像と調整済み音声をmux（多重化）
        print(f"[DEBUG] Mux開始: video={video_path}, audio={temp_audio_path}, out={output_path}")
        print(f"[DEBUG] ffmpeg muxコマンド (相当): ffmpeg -y -i \"{video_path}\" -i \"{temp_audio_path}\" -c:v copy -c:a aac -b:a 192k \"{output_path}\"")
        
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(temp_audio_path)
        
        (
            ffmpeg
            .output(input_video['v'], input_audio['a'], output_path, vcodec='libx264', pix_fmt='yuv420p', crf=23, acodec='aac', audio_bitrate='192k') # H.264変換
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("[DEBUG] Mux完了")

    except ffmpeg.Error as e:
        print(f"ffmpeg error during audio adjustment or mux (video: {video_path}, audio_in: {audio_path}):")
        stdout = e.stdout.decode(errors='replace') if e.stdout else "No stdout"
        stderr = e.stderr.decode(errors='replace') if e.stderr else "No stderr"
        print(f"FFMPEG STDOUT:\n{stdout}")
        print(f"FFMPEG STDERR:\n{stderr}")
        raise # エラーを再送出して呼び出し元で処理できるようにする
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"[DEBUG] 一時音声ファイル {temp_audio_path} を削除しました。")
            except OSError as e:
                print(f"[ERROR] 一時音声ファイル {temp_audio_path} の削除に失敗しました: {e}")

def transcode_to_h264(input_path, output_path):
    import ffmpeg
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec='libx264', pix_fmt='yuv420p', crf=23)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception as e:
        print(f"Transcoding failed: {e}")
        return False

# Helper for muxing ORIGINAL audio if no extra audio added
def mux_original_audio(video_path, audio_source, output_path):
    import ffmpeg
    try:
        # Check if audio source has audio stream
        probe = ffmpeg.probe(audio_source)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if not audio_streams:
            return False # No audio to mux
        
        # Create temp output for muxing
        temp_mux_out = output_path + ".temp_mux.mp4"
        
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_source)

        (
            ffmpeg
            .output(input_video['v'], input_audio['a'], temp_mux_out, vcodec='libx264', pix_fmt='yuv420p', crf=23, acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
        
        if os.path.exists(temp_mux_out):
            if os.path.exists(output_path): os.remove(output_path)
            os.rename(temp_mux_out, output_path)
            return True
    except Exception as e:
        print(f"Audio muxing failed: {e}")
        if os.path.exists(temp_mux_out): os.remove(temp_mux_out)
    return False

def rescan_video(video_path, model_detect, model_nudenet, pattern, names):
    """Post-scan verification: re-scan output video and fix any missed areas."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video for rescan: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_rescan = os.path.join(TEMP_DIR, f"rescan_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(temp_rescan, fourcc, fps, (width, height))
    
    # Progress bar
    if tk._default_root:
         progress_root = tk.Toplevel()
    else:
         progress_root = tk.Tk()
    progress_root.title(f"再スキャン検証中: {os.path.basename(video_path)}")
    progress_root.geometry("440x170")
    if not tk._default_root: # Only config bg for Tk, Toplevel inherits or set separately? Toplevel accepts config.
         progress_root.configure(bg="#23272e")
    else:
         progress_root.configure(bg="#23272e")
    style = ttk.Style(progress_root)
    style.theme_use("clam")
    style.layout("Rescan.Horizontal.TProgressbar",
                [('Horizontal.Progressbar.trough', {'children': [
                    ('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})], 'sticky': 'nswe'})])
    style.configure("Rescan.Horizontal.TProgressbar",
        troughcolor="#181a20", bordercolor="#23272e", background="#ff9500",
        lightcolor="#ff9500", darkcolor="#cc7700", thickness=22, borderwidth=2, relief="flat")
    tk.Label(progress_root, text="再スキャン検証中...", font=("Segoe UI", 15, "bold"), bg="#23272e", fg="#fff").pack(pady=12)
    progress_var = tk.DoubleVar()
    progress = ttk.Progressbar(progress_root, variable=progress_var, maximum=total, length=380, style="Rescan.Horizontal.TProgressbar")
    progress.pack(pady=8)
    status_label = tk.Label(progress_root, text="", font=("Segoe UI", 12), bg="#23272e", fg="#fff")
    status_label.pack(pady=2)
    percent_label = tk.Label(progress_root, text="", font=("Segoe UI", 12), bg="#23272e", fg="#fff")
    percent_label.pack(pady=2)
    progress_root.update()
    
    fixed_count = 0
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        
        if idx % 10 == 0 or idx == 1 or idx == total:
            status_label.config(text=f"{idx}/{total} フレーム (修正: {fixed_count})")
            percent = int(idx / total * 100)
            percent_label.config(text=f"進捗: {percent}%")
            progress_var.set(idx)
            progress_root.update()
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_rgb = np.array(img)
        detected_boxes = []
        
        # Detection with model_detect (isolated YOLO)
        try:
            results = model_detect(frame_rgb, conf=0.10, iou=0.3, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, cls_idx in zip(boxes, clss):
                    cls_name = names[cls_idx] if cls_idx < len(names) else ""
                    if cls_name in ["make_love", "nipple"]: continue
                    x1, y1, x2, y2 = box
                    sbox = shrink_box(x1, y1, x2, y2, cls_name)
                    if sbox:
                        detected_boxes.append(sbox)
        except Exception:
            pass
        
        # Detection with NudeNet
        if model_nudenet is not None:
            try:
                nn_tmp = os.path.join(TEMP_DIR, '_nn_rescan_tmp.jpg')
                cv2.imwrite(nn_tmp, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                nn_results = model_nudenet.detect(nn_tmp)
                for det in nn_results:
                    label = det.get('class', '')
                    score = det.get('score', 0)
                    if label not in NUDENET_NSFW_LABELS: continue
                    if score < 0.3: continue
                    nn_box = det.get('box', [])
                    if len(nn_box) != 4: continue
                    x1, y1, x2, y2 = int(nn_box[0]), int(nn_box[1]), int(nn_box[2]), int(nn_box[3])
                    sbox = shrink_box(x1, y1, x2, y2)
                    if sbox:
                        detected_boxes.append(sbox)
            except Exception:
                pass
        
        detected_boxes = merge_boxes(detected_boxes)
        if detected_boxes:
            fixed_count += 1
            for (sx1, sy1, sx2, sy2) in detected_boxes:
                region = img.crop((sx1, sy1, sx2, sy2))
                mosaic = apply_pattern(region, pattern)
                img.paste(mosaic, (sx1, sy1, sx2, sy2))
        
        out_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(out_frame)
    
    cap.release()
    out.release()
    progress_root.destroy()
    
    if fixed_count > 0:
        has_audio = mux_original_audio(temp_rescan, video_path, video_path + ".tmp")
        if has_audio:
            if os.path.exists(video_path): os.remove(video_path)
            os.rename(video_path + ".tmp", video_path)
            if os.path.exists(temp_rescan): os.remove(temp_rescan)
        else:
            # No audio, force transcode to H.264
            transcode_to_h264(temp_rescan, video_path)
        print(f"[INFO] Rescan complete: {fixed_count} frames fixed.")
    else:
        if os.path.exists(temp_rescan): os.remove(temp_rescan)
        print("[INFO] Rescan complete: no additional fixes needed.")
    
    return fixed_count


def main():
    
    # Check if tmp and output dirs exist
    if not os.path.exists(TEMP_DIR):
        try: os.makedirs(TEMP_DIR)
        except OSError: pass
            
    if not os.path.exists(OUTPUT_DIR):
        try: os.makedirs(OUTPUT_DIR)
        except OSError: pass

    # モデル・クラス名
    names = ['anus', 'make_love', 'nipple', 'penis', 'vagina']
    yolo_model_path = os.path.join(os.path.dirname(__file__), 'erax_nsfw_yolo11m.pt')
    if not os.path.exists(yolo_model_path):
        tkMessageBox.showerror("エラー", f"YOLOモデルファイルが見つかりません: {yolo_model_path}")
        return
    try:
        model = YOLO(yolo_model_path)          # For Layer 1: Tracking
        model_detect = YOLO(yolo_model_path)   # For Layer 2: Standalone detection (isolated)
    except Exception as e:
        tkMessageBox.showerror("エラー", f"YOLOモデルのロードに失敗しました: {e}")
        return
    
    # Layer 4: NudeNet (optional, graceful skip if unavailable)
    model_nudenet = None
    if NudeDetector is not None:
        try:
            model_nudenet = NudeDetector()
            print("[INFO] NudeNet Layer 4 loaded successfully.")
        except Exception as e:
            print(f"[WARNING] NudeNet initialization failed (Layer 4 disabled): {e}")

    mode = ask_video_mode()
    if mode == 'file':
        root = tk.Tk(); root.withdraw()
        video_path_tuple = tkFileDialog.askopenfilename(
            title="動画ファイルを選択してください",
            filetypes=[
                ("動画ファイル", "*.mp4;*.avi;*.mov;*.mkv"), # .mkv追加
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
                ("All files", "*.*")
            ]) # askopenfilenameは単一選択の場合文字列を返す
        root.destroy()
        if not video_path_tuple: # キャンセルされた場合
             print("動画ファイルが選択されませんでした。処理を中止します。")
             return
        video_paths = [video_path_tuple]
    elif mode == 'folder':
        root = tk.Tk(); root.withdraw()
        folder = tkFileDialog.askdirectory(title="動画フォルダを選択してください")
        root.destroy()
        if not folder:
            print("フォルダが選択されませんでした。処理を中止します。")
            return
        video_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if not video_paths:
            tkMessageBox.showinfo("動画なし", "選択フォルダに対応動画がありません。", parent=None)
            return
    else: # ask_video_modeがNoneを返した場合 (キャンセルなど)
        print("処理モードが選択されませんでした。処理を中止します。")
        return

    pattern = ask_mosaic_pattern()
    if pattern is None:
        print("モザイクパターンが選択されませんでした。処理を中止します。")
        return

    add_audio = ask_audio_add()
    audio_path = None
    if add_audio:
        audio_path = ask_audio_file()
        if not audio_path:
            tkMessageBox.showinfo("音声なし", "音声ファイルが選択されませんでした。音声なしで進めます。", parent=None)
            add_audio = False # 音声ファイルが選択されなかったのでフラグをFalseに

    processed_outputs = []  # 追加
    
    for video_path in video_paths:
        if not video_path or not os.path.exists(video_path): # パスが空かファイルが存在しない場合スキップ
            print(f"無効なビデオパス、またはファイルが存在しません: {video_path}")
            continue

        print(f"処理開始: {video_path}")
        base, ext = os.path.splitext(video_path)
        filename = os.path.basename(video_path)
        name_only = os.path.splitext(filename)[0]
        
        # Output directory handling
        if ext.lower() == ".mp4" or ext.lower() == ".mov" or ext.lower() == ".mkv":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # H.264エンコードなら 'X264' や 'H264' も検討
            out_filename = name_only + "_mc" + ext # Keep original extension if supported container
            if ext.lower() == ".mkv": # Optional: force mp4 for mkv if preferred, but existing logic kept
                pass
        elif ext.lower() == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_filename = name_only + "_mc.avi"
        else:
            tkMessageBox.showwarning("未対応形式", f"動画ファイル形式 {ext} は部分的な対応となる可能性があります。MP4として処理を試みます。")
            print(f"未対応の動画ファイル形式 {ext} です。MP4 (mp4v) として処理を試みます。")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_filename = name_only + "_mc.mp4" # 出力拡張子を .mp4 に固定

        out_path = os.path.join(OUTPUT_DIR, out_filename)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けませんでした: {video_path}")
            tkMessageBox.showerror("エラー", f"動画ファイルを開けませんでした: {video_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0 : # フレーム数が取得できない場合
            print(f"警告: 動画の総フレーム数を取得できませんでした: {video_path}")
            total_frames = 1 # ダミー値（プログレスバー表示のため）

        # モザイク処理用の一時ビデオファイル (音声なし)
        temp_video_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=TEMP_DIR) as tmp_vid:
                temp_video_path = tmp_vid.name
            
            out_video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            if not out_video_writer.isOpened():
                print(f"エラー: 一時ビデオファイルの作成に失敗しました: {temp_video_path}")
                tkMessageBox.showerror("エラー", f"一時ビデオファイルの作成に失敗しました。")
                cap.release()
                if os.path.exists(temp_video_path): os.remove(temp_video_path)
                continue
            
            # Tracking History
            track_history = {}
            MAX_LOST_FRAMES = 15
            last_known_boxes = []
            no_detection_count = 0

            # 進捗バーGUI
            progress_root = tk.Tk()
            progress_root.title(f"動画モザイク処理進捗: {os.path.basename(video_path)}")
            progress_root.geometry("440x170")
            progress_root.configure(bg="#23272e")
            style = ttk.Style(progress_root)
            style.theme_use("clam")
            style.layout("Cool.Horizontal.TProgressbar",
                        [('Horizontal.Progressbar.trough', {'children': [
                            ('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})], 'sticky': 'nswe'})])
            style.configure("Cool.Horizontal.TProgressbar",
                            troughcolor="#181a20", bordercolor="#23272e", background="#00bfff", lightcolor="#00bfff", darkcolor="#005f8f", thickness=22, borderwidth=2, relief="flat")
            tk.Label(progress_root, text=f"動画を処理中...", font=("Segoe UI", 15, "bold"), bg="#23272e", fg="#fff").pack(pady=12)
            progress_var = tk.DoubleVar()
            progress = ttk.Progressbar(progress_root, variable=progress_var, maximum=total_frames, length=380, style="Cool.Horizontal.TProgressbar")
            progress.pack(pady=8)
            status_label = tk.Label(progress_root, text="", font=("Segoe UI", 12), bg="#23272e", fg="#fff")
            status_label.pack(pady=2)
            percent_label = tk.Label(progress_root, text="", font=("Segoe UI", 12), bg="#23272e", fg="#fff")
            percent_label.pack(pady=2)
            progress_root.update()

            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # --- Performance Optimization: GUI Update every 10 frames ---
                if (frame_idx + 1) % 10 == 0 or frame_idx == 0 or frame_idx == total_frames - 1:
                    status_label.config(text=f"{frame_idx + 1}/{total_frames} フレーム")
                    percent = int((frame_idx + 1) / total_frames * 100) if total_frames > 0 else 0
                    percent_label.config(text=f"進捗: {percent}%")
                    progress_var.set(frame_idx + 1)
                    progress_root.update()

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # --- Multi-Layer Detection Logic Start ---
                frame_rgb = np.array(img)
                layer1_boxes = []  # Boxes from tracking
                layer2_boxes = []  # Boxes from standalone detection
                current_ids = set()
                
                # ===== LAYER 1: Tracking Detection =====
                try:
                    results = model.track(frame_rgb, persist=True, conf=0.10, iou=0.3, tracker="bytetrack.yaml", verbose=False)
                    
                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(boxes)
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)
                        
                        for box, track_id, cls_idx in zip(boxes, ids, clss):
                            cls_name = names[cls_idx] if cls_idx < len(names) else ""
                            if cls_name in ["make_love", "nipple"]: continue
                            
                            x1, y1, x2, y2 = box
                            sbox = shrink_box(x1, y1, x2, y2, cls_name)
                            if sbox:
                                layer1_boxes.append(sbox)
                                if track_id is not None:
                                    track_history[track_id] = {'box': sbox, 'lost_count': 0}
                                    current_ids.add(track_id)
                except Exception as e:
                    print(f"[WARNING] Layer 1 (tracking) failed on frame {frame_idx}: {e}")
                
                # ===== LAYER 2: Standalone Detection (ALWAYS runs as cross-check) =====
                try:
                    results2 = model_detect(frame_rgb, conf=0.10, iou=0.3, verbose=False)
                    
                    if results2 and results2[0].boxes is not None and len(results2[0].boxes) > 0:
                        boxes2 = results2[0].boxes.xyxy.cpu().numpy().astype(int)
                        clss2 = results2[0].boxes.cls.cpu().numpy().astype(int)
                        
                        for box, cls_idx in zip(boxes2, clss2):
                            cls_name = names[cls_idx] if cls_idx < len(names) else ""
                            if cls_name in ["make_love", "nipple"]: continue
                            
                            x1, y1, x2, y2 = box
                            sbox = shrink_box(x1, y1, x2, y2, cls_name)
                            if sbox:
                                layer2_boxes.append(sbox)
                except Exception as e:
                    print(f"[WARNING] Layer 2 (detection) failed on frame {frame_idx}: {e}")
                
                # ===== LAYER 4: NudeNet Cross-Check =====
                layer4_boxes = []
                if model_nudenet is not None:
                    try:
                        # Save to temp for NudeNet (some versions/backends prefer file paths)
                        nn_tmp = os.path.join(TEMP_DIR, '_nn_tmp_speek.jpg')
                        cv2.imwrite(nn_tmp, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        nn_results = model_nudenet.detect(nn_tmp)
                        
                        for det in nn_results:
                            label = det.get('class', '')
                            score = det.get('score', 0)
                            if label not in NUDENET_NSFW_LABELS: continue
                            if score < 0.3: continue
                            
                            nn_box = det.get('box', [])
                            if len(nn_box) != 4: continue
                            x1, y1, x2, y2 = int(nn_box[0]), int(nn_box[1]), int(nn_box[2]), int(nn_box[3])
                            
                            sbox = shrink_box(x1, y1, x2, y2)
                            if sbox:
                                layer4_boxes.append(sbox)
                    except Exception as e:
                        print(f"[WARNING] Layer 4 (NudeNet) failed on frame {frame_idx}: {e}")
                
                # Merge results from all layers
                all_boxes = layer1_boxes + layer2_boxes + layer4_boxes
                merged_boxes = merge_boxes(all_boxes)
                
                # Apply mosaic to all merged boxes
                for (sx1, sy1, sx2, sy2) in merged_boxes:
                    region = img.crop((sx1, sy1, sx2, sy2))
                    mosaic = apply_pattern(region, pattern)
                    img.paste(mosaic, (sx1, sy1, sx2, sy2))
                
                # Update last_known_boxes if we found anything
                if len(merged_boxes) > 0:
                    last_known_boxes = merged_boxes
                    no_detection_count = 0
                else:
                    no_detection_count += 1
                
                # ===== LAYER 3: History Fallback =====
                if len(merged_boxes) == 0 and last_known_boxes and no_detection_count <= MAX_LOST_FRAMES:
                    for (sx1, sy1, sx2, sy2) in last_known_boxes:
                        img_w, img_h = img.size
                        sx1 = max(0, sx1); sy1 = max(0, sy1)
                        sx2 = min(img_w, sx2); sy2 = min(img_h, sy2)
                        if sx2 > sx1 and sy2 > sy1:
                            region = img.crop((sx1, sy1, sx2, sy2))
                            mosaic = apply_pattern(region, pattern)
                            img.paste(mosaic, (sx1, sy1, sx2, sy2))
                
                # Handle tracked-ID-based Lost Tracks
                for track_id, data in track_history.items():
                    if track_id not in current_ids:
                        data['lost_count'] += 1
                        if data['lost_count'] <= MAX_LOST_FRAMES:
                            sx1, sy1, sx2, sy2 = data['box']
                            img_w, img_h = img.size
                            sx1 = max(0, sx1); sy1 = max(0, sy1)
                            sx2 = min(img_w, sx2); sy2 = min(img_h, sy2)
                            if sx2 > sx1 and sy2 > sy1:
                                region = img.crop((sx1, sy1, sx2, sy2))
                                mosaic = apply_pattern(region, pattern)
                                img.paste(mosaic, (sx1, sy1, sx2, sy2))
                
                # Clean up old tracks
                track_history = {k: v for k, v in track_history.items() if v['lost_count'] <= MAX_LOST_FRAMES}
                
                # Preventive periodic tracker reset every 100 frames
                if (frame_idx + 1) % 100 == 0:
                    try:
                        model.predictor = None
                        # Don't clear track_history — Layer 3 fallback will still work
                    except:
                        pass
                # --- Multi-Layer Detection Logic End ---

                out_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                out_video_writer.write(out_frame)
            
            cap.release()
            out_video_writer.release()
            progress_root.destroy()
            print(f"モザイク処理完了: {temp_video_path}")

            # 音声追加処理
            if add_audio and audio_path and os.path.exists(audio_path):
                # User selected external audio
                print(f"音声合成処理開始: audio={audio_path}, temp_video={temp_video_path}, output={out_path}")
                try:
                    adjust_audio_to_video(audio_path, temp_video_path, out_path)
                    print(f"音声合成成功。最終出力: {out_path}")
                    processed_outputs.append(out_path)
                except Exception as e:
                    print(f"音声合成に失敗しました: {e}")
                    tkMessageBox.showerror("音声合成エラー", f"音声合成に失敗しました: {e}\nモザイク処理済みの動画（音声なし）を保存します。")
                    if os.path.exists(temp_video_path):
                         if os.path.exists(out_path): os.remove(out_path)
                         shutil.move(temp_video_path, out_path)
                         processed_outputs.append(out_path)
            else:
                # No external audio selected. Try to copy ORIGINAL audio.
                if not add_audio:
                     print("外部音声追加は選択されていません。元動画の音声を試みます。")
                
                # Try muxing original audio
                has_audio = mux_original_audio(temp_video_path, video_path, out_path)
                if not has_audio:
                    # Just move/copy temp video -> Now transcode to H.264
                    print(f"音声が見つかりません。映像をH.264に変換して保存します: {out_path}")
                    transcode_to_h264(temp_video_path, out_path)
                    print(f"音声なし。モザイク処理済み動画を保存: {out_path}")
                else:
                    print(f"元動画の音声を合成しました: {out_path}")
                
                processed_outputs.append(out_path)
            
            # tkMessageBox.showinfo("完了", ... ) -> Removed per request to do all at end, 
            # BUT original code had it per video. 
            # I will remove individual popup if folder mode, but keeping loop structure.
            # Actually, user wants "Finally message -> OK -> Close". 
            # So let's NOT show individual popups if we process multiple?
            # Or just show one final one.
            pass # We do final message outside loop.

        except Exception as e:
            print(f"ビデオ {video_path} の処理中にエラーが発生しました: {e}")
            tkMessageBox.showerror("処理エラー", f"ビデオ {os.path.basename(video_path)} の処理中にエラーが発生しました: {e}")
        finally:
            if cap.isOpened(): cap.release()
            if 'out_video_writer' in locals() and out_video_writer.isOpened(): out_video_writer.release() # locals()で存在確認
            if os.path.exists(temp_video_path):
                try:
                    # OUTPUTにmoveされた場合は元のtempは消えているはずだが、copyされた場合やエラー残存の場合に備え削除トライ
                    if os.path.exists(temp_video_path):
                         # out_path と temp_video_path が同じパスでない場合に限り削除
                         if not (os.path.exists(out_path) and os.path.abspath(temp_video_path) == os.path.abspath(out_path)):
                              os.remove(temp_video_path)
                              print(f"一時ビデオファイル {temp_video_path} を削除しました。")
                except OSError as e:
                    print(f"[ERROR] 一時ビデオファイル {temp_video_path} の削除に失敗しました: {e}")
    
    # Clean up all temp files at the end of session
    cleanup_tmp_dir()

    # 通知処理
    msg = ""
    if processed_outputs:
        if mode == 'file':
            msg = f"全てのフレームの処理が完了しました。\n出力: {processed_outputs[0]}"
        else:
            outlist = '\n'.join(processed_outputs)
            msg = f"全ての動画の処理が完了しました。\n出力数: {len(processed_outputs)}\n(詳細はコンソールを確認してください)"
    
    if msg:
        # Create a hidden root to ensuring the dialog appears
        final_root = tk.Tk()
        final_root.withdraw()
        final_root.attributes('-topmost', True)
        final_root.lift()
        final_root.focus_force()
        
        # Offer rescan option
        do_rescan = tkMessageBox.askyesno(
            "完了",
            msg + "\n\n再スキャン検証を行いますか？\n（出力動画を再チェックしてモザイク漏れを修正します）",
            parent=final_root
        )
        
        if do_rescan:
            rescan_total_fixed = 0
            for out_path in processed_outputs:
                if os.path.exists(out_path):
                    fixed = rescan_video(out_path, model_detect, model_nudenet, pattern, names)
                    if fixed:
                        rescan_total_fixed += fixed
            
            if rescan_total_fixed > 0:
                tkMessageBox.showinfo("再スキャン完了", f"再スキャン完了: {rescan_total_fixed}フレームを修正しました。", parent=final_root)
            else:
                tkMessageBox.showinfo("再スキャン完了", "再スキャン完了: 修正が必要なフレームはありませんでした。", parent=final_root)
        
        final_root.destroy()
    
    # FORCE EXIT APP
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
