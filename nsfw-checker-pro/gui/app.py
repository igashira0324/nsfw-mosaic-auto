# -*- coding: utf-8 -*-
"""
nsfw-checker-pro - Premium GUI Application
Tabbed interface with scan, detail, and settings views.
"""

import tkinter as tk
from tkinter import filedialog, ttk
import customtkinter as ctk
import threading
import queue
import os
import time
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    UI_THEME, UI_COLOR_THEME, SUPPORTED_EXTENSIONS,
    VERDICT_ICONS, STYLE_COLORS, CATEGORY_SCORE_COLORS
)
from core.analyzer import MultiEngineAnalyzer
from core.scorer import Scorer, ScoringResult
from core.file_handler import FileHandler
from reports.exporters import export_csv, export_json, export_html

# Design Setup
ctk.set_appearance_mode(UI_THEME)
ctk.set_default_color_theme(UI_COLOR_THEME)


class NSFWCheckerApp:
    """メイン GUI アプリケーション"""

    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("nsfw-checker-pro — Multi-Engine NSFW Analyzer")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)

        # State
        self.file_items = {}  # {tree_id: Path}
        self.results = {}     # {tree_id: ScoringResult}
        self.result_dicts = []  # For export
        self.processing = False
        self.stop_flag = False
        self.result_queue = queue.Queue()

        # Initialize engines
        self._init_status = "Loading engines..."
        self._setup_loading_screen()
        self.root.after(100, self._init_engines)

    def _setup_loading_screen(self):
        """Show a loading screen while engines initialize."""
        self.loading_frame = ctk.CTkFrame(self.root, fg_color="#0d1117")
        self.loading_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(
            self.loading_frame, text="🔍 nsfw-checker-pro",
            font=ctk.CTkFont(size=32, weight="bold"), text_color="#58a6ff"
        ).pack(pady=(120, 10))

        ctk.CTkLabel(
            self.loading_frame, text="Multi-Engine NSFW Analyzer",
            font=ctk.CTkFont(size=16), text_color="#8b949e"
        ).pack(pady=(0, 30))

        self.loading_label = ctk.CTkLabel(
            self.loading_frame, text="エンジンを初期化中...",
            font=ctk.CTkFont(size=14), text_color="#f0f6fc"
        )
        self.loading_label.pack(pady=10)

        self.loading_progress = ctk.CTkProgressBar(
            self.loading_frame, width=400, mode="indeterminate",
            progress_color="#58a6ff"
        )
        self.loading_progress.pack(pady=10)
        self.loading_progress.start()

    def _init_engines(self):
        """Initialize engines in background thread."""
        def init_worker():
            try:
                self.analyzer = MultiEngineAnalyzer(enable_vision=True, enable_vit=True, enable_lfm=True)
                self.scorer = Scorer()
                self.root.after(0, self._on_engines_loaded)
            except Exception as e:
                self.root.after(0, lambda: self._on_engine_error(str(e)))

        t = threading.Thread(target=init_worker, daemon=True)
        t.start()

    def _on_engines_loaded(self):
        """Called when engines are ready."""
        self.loading_progress.stop()
        self.loading_frame.destroy()
        self._setup_layout()
        self._start_resource_monitor()

    def _on_engine_error(self, error_msg):
        """Called if engine loading fails."""
        self.loading_label.configure(text=f"エラー: {error_msg}")
        self.loading_progress.stop()

    # ─── Layout Setup ───

    def _setup_layout(self):
        """Build the main application layout."""
        # Top bar
        top_bar = ctk.CTkFrame(self.root, height=50, fg_color="#161b22", corner_radius=0)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)

        ctk.CTkLabel(
            top_bar, text="🔍 nsfw-checker-pro",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#58a6ff"
        ).pack(side="left", padx=15)

        # Engine status
        engines = self.analyzer.get_available_engines()
        engine_text = f"✅ {len(engines)} engines active"
        ctk.CTkLabel(
            top_bar, text=engine_text,
            font=ctk.CTkFont(size=12), text_color="#2ecc71"
        ).pack(side="left", padx=10)

        # Resource monitor
        self.resource_label = ctk.CTkLabel(
            top_bar, text="CPU: --% | GPU: --% | VRAM: --",
            font=ctk.CTkFont(size=11), text_color="#8b949e"
        )
        self.resource_label.pack(side="right", padx=15)

        # Main content area: left (file list) + right (preview/detail)
        main_pane = ctk.CTkFrame(self.root, fg_color="#0d1117")
        main_pane.pack(fill="both", expand=True, padx=8, pady=8)

        # ─── Left Panel: Controls + File List ───
        left_panel = ctk.CTkFrame(main_pane, fg_color="#161b22", corner_radius=8, width=950)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 4))

        # Control buttons
        ctrl_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkButton(
            ctrl_frame, text="📁 フォルダ選択", width=130,
            command=self._select_folder, fg_color="#21262d", hover_color="#30363d"
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            ctrl_frame, text="📄 ファイル選択", width=130,
            command=self._select_files, fg_color="#21262d", hover_color="#30363d"
        ).pack(side="left", padx=3)

        self.scan_btn = ctk.CTkButton(
            ctrl_frame, text="▶ スキャン開始", width=140,
            command=self._start_analysis, fg_color="#238636", hover_color="#2ea043",
            font=ctk.CTkFont(weight="bold")
        )
        self.scan_btn.pack(side="left", padx=3)

        self.stop_btn = ctk.CTkButton(
            ctrl_frame, text="⏹ 停止", width=80,
            command=self._stop_analysis, fg_color="#da3633", hover_color="#f85149",
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=3)

        ctk.CTkButton(
            ctrl_frame, text="🗑 クリア", width=80,
            command=self._clear_list, fg_color="#21262d", hover_color="#30363d"
        ).pack(side="left", padx=3)

        # Export buttons
        export_frame = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        export_frame.pack(side="right")

        ctk.CTkButton(
            export_frame, text="📊 CSV", width=65,
            command=lambda: self._export('csv'), fg_color="#21262d", hover_color="#30363d"
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            export_frame, text="📋 JSON", width=70,
            command=lambda: self._export('json'), fg_color="#21262d", hover_color="#30363d"
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            export_frame, text="🌐 HTML", width=70,
            command=lambda: self._export('html'), fg_color="#21262d", hover_color="#30363d"
        ).pack(side="left", padx=2)

        # Status bar
        self.status_label = ctk.CTkLabel(
            left_panel, text="待機中...", font=ctk.CTkFont(size=12), text_color="#8b949e"
        )
        self.status_label.pack(fill="x", padx=10, pady=(0, 5))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(left_panel, width=800, progress_color="#58a6ff")
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 5))
        self.progress_bar.set(0)

        # Treeview for results
        tree_frame = ctk.CTkFrame(left_panel, fg_color="#0d1117", corner_radius=4)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        columns = ('verdict', 'score', 'style', 'gender', 'art_style',
                   'nn_score', 'wd14_score', 'vision_score', 'vit_score', 'lfm_score', 'labels')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', selectmode='browse')

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#0d1117", foreground="#c9d1d9",
                        fieldbackground="#0d1117", font=('Segoe UI', 10), rowheight=26)
        style.configure("Treeview.Heading", background="#21262d", foreground="#8b949e",
                        font=('Segoe UI', 10, 'bold'))
        style.map("Treeview", background=[('selected', '#1f6feb')])

        self.tree.heading('verdict', text='判定')
        self.tree.heading('score', text='スコア')
        self.tree.heading('style', text='スタイル')
        self.tree.heading('gender', text='性別')
        self.tree.heading('art_style', text='画風')
        self.tree.heading('nn_score', text='NudeNet')
        self.tree.heading('wd14_score', text='WD14')
        self.tree.heading('vision_score', text='Vision')
        self.tree.heading('vit_score', text='ViT')
        self.tree.heading('lfm_score', text='LFM')
        self.tree.heading('labels', text='詳細ラベル')

        self.tree.column('verdict', width=100, anchor='center')
        self.tree.column('score', width=65, anchor='center')
        self.tree.column('style', width=80, anchor='center')
        self.tree.column('gender', width=55, anchor='center')
        self.tree.column('art_style', width=60, anchor='center')
        self.tree.column('nn_score', width=70, anchor='center')
        self.tree.column('wd14_score', width=65, anchor='center')
        self.tree.column('vision_score', width=65, anchor='center')
        self.tree.column('vit_score', width=60, anchor='center')
        self.tree.column('lfm_score', width=60, anchor='center')
        self.tree.column('labels', width=220)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        # ─── Right Panel: Preview + Detail ───
        right_panel = ctk.CTkFrame(main_pane, fg_color="#161b22", corner_radius=8, width=500)
        right_panel.pack(side="right", fill="both", padx=(4, 0))
        right_panel.pack_propagate(False)

        ctk.CTkLabel(
            right_panel, text="📸 プレビュー",
            font=ctk.CTkFont(size=14, weight="bold"), text_color="#58a6ff"
        ).pack(pady=(10, 5))

        # Preview area
        self.preview_label = ctk.CTkLabel(right_panel, text="画像を選択してください", text_color="#8b949e")
        self.preview_label.pack(pady=5, padx=10)

        # Detail text area
        self.detail_text = ctk.CTkTextbox(
            right_panel, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#0d1117", text_color="#c9d1d9", height=350, wrap="word"
        )
        self.detail_text.pack(fill="both", expand=True, padx=10, pady=(5, 10))

    # ─── File Selection ───

    def _select_folder(self):
        folder = filedialog.askdirectory(title="フォルダを選択")
        if folder:
            images = FileHandler.get_images_from_folder(folder)
            for img in images:
                self._add_file(img)
            self.status_label.configure(text=f"{len(self.file_items)} ファイル読込済み")

    def _select_files(self):
        files = filedialog.askopenfilenames(
            title="ファイルを選択",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.gif *.webp *.bmp")]
        )
        for f in files:
            self._add_file(Path(f))
        self.status_label.configure(text=f"{len(self.file_items)} ファイル読込済み")

    def _add_file(self, path: Path):
        """Add a file to the tree."""
        item_id = self.tree.insert('', 'end', text=path.name, values=(
            '⏳ 待機', '-', '-', '-', '-', '-', '-', '-', '-', '-', path.name
        ))
        self.file_items[item_id] = path

    def _clear_list(self):
        self.tree.delete(*self.tree.get_children())
        self.file_items.clear()
        self.results.clear()
        self.result_dicts.clear()
        self.progress_bar.set(0)
        self._clear_preview()
        self.status_label.configure(text="クリアしました")

    # ─── Analysis ───

    def _start_analysis(self):
        if not self.file_items:
            return
        if self.processing:
            return

        self.processing = True
        self.stop_flag = False
        self.result_dicts.clear()
        self.scan_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="スキャン中...")

        items = list(self.file_items.items())
        thread = threading.Thread(target=self._process_worker, args=(items,), daemon=True)
        thread.start()
        self._poll_results()

    def _stop_analysis(self):
        self.stop_flag = True
        self.status_label.configure(text="停止中...")

    def _process_worker(self, items):
        """Worker thread for processing files."""
        total = len(items)
        for i, (item_id, path) in enumerate(items):
            if self.stop_flag:
                self.result_queue.put(('stopped', None, None))
                return

            try:
                raw = self.analyzer.analyze_image(path)
                scored = self.scorer.score(raw)
                self.result_queue.put(('result', item_id, scored))
            except Exception as e:
                err_result = ScoringResult()
                err_result.verdict = 'ERROR'
                err_result.verdict_icon = '❌'
                err_result.labels_summary = str(e)
                self.result_queue.put(('result', item_id, err_result))

        self.result_queue.put(('done', None, None))

    def _poll_results(self):
        """Poll result queue and update UI."""
        try:
            while not self.result_queue.empty():
                msg_type, item_id, data = self.result_queue.get_nowait()

                if msg_type == 'result':
                    self._update_tree_item(item_id, data)
                    processed = len(self.results)
                    total = len(self.file_items)
                    self.progress_bar.set(processed / total if total > 0 else 0)
                    self.status_label.configure(text=f"処理中... {processed}/{total}")

                elif msg_type == 'done':
                    self._analysis_complete()
                    return

                elif msg_type == 'stopped':
                    self._analysis_stopped()
                    return

        except queue.Empty:
            pass

        if self.processing:
            self.root.after(100, self._poll_results)

    def _update_tree_item(self, item_id, result: ScoringResult):
        """Update a treeview row with scored result."""
        self.results[item_id] = result

        self.tree.item(item_id, values=(
            f"{result.verdict_icon} {result.verdict}",
            f"{result.total_score:.1f}",
            result.primary_style,
            result.gender,
            result.image_style,
            f"{result.engine_scores.get('nudenet', 0):.1f}",
            f"{result.engine_scores.get('wd14', 0):.1f}",
            f"{result.engine_scores.get('vision_api', 0):.1f}",
            f"{result.engine_scores.get('vit_nsfw', 0):.1f}",
            f"{result.engine_scores.get('lfm_vl', 0):.1f}",
            result.labels_summary
        ))

        # Color-code the row
        verdict = result.verdict
        tag_name = f"tag_{item_id}"
        colors = {
            'SAFE': '#1a3a1a', 'LOW_RISK': '#3a3a1a', 'MODERATE': '#3a2a1a',
            'HIGH_RISK': '#3a1a1a', 'UNSAFE': '#4a1a1a', 'ERROR': '#2a2a2a'
        }
        bg = colors.get(verdict, '#0d1117')
        self.tree.tag_configure(tag_name, background=bg)
        self.tree.item(item_id, tags=(tag_name,))

        # Store for export
        path = self.file_items.get(item_id)
        self.result_dicts.append({
            'filename': path.name if path else '',
            'path': str(path) if path else '',
            'verdict': result.verdict,
            'verdict_icon': result.verdict_icon,
            'total_score': result.total_score,
            'primary_style': result.primary_style,
            'gender': result.gender,
            'image_style': result.image_style,
            'engine_scores': result.engine_scores,
            'labels_summary': result.labels_summary,
            'all_tags': result.all_tags,
            'safe_search': result.safe_search,
            'vit_label': result.vit_label,
            'vit_nsfw_score': result.vit_nsfw_score,
            'lfm_safety_level': result.lfm_safety_level,
            'lfm_nsfw_score': result.lfm_nsfw_score,
            'lfm_description': result.lfm_description
        })

    def _analysis_complete(self):
        self.processing = False
        self.scan_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_bar.set(1.0)
        total = len(self.results)
        unsafe_count = sum(1 for r in self.results.values() if r.verdict == 'UNSAFE')
        self.status_label.configure(
            text=f"完了! {total}ファイル処理済み | 🔴 UNSAFE: {unsafe_count}"
        )

    def _analysis_stopped(self):
        self.processing = False
        self.scan_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text=f"停止しました ({len(self.results)}/{len(self.file_items)})")

    # ─── Preview ───

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item_id = sel[0]
        path = self.file_items.get(item_id)
        result = self.results.get(item_id)

        if path and path.exists():
            self._show_preview(path)

        if result:
            self._show_detail(result, path)

    def _show_preview(self, path: Path):
        """Display image preview with fit-to-frame scaling."""
        try:
            with open(path, 'rb') as f:
                data = f.read()
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return

            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)

            # Fit to 460x300
            max_w, max_h = 460, 300
            pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            tk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
            self.preview_label.configure(image=tk_img, text="")
            self.preview_label._image = tk_img  # Prevent GC
        except Exception as e:
            self.preview_label.configure(text=f"プレビュー失敗: {e}", image=None)

    def _show_detail(self, result: ScoringResult, path: Path = None):
        """Show detailed analysis in the text area."""
        self.detail_text.delete("0.0", "end")
        lines = []
        lines.append(f"═══ 分析結果 ═══")
        if path:
            lines.append(f"ファイル: {path.name}")
        lines.append(f"")
        lines.append(f"【総合判定】 {result.verdict_icon} {result.verdict}  (スコア: {result.total_score:.1f})")
        lines.append(f"【スタイル】 {result.primary_style}")
        lines.append(f"【性別】    {result.gender}")
        lines.append(f"【画風】    {result.image_style}")
        lines.append(f"")
        lines.append(f"─── エンジン別スコア ───")
        for eng, score in result.engine_scores.items():
            name_map = {
                'nudenet': 'NudeNet v3',
                'wd14': 'WD14-Tagger V3',
                'vision_api': 'Vision API',
                'vit_nsfw': 'ViT NSFW',
                'lfm_vl': 'LFM2.5-VL',
                'anime_cls': 'Anime Cls'
            }
            lines.append(f"  {name_map.get(eng, eng):16s}: {score:6.1f}")
        lines.append(f"")

        if result.safe_search:
            lines.append(f"─── Vision API SafeSearch ───")
            for k, v in result.safe_search.items():
                lines.append(f"  {k:12s}: {v}")
            lines.append(f"")

        if result.vit_label:
            lines.append(f"─── ViT NSFW ───")
            lines.append(f"  Label : {result.vit_label}")
            lines.append(f"  Score : {result.vit_nsfw_score:.4f}")
            lines.append(f"")

        lines.append(f"─── NudeNet 検出ラベル ───")
        lines.append(f"  {result.labels_summary}")
        lines.append(f"")

        if result.all_tags:
            lines.append(f"─── WD14 タグ (Top) ───")
            lines.append(f"  {result.all_tags}")
            lines.append(f"")

        if result.lfm_safety_level:
            lines.append(f"─── LFM2.5-VL 分析 ───")
            lines.append(f"  Safety : {result.lfm_safety_level}")
            lines.append(f"  Score  : {result.lfm_nsfw_score:.4f}")
            if result.lfm_description:
                lines.append(f"  Detail : {result.lfm_description}")

        self.detail_text.insert("0.0", '\n'.join(lines))

    def _clear_preview(self):
        self.preview_label.configure(text="画像を選択してください", image=None)
        self.detail_text.delete("0.0", "end")

    # ─── Export ───

    def _export(self, fmt: str):
        if not self.result_dicts:
            return

        ext_map = {'csv': '.csv', 'json': '.json', 'html': '.html'}
        ext = ext_map.get(fmt, '.csv')
        output = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} files", f"*{ext}")],
            title=f"{fmt.upper()} エクスポート"
        )
        if not output:
            return

        try:
            if fmt == 'csv':
                export_csv(self.result_dicts, output)
            elif fmt == 'json':
                export_json(self.result_dicts, output)
            elif fmt == 'html':
                export_html(self.result_dicts, output)
            self.status_label.configure(text=f"エクスポート完了: {output}")
        except Exception as e:
            self.status_label.configure(text=f"エクスポートエラー: {e}")

    # ─── Resource Monitor ───

    def _start_resource_monitor(self):
        self._update_resource_usage()

    def _update_resource_usage(self):
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory().percent

            gpu_text = ""
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    gpu_text = f" | GPU: {g.load*100:.0f}% | VRAM: {g.memoryUsed:.0f}/{g.memoryTotal:.0f}MB"
            except Exception:
                pass

            self.resource_label.configure(text=f"CPU: {cpu:.0f}% | MEM: {mem:.0f}%{gpu_text}")
        except Exception:
            pass

        self.root.after(3000, self._update_resource_usage)


def launch():
    """Launch the application."""
    root = ctk.CTk()
    app = NSFWCheckerApp(root)
    root.mainloop()
