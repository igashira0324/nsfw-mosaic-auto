# -*- coding: utf-8 -*-
"""
NSFW Image Checker - Premium GUI
CustomTkinterベースのモダンなGUIインターフェース
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import threading
import queue
import time
import csv
import json
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import API_KEY, SUPPORTED_EXTENSIONS, VERDICT_ICONS
from vision_client import VisionClient, VisionAPIError
from scorer import Scorer, ScoringResult
from file_handler import FileHandler

# デザイン設定
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ReferenceWindow(ctk.CTkToplevel):
    """グラフィカルな早見表ウィンドウ (タブ付き・詳細解説・参考URL付き)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("NSFW 判定基準・詳細ガイド")
        self.geometry("1100x850")
        self.attributes("-topmost", True)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # タブの作成
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.tabview.add("総合判定 & 活用シーン")
        self.tabview.add("カテゴリ解説 & 経緯")
        self.tabview.add("精度と信頼性指標")

        self._setup_verdict_tab()
        self._setup_category_tab()
        self._setup_accuracy_tab()

    def _add_url_section(self, parent, title, url_list):
        """参考URLセクションを追加"""
        frame = ctk.CTkFrame(parent, fg_color="#2c3e50")
        frame.pack(fill="x", padx=10, pady=20)
        
        ctk.CTkLabel(frame, text=f"🔗 {title}", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        for label, url in url_list:
            link = ctk.CTkLabel(frame, text=f"・{label}", text_color="#3498db", cursor="hand2", font=ctk.CTkFont(size=12, underline=True))
            link.pack(anchor="w", padx=20, pady=2)
            link.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

    def _setup_verdict_tab(self):
        tab = self.tabview.tab("総合判定 & 活用シーン")
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll_frame, text="総合スコアによる判定区分と運用ガイド", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 20))

        # テーブル
        table_main = ctk.CTkFrame(scroll_frame)
        table_main.pack(fill="x", padx=10, pady=10)

        headers = ["区分名", "スコア", "カラー", "想定される活用シーン / システム運用"]
        h_frame = ctk.CTkFrame(table_main, fg_color="#333")
        h_frame.pack(fill="x")
        ctk.CTkLabel(h_frame, text=headers[0], font=ctk.CTkFont(weight="bold"), width=100).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(h_frame, text=headers[1], font=ctk.CTkFont(weight="bold"), width=100).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(h_frame, text=headers[2], font=ctk.CTkFont(weight="bold"), width=100).grid(row=0, column=2, padx=5, pady=5)
        ctk.CTkLabel(h_frame, text=headers[3], font=ctk.CTkFont(weight="bold"), width=550, anchor="w").grid(row=0, column=3, padx=5, pady=5, sticky="w")

        criteria = [
            ("SAFE", "0-20", "#2ecc71", "✅ 一般公開。SNS等での無制限な表示に適しています。"),
            ("LOW_RISK", "20-40", "#f1c40f", "⚠️ 注意。センシティブ設定、年齢確認の導入検討。"),
            ("MODERATE", "40-60", "#f39c12", "⚠️ 警告。事前ぼかし（スプーラー）の適用推奨。"),
            ("HIGH_RISK", "60-80", "#e67e22", "🔶 制限。手動検閲への回送、または限定公開設定。"),
            ("UNSAFE", "80-100", "#e74c3c", "🔴 遮断。自動削除、または即時の非表示化。")
        ]

        for r, (verdict, score, col, scene) in enumerate(criteria):
            f = ctk.CTkFrame(table_main, fg_color="transparent")
            f.pack(fill="x")
            ctk.CTkLabel(f, text=verdict, width=100, text_color=col, font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=5)
            ctk.CTkLabel(f, text=score, width=100).grid(row=0, column=1, padx=5, pady=5)
            tk.Label(f, bg=col, width=3).grid(row=0, column=2, padx=30, pady=5)
            ctk.CTkLabel(f, text=scene, width=550, anchor="w").grid(row=0, column=3, padx=5, pady=5, sticky="w")

        usage_text = """
【判定ロジックの活用背景】
この指標は、大量の生成画像や投稿データを「機械的に前捌き」するために開発されました。
スコア20以下を「完全ホワイト」として自動承認し、60以上を「要確認・ブラック候補」として
人間の検閲リソースを集中させることで、安全性と運用コストのバランスを最適化します。
        """
        ctk.CTkLabel(scroll_frame, text=usage_text, justify="left", font=ctk.CTkFont(size=12), text_color="#bdc3c7").pack(pady=10, padx=20, anchor="w")

        self._add_url_section(scroll_frame, "活用例・参考URL", [
            ("Google Cloud SafeSearch 概要", "https://cloud.google.com/vision/docs/detecting-safe-search"),
            ("SafeSearch 判定のチュートリアル", "https://cloud.google.com/vision/docs/detecting-safe-search#vision_safe_search_detection-python")
        ])

    def _setup_category_tab(self):
        tab = self.tabview.tab("カテゴリ解説 & 経緯")
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll_frame, text="判定項目の詳細と重み付けの意図", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 20))

        info_frame = ctk.CTkFrame(scroll_frame)
        info_frame.pack(fill="x", padx=10, pady=10)

        # フィールド項目名 (ヘッダー)
        headers = ["判定項目", "重み係数", "評価される内容と重み付けの理由"]
        h_frame = ctk.CTkFrame(info_frame, fg_color="#333")
        h_frame.pack(fill="x")
        ctk.CTkLabel(h_frame, text=headers[0], font=ctk.CTkFont(weight="bold"), width=120).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(h_frame, text=headers[1], font=ctk.CTkFont(weight="bold"), width=100).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(h_frame, text=headers[2], font=ctk.CTkFont(weight="bold"), width=600, anchor="w").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        cats = [
            ("Adult", "1.5", "性的描写。コミュニティ規定で最も厳格に制限されるため、最大級の重み。"),
            ("Violence", "1.2", "暴力・残酷表現。不快感を与えるリスクが高いため、二番目に重視。"),
            ("Racy", "1.0", "露出・挑発。広告基準など「グレーゾーン」の判定に用いられる標準重み。"),
            ("Medical", "0.5", "医療行為。学習データ選別等では許容されることが多いため、半分に軽減。"),
            ("Spoof", "0.3", "パロディ・加工・コラ画像。ジョーク要素も多いため、最も低い評価重みに設定。")
        ]

        for i, (name, weight, reason) in enumerate(cats):
            f = ctk.CTkFrame(info_frame, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=name, width=120, font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=10)
            ctk.CTkLabel(f, text=f"× {weight}", width=100, text_color="#3498db").grid(row=0, column=1, padx=5, pady=10)
            ctk.CTkLabel(f, text=reason, width=600, anchor="w").grid(row=0, column=2, padx=5, pady=10, sticky="w")

        context_text = """
【重み付けの経緯】
Vision APIのSafeSearchは、単一の項目が高いだけで「不適切」と判定される傾向があります。
しかし、実際の検証フロー（例：AI学習データ選別）では、
「医療画像のグロテスクさ」と「成人向け描写」は区別して扱う必要があります。
本ツールでは、これらの差分を「重み係数」によって正規化し、
人間に近い感覚で「どの程度不快・危険か」を一つの数字で表現できるよう調整されています。
        """
        ctk.CTkLabel(scroll_frame, text=context_text, justify="left", font=ctk.CTkFont(size=12), text_color="#bdc3c7").pack(pady=10, padx=20, anchor="w")

        self._add_url_section(scroll_frame, "技術詳細・参考URL", [
            ("Vision API Categories (REST)", "https://cloud.google.com/vision/docs/reference/rest/v1/AnnotateImageResponse#SafeSearchAnnotation"),
            ("AIにおける安全性基準の考え方", "https://ai.google/principles/")
        ])

    def _setup_accuracy_tab(self):
        tab = self.tabview.tab("精度と信頼性指標")
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(scroll_frame, text="Likelihood値の精度特性と検証への活用", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(10, 20))

        stats_frame = ctk.CTkFrame(scroll_frame)
        stats_frame.pack(fill="x", padx=10, pady=10)

        # フィールド項目名
        headers = ["可能性指標", "信頼度(推計)", "信頼度に基づく推奨アクション"]
        h_frame = ctk.CTkFrame(stats_frame, fg_color="#333")
        h_frame.pack(fill="x")
        ctk.CTkLabel(h_frame, text=headers[0], font=ctk.CTkFont(weight="bold"), width=150).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(h_frame, text=headers[1], font=ctk.CTkFont(weight="bold"), width=100).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(h_frame, text=headers[2], font=ctk.CTkFont(weight="bold"), width=600, anchor="w").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        levels = [
            ("VERY_LIKELY", "95%+", "極めて高精度。即時遮断の根拠として信頼できます。"),
            ("LIKELY", "80%+", "高い確信。システム的な自動フィルタリングが有効。"),
            ("POSSIBLE", "50%+", "境界線。多くはグレーゾーンで、人間による二次確認を推奨。"),
            ("UNLIKELY", "20%以下", "ほぼ安全。まれな背景誤認を除き、パスさせて問題なし。"),
            ("VERY_UNLIKELY", "5%以下", "極めて安全。意図的に隠された要素以外は考慮不要。")
        ]

        for i, (lvl, acc, guide) in enumerate(levels):
            f = ctk.CTkFrame(stats_frame, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=lvl, width=150, font=ctk.CTkFont(family="Consolas")).grid(row=0, column=0, padx=5, pady=10, sticky="w")
            ctk.CTkLabel(f, text=acc, width=100, text_color="#1abc9c", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, padx=5, pady=10)
            ctk.CTkLabel(f, text=guide, width=600, anchor="w").grid(row=0, column=2, padx=5, pady=10, sticky="w")

        validation_guide = """
【実際の検証と活用シーン】
1. **データセットの監査**: 数万枚の画像をスクリーニングし、不適切なものが混入していないかの「合格証」としてスコアを活用。
2. **モデル出力の検証**: 画像生成モデルが、どの程度の確率で不適切な画像を生成しうるかの統計的検証に使用。
3. **ワークフローの自動化**:
   - スコア < 20: ワークフロー継続
   - スコア 20-60: 保留フォルダへ移動し、人間に通知
   - スコア > 60: 即時廃棄・アラート
        """
        ctk.CTkLabel(scroll_frame, text=validation_guide, justify="left", font=ctk.CTkFont(size=12), text_color="#bdc3c7").pack(pady=10, padx=20, anchor="w")

        self._add_url_section(scroll_frame, "精度検証・参考URL", [
            ("Vision API リリースノート (最新情報)", "https://cloud.google.com/vision/docs/release-notes"),
            ("画像生成AI(Stable Diffusion)と自動検閲の例", "https://github.com/AUTOMATIC1111/stable-diffusion-webui")
        ])

class NSFWCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NSFW Image Checker Pro")
        self.root.geometry("1200x800")

        # インスタンス
        self.client = VisionClient()
        self.scorer = Scorer()
        self.file_handler = FileHandler()
        
        self.processing_queue = queue.Queue()
        self.is_running = False
        self.results = []
        
        self._setup_layout()
        self._check_api_key()

    def _setup_layout(self):
        # グリッド設定
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- サイドバー ---
        self.sidebar_frame = ctk.CTkFrame(self.root, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="NSFW Checker", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.select_file_btn = ctk.CTkButton(self.sidebar_frame, text="ファイルを個別に選択", command=self._select_files)
        self.select_file_btn.grid(row=1, column=0, padx=20, pady=10)

        self.select_folder_btn = ctk.CTkButton(self.sidebar_frame, text="フォルダを一括選択", command=self._select_folder)
        self.select_folder_btn.grid(row=2, column=0, padx=20, pady=10)

        self.recursive_switch = ctk.CTkSwitch(self.sidebar_frame, text="サブフォルダも含める")
        self.recursive_switch.grid(row=3, column=0, padx=20, pady=10)

        # 説明文 (もっと簡潔に分かりやすく)
        self.recursive_info = ctk.CTkLabel(self.sidebar_frame, text="子フォルダ内の画像も\nスキャン対象に含めます", 
                                          font=ctk.CTkFont(size=11), text_color="#95a5a6", justify="left")
        self.recursive_info.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="nw")

        self.ref_btn = ctk.CTkButton(self.sidebar_frame, text="判定基準（活用ガイド）", fg_color="#34495e", hover_color="#5d6d7e", command=self._show_reference)
        self.ref_btn.grid(row=5, column=0, padx=20, pady=10)

        self.export_btn = ctk.CTkButton(self.sidebar_frame, text="結果をエクスポート", fg_color="#2c3e50", command=self._export_results)
        self.export_btn.grid(row=6, column=0, padx=20, pady=(10, 20))

        # --- メインコンテンツ ---
        self.main_content = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(2, weight=1)

        # 操作ボタン
        self.top_ctrl = ctk.CTkFrame(self.main_content, height=50, fg_color="transparent")
        self.top_ctrl.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        self.start_btn = ctk.CTkButton(self.top_ctrl, text="▶ 分析を開始する", font=ctk.CTkFont(size=15, weight="bold"), 
                                      fg_color="#27ae60", hover_color="#2ecc71", command=self._start_analysis)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ctk.CTkButton(self.top_ctrl, text="リストをクリア", fg_color="#c0392b", hover_color="#e74c3c", command=self._clear_list)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.deselect_btn = ctk.CTkButton(self.top_ctrl, text="選択解除", fg_color="#7f8c8d", hover_color="#95a5a6", command=self._deselect_all)
        self.deselect_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ctk.CTkLabel(self.top_ctrl, text="ステータス: 待機中", text_color="gray")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # 進捗バー
        self.progress_bar = ctk.CTkProgressBar(self.main_content)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        self.progress_bar.set(0)

        # テーブル
        self.table_frame = tk.Frame(self.main_content, bg="#2b2b2b")
        self.table_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # CustomTkinter環境でのttkスタイル設定
        try:
            style = ttk.Style()
            style.theme_use("default")
            style.configure("Treeview", 
                            background="#2b2b2b", 
                            foreground="white", 
                            fieldbackground="#2b2b2b", 
                            borderwidth=0,
                            rowheight=35)
            style.map("Treeview", background=[('selected', '#3498db')])
            style.configure("Treeview.Heading", background="#333", foreground="white", relief="flat")
        except Exception as e:
            print(f"Warning: Could not set ttk styles: {e}")

        # 日本語カラム名
        columns = ("Filename", "Score", "Verdict", "Adult", "Racy", "Violence", "Medical", "Spoof", "Description")
        display_names = ("ファイル名", "スコア", "判定", "Adult", "Racy", "Violence", "Medical", "Spoof", "画像の内容 (Labels)")
        
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show='headings', selectmode="extended")

        # カラム設定
        col_widths = {"Filename": 180, "Score": 70, "Verdict": 100, "Adult": 70, "Racy": 70, "Violence": 70, "Medical": 70, "Spoof": 70, "Description": 220}
        for i, col in enumerate(columns):
            self.tree.heading(col, text=display_names[i])
            self.tree.column(col, width=col_widths[col], anchor=tk.CENTER if i > 0 and i < 8 else tk.W)

        scrollbar = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 右クリックメニュー
        self.context_menu = tk.Menu(self.root, tearoff=0, bg="#333", fg="white", activebackground="#3498db")
        self.context_menu.add_command(label="選択を解除", command=self._deselect_all)
        self.context_menu.add_command(label="選択した項目をリストから削除", command=self._clear_list)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="選択した項目のみエクスポート", command=self._export_results)

        self.tree.bind("<Button-3>", self._show_context_menu)
        # 空白部分クリックで解除
        self.tree.bind("<Button-1>", self._on_tree_click)

        # タグ設定（カラー化を強化）
        self.tree.tag_configure('SAFE', foreground='#2ecc71')
        self.tree.tag_configure('LOW_RISK', foreground='#f1c40f')
        self.tree.tag_configure('MODERATE', foreground='#f39c12')
        self.tree.tag_configure('HIGH_RISK', foreground='#e67e22')
        self.tree.tag_configure('UNSAFE', foreground='#e74c3c', font=('Helvetica', 9, 'bold'))
        self.tree.tag_configure('ERROR', foreground='gray')

    def _check_api_key(self):
        if not API_KEY or API_KEY == "PASTE_YOUR_API_KEY_HERE":
            messagebox.showwarning("APIキー未設定", "config.py 内に有効な Google Cloud Vision API キーが設定されていません。")

    def _select_files(self):
        files = filedialog.askopenfilenames(
            title="分析する画像を選択",
            filetypes=[("画像ファイル", "*.jpg *.jpeg *.png *.gif *.webp *.bmp")]
        )
        if files:
            for f in files:
                self.tree.insert("", tk.END, values=(Path(f).name, "-", "-", "-", "-", "-", "-", "-", "-", f))
            self.status_label.configure(text=f"準備完了: {len(self.tree.get_children())} 個のファイルを読み込み済み")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="分析するフォルダを選択")
        if folder:
            images = self.file_handler.collect_images(Path(folder), self.recursive_switch.get())
            for f in images:
                self.tree.insert("", tk.END, values=(f.name, "-", "-", "-", "-", "-", "-", "-", "-", str(f)))
            self.status_label.configure(text=f"準備完了: {len(self.tree.get_children())} 枚の画像をフォルダから読み込み済み")

    def _deselect_all(self):
        self.tree.selection_remove(self.tree.selection())

    def _show_context_menu(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            if item not in self.tree.selection():
                self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)

    def _on_tree_click(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            self._deselect_all()

    def _clear_list(self):
        if self.is_running: return
        
        selected = self.tree.selection()
        if selected:
            # 選択された項目のみ削除
            for item in selected:
                self.tree.delete(item)
            # 内部データ（解析結果）も同期
            self.results = [r for r in self.results if r.get('id') not in selected]
            self.status_label.configure(text=f"ステータス: {len(selected)} 個の項目を削除しました")
        else:
            # 選択がない場合は全削除（従来通り）
            items = self.tree.get_children()
            if not items:
                messagebox.showinfo("お知らせ", "画像データがありません。")
                return
            
            if not messagebox.askyesno("確認", "リストのすべての項目をクリアしますか？"):
                return

            for item in items:
                self.tree.delete(item)
            self.results = []
            self.progress_bar.set(0)
            self.status_label.configure(text="ステータス: 待機中")

    def _show_reference(self):
        ReferenceWindow(self.root)

    def _start_analysis(self):
        if self.is_running: return
        items = self.tree.get_children()
        if not items: 
            messagebox.showinfo("お知らせ", "画像データがありません。")
            return

        self.is_running = True
        self.start_btn.configure(state="disabled", fg_color="gray")
        self.progress_bar.set(0)
        self.status_label.configure(text="ステータス: 分析中...")
        
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(100, self._process_queue)

    def _worker(self):
        items = self.tree.get_children()
        for idx, item in enumerate(items):
            if not self.is_running: break
            values = self.tree.item(item, 'values')
            image_path = Path(values[-1])
            try:
                result = self.client.analyze_image(image_path)
                score_result = self.scorer.score(result)
                self.processing_queue.put(('success', item, score_result, image_path))
            except Exception as e:
                self.processing_queue.put(('error', item, str(e), image_path))
            time.sleep(0.1)
        self.processing_queue.put(('done', None, None, None))

    def _process_queue(self):
        try:
            while True:
                msg_type, item, data, path = self.processing_queue.get_nowait()
                if msg_type == 'done':
                    self.is_running = False
                    self.start_btn.configure(state="normal", fg_color="#27ae60")
                    self.status_label.configure(text="ステータス: 完了")
                    return

                # 進捗更新
                total = len(self.tree.get_children())
                current = len(self.results) + 1
                self.progress_bar.set(current / total)
                
                if msg_type == 'success':
                    sr = data
                    cats = sr.categories
                    self.tree.item(item, values=(
                        path.name, sr.total_score, sr.verdict,
                        cats['adult'].likelihood, cats['racy'].likelihood, cats['violence'].likelihood,
                        cats['medical'].likelihood, cats['spoof'].likelihood, sr.description, str(path)
                    ), tags=(sr.verdict,))
                    self.results.append({
                        'id': item, 'filename': path.name, 'score': sr.total_score, 'verdict': sr.verdict, 
                        'desc': sr.description, 'categories': {c: vars(cr) for c, cr in cats.items()}, 'path': str(path)
                    })
                else:
                    self.tree.item(item, values=(path.name, "エラー", "Error", "-", "-", "-", "-", "-", "-", str(path)), tags=('ERROR',))
                    self.results.append({'id': item, 'filename': path.name, 'score': 0, 'verdict': 'ERROR', 'path': str(path)})

        except queue.Empty:
            pass
        if self.is_running or not self.processing_queue.empty():
            self.root.after(100, self._process_queue)

    def _export_results(self):
        selected = self.tree.selection()
        if selected:
            # 選択された項目のみエクスポート対象にする
            export_data = [r for r in self.results if r.get('id') in selected]
            if not export_data:
                messagebox.showinfo("お知らせ", "選択された項目のうち、分析済みのデータがありません。")
                return
            target_desc = f"選択した {len(export_data)} 件"
        else:
            # 選択がない場合は全エクスポート
            if not self.results: 
                messagebox.showinfo("お知らせ", "画像データ（分析結果）がありません。")
                return
            export_data = self.results
            target_desc = "すべての結果"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            filetypes=[("CSV (Excel読み込み可)", "*.csv"), ("JSON データ", "*.json")]
        )
        if not file_path: return
        try:
            p = Path(file_path)
            if p.suffix == '.json':
                with open(p, 'w', encoding='utf-8') as f: json.dump(export_data, f, ensure_ascii=False, indent=2)
            else:
                with open(p, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(["ファイル名", "総合スコア", "判定", "内容説明", "Adult", "Racy", "Violence", "Medical", "Spoof", "ファイルパス"])
                    for r in export_data:
                        if r['verdict'] == 'ERROR':
                            writer.writerow([r['filename'], 0, "ERROR", "", "", "", "", "", "", r['path']])
                            continue
                        cats = r['categories']
                        writer.writerow([r['filename'], r['score'], r['verdict'], r.get('desc', ''),
                                        cats['adult']['likelihood'], cats['racy']['likelihood'], cats['violence']['likelihood'],
                                        cats['medical']['likelihood'], cats['spoof']['likelihood'], r['path']])
            messagebox.showinfo("成功", f"{target_desc} を保存しました:\n{p.name}")
        except Exception as e: messagebox.showerror("エラー", f"出力に失敗しました: {str(e)}")

def launch_gui():
    root = ctk.CTk()
    app = NSFWCheckerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
