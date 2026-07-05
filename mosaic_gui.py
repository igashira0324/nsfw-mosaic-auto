# -*- coding: utf-8 -*-
"""
mosaic_gui.py — モザイクツール共通GUI部品 (tkinter)

パターン選択 / モード選択 / 進捗ウィンドウ(ETA+キャンセル対応) を提供する。
"""

import time
import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
from tkinter import ttk

from mosaic_core import PATTERNS, PATTERN_LABELS

BG = "#23272e"
BG_DARK = "#181a20"
FG = "#ffffff"
ACCENT = "#00bfff"
DANGER = "#ff5555"


def _style_button(btn: tk.Button, hover: str = ACCENT) -> None:
    btn.bind("<Enter>", lambda e: e.widget.config(bg=hover, fg=FG))
    btn.bind("<Leave>", lambda e: e.widget.config(bg=BG, fg=FG))


def ask_mosaic_pattern() -> str:
    """モザイクパターン選択ダイアログ。戻り値: 正準パターン名 / None(キャンセル)"""
    selected = [PATTERNS[0]]
    cancelled = [False]

    root = tk.Tk()
    root.title("モザイクパターン選択")
    root.geometry("640x420")
    root.configure(bg=BG)
    tk.Label(root, text="モザイクパターンを選択してください",
             font=("Segoe UI", 16, "bold"), bg=BG, fg=FG).pack(padx=10, pady=14)

    frame = tk.Frame(root, bg=BG)
    frame.pack(padx=24, pady=4, fill=tk.BOTH, expand=True)
    listbox = tk.Listbox(frame, height=len(PATTERNS), font=("Segoe UI", 13),
                         bg=BG_DARK, fg=FG, selectbackground=ACCENT,
                         selectforeground=FG, relief="flat",
                         highlightthickness=0, bd=0, activestyle="none")
    for p in PATTERNS:
        listbox.insert(tk.END, "  " + PATTERN_LABELS.get(p, p))
    listbox.selection_set(0)
    listbox.pack(fill=tk.BOTH, expand=True)

    def commit(_=None):
        idx = listbox.curselection()
        if idx:
            selected[0] = PATTERNS[idx[0]]
        root.quit()

    def cancel():
        cancelled[0] = True
        root.quit()

    listbox.bind("<Double-1>", commit)
    listbox.bind("<Return>", commit)

    btns = tk.Frame(root, bg=BG)
    btns.pack(pady=14)
    ok = tk.Button(btns, text="OK", font=("Segoe UI", 13), width=10, height=1,
                   bg=BG, fg=FG, relief="raised", bd=3,
                   activebackground=ACCENT, activeforeground=FG, command=commit)
    ok.pack(side=tk.LEFT, padx=14)
    _style_button(ok)
    ng = tk.Button(btns, text="キャンセル", font=("Segoe UI", 13), width=10, height=1,
                   bg=BG, fg=FG, relief="raised", bd=3,
                   activebackground=DANGER, activeforeground=FG, command=cancel)
    ng.pack(side=tk.LEFT, padx=14)
    _style_button(ng, DANGER)

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    root.destroy()
    return None if cancelled[0] else selected[0]


def ask_video_mode() -> str:
    """動画処理モード選択。戻り値: 'file' / 'folder' / None"""
    mode = {"value": None}

    root = tk.Tk()
    root.title("動画処理モード選択")
    root.geometry("480x260")
    root.configure(bg=BG)
    tk.Label(root, text="処理方法を選択してください",
             font=("Segoe UI", 17, "bold"), bg=BG, fg=FG).pack(pady=28)

    def pick(v):
        mode["value"] = v
        root.quit()

    b1 = tk.Button(root, text="動画ファイルを選択", font=("Segoe UI", 14), width=24, height=2,
                   bg=BG, fg=FG, relief="raised", bd=3,
                   activebackground=ACCENT, activeforeground=FG, command=lambda: pick("file"))
    b1.pack(pady=10)
    _style_button(b1)
    b2 = tk.Button(root, text="フォルダ内の全動画を一括処理", font=("Segoe UI", 14), width=28, height=2,
                   bg=BG, fg=FG, relief="raised", bd=3,
                   activebackground=ACCENT, activeforeground=FG, command=lambda: pick("folder"))
    b2.pack(pady=10)
    _style_button(b2)

    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
    root.destroy()
    return mode["value"]


def pick_file(title: str, filetypes) -> str:
    root = tk.Tk()
    root.withdraw()
    path = tkFileDialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def pick_folder(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    folder = tkFileDialog.askdirectory(title=title)
    root.destroy()
    return folder


def show_info(title: str, msg: str) -> None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    tkMessageBox.showinfo(title, msg, parent=root)
    root.destroy()


def show_error(title: str, msg: str) -> None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    tkMessageBox.showerror(title, msg, parent=root)
    root.destroy()


def ask_yesno(title: str, msg: str) -> bool:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    res = tkMessageBox.askyesno(title, msg, parent=root)
    root.destroy()
    return res


class ProgressWindow:
    """ETA表示とキャンセルボタン付き進捗ウィンドウ。

    使い方:
        pw = ProgressWindow("動画モザイク処理")
        pw.update("検出パス", cur, total, extra="file.mp4")
        if pw.cancelled: ...
        pw.close()
    """

    def __init__(self, title: str, accent: str = ACCENT):
        self.cancelled = False
        self._stage = None
        self._t0 = time.time()

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("480x230")
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.layout("Mosaic.Horizontal.TProgressbar",
                     [("Horizontal.Progressbar.trough",
                       {"children": [("Horizontal.Progressbar.pbar",
                                      {"side": "left", "sticky": "ns"})],
                        "sticky": "nswe"})])
        style.configure("Mosaic.Horizontal.TProgressbar",
                        troughcolor=BG_DARK, bordercolor=BG, background=accent,
                        lightcolor=accent, darkcolor=accent,
                        thickness=22, borderwidth=2, relief="flat")

        self.title_label = tk.Label(self.root, text=title, font=("Segoe UI", 14, "bold"), bg=BG, fg=FG)
        self.title_label.pack(pady=(14, 6))
        self.progress_var = tk.DoubleVar()
        self.bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100,
                                   length=420, style="Mosaic.Horizontal.TProgressbar")
        self.bar.pack(pady=6)
        self.status_label = tk.Label(self.root, text="", font=("Segoe UI", 11), bg=BG, fg=FG)
        self.status_label.pack(pady=2)
        self.eta_label = tk.Label(self.root, text="", font=("Segoe UI", 11), bg=BG, fg="#9adcff")
        self.eta_label.pack(pady=2)

        self.cancel_btn = tk.Button(self.root, text="キャンセル", font=("Segoe UI", 11),
                                    width=12, bg=BG, fg=FG, relief="raised", bd=2,
                                    activebackground=DANGER, activeforeground=FG,
                                    command=self._cancel)
        self.cancel_btn.pack(pady=8)
        _style_button(self.cancel_btn, DANGER)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)
        self.root.update()

    def _cancel(self):
        self.cancelled = True
        self.status_label.config(text="キャンセル中...")
        try:
            self.root.update()
        except Exception:
            pass

    def update(self, stage: str, cur: int, total: int, extra: str = "") -> None:
        if self.cancelled:
            return
        if stage != self._stage:
            self._stage = stage
            self._t0 = time.time()
            self.title_label.config(text=stage)
        try:
            if total and total > 0:
                pct = min(100.0, cur * 100.0 / total)
                self.progress_var.set(pct)
                elapsed = time.time() - self._t0
                eta_txt = ""
                if cur > 3 and elapsed > 2:
                    remain = elapsed / cur * (total - cur)
                    m, s = divmod(int(remain), 60)
                    fps = cur / elapsed
                    eta_txt = f"残り約 {m}分{s:02d}秒 ({fps:.1f} fps)"
                self.status_label.config(text=f"{extra}  {cur}/{total} ({pct:.0f}%)".strip())
                self.eta_label.config(text=eta_txt)
            else:
                self.progress_var.set(0)
                self.status_label.config(text=f"{extra}  {cur} フレーム処理済み".strip())
                self.eta_label.config(text="")
            self.root.update()
        except tk.TclError:
            self.cancelled = True

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
