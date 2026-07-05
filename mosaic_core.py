# -*- coding: utf-8 -*-
"""
mosaic_core.py — NSFW自動モザイク 共通コアエンジン (2026-07 全面刷新)

mosaic-image.py / mosaic-video.py / mosaic-video-speek.py から共通利用される。

主な機能:
  * 審査基準準拠モザイク: セル寸法 = max(4px, ceil(画像長辺 / 100)) (FANZA/審査団体デファクト)
    - グリッドを画像原点に整列させ、動画でセルが「這う」ちらつきを防止
  * マルチモデル検出アンサンブル: EraX YOLO11m (V1.0 + Anti-NSFW V1.1) + NudeNet 640m/320n
    - NudeNet は ndarray 直接入力 (旧実装の一時JPEG書き出しを廃止)
    - NudeNet box は [x, y, w, h] 形式 (旧実装の xyxy 誤解釈バグを修正)
    - onnxruntime CUDA が使えれば NudeNet も GPU 実行
  * 動画 2パス方式:
    Pass1: BoT-SORT トラッキング検出 + シーンカット検出
    時系列後処理: 検出ギャップ線形補間 / リードイン・アウト延長 / 移動平均平滑化(和集合保証)
    Pass2: モザイク適用 + ffmpeg パイプ直接エンコード (libx264 CRF18, 二重再エンコード排除)
  * 静止画: TTA / 高解像度タイル推論 / MobileSAM による箱→マスク精密モザイク
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "mosaic_config.json")
TRACKER_YAML = os.path.join(BASE_DIR, "mosaic_botsort.yaml")

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "detector": {
        # 存在するものだけ読み込まれる。先頭がトラッキング用プライマリ。
        "yolo_models": [
            "erax_nsfw_yolo11m.pt",
            "models/erax-anti-nsfw-yolo11m-v1.1.pt",
        ],
        "yolo_conf": 0.10,
        "yolo_iou": 0.45,
        "imgsz": 0,               # 0 = 自動 (長辺<1100:640 / それ以上:1280)
        "half": True,             # CUDA時にFP16推論
        "tta_images": True,       # 静止画のみ Test-Time Augmentation
        "tile_min_side": 2048,    # 長辺がこの値以上の静止画はタイル推論を併用 (0で無効)
        "tile_size": 1024,
        "tile_overlap": 0.2,
        "target_classes": ["anus", "penis", "vagina"],
        "nudenet": {
            "enabled": True,
            "model_path": "models/nudenet_640m.onnx",   # 無ければ同梱320nへ自動フォールバック
            "conf": 0.30,
            "labels": [
                "FEMALE_GENITALIA_EXPOSED",
                "MALE_GENITALIA_EXPOSED",
                "ANUS_EXPOSED",
            ],
            "frame_interval": 1,  # 動画でNフレーム毎に実行 (CPU実行時は自動で間引き)
        },
    },
    "mosaic": {
        "cell_divisor": 100,      # セル = ceil(長辺 / divisor)
        "min_cell": 4,            # 最小 4x4 px (業界基準)
        "strength": {"モザイク標準": 1.0, "モザイク強": 1.5, "モザイク特大": 2.0},
        # 検出箱→モザイク領域の縮小率 (EraXの箱は実領域より大きめのため)。既存チューニング値を踏襲
        "shrink_ratios": {
            "penis": [0.70, 0.55],
            "vagina": [0.70, 0.55],
            "anus": [0.65, 0.65],
            "default": [0.60, 0.50],
        },
        "mask_mode_image": True,  # 静止画: MobileSAMで箱→マスク精密化 (失敗時は矩形)
        "mask_model": "models/mobile_sam.pt",
        "min_box_px": 10,
    },
    "video": {
        "tracker_yaml": "mosaic_botsort.yaml",
        "max_gap_frames": 20,     # このフレーム数以下の検出穴は線形補間で埋める
        "pad_frames": 4,          # 各トラックの前後に延長するフレーム数 (リードイン/アウト)
        "smooth_window": 5,       # 移動平均平滑化の窓幅 (奇数)
        "scene_cut_threshold": 0.45,  # HSVヒスト相関がこれ未満でシーンカット扱い
        "crf": 18,
        "preset": "medium",
        "audio_bitrate": "192k",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    """mosaic_config.json があれば既定値にマージ。無ければ既定値で自動生成する。"""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user = json.load(f)
            cfg = _deep_merge(cfg, user)
        except Exception as e:
            print(f"[WARNING] mosaic_config.json の読み込みに失敗 (既定値を使用): {e}")
    else:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return cfg


# ---------------------------------------------------------------------------
# モザイクパターン
# ---------------------------------------------------------------------------

# 正準パターン名
PATTERNS = ["モザイク標準", "モザイク強", "モザイク特大", "ぼかし", "黒塗り"]

# GUI表示用ラベル
PATTERN_LABELS = {
    "モザイク標準": "モザイク標準（審査基準: 長辺1/100・最小4px）",
    "モザイク強": "モザイク強（基準×1.5 / AI復元耐性重視）",
    "モザイク特大": "モザイク特大（基準×2.0）",
    "ぼかし": "ぼかし（強ガウス / 参考: 復元耐性はモザイクより低め）",
    "黒塗り": "黒塗り（完全不可逆）",
}

# 旧バージョンのパターン名からの互換マッピング
PATTERN_ALIASES = {
    "モザイク小": "モザイク標準",
    "モザイク中": "モザイク強",
    "モザイク大": "モザイク特大",
}


def normalize_pattern(pattern: str) -> str:
    return PATTERN_ALIASES.get(pattern, pattern)


def compute_cell(frame_w: int, frame_h: int, pattern: str, cfg: dict) -> int:
    """審査基準準拠のモザイクセル寸法(px)を画像全体サイズから算出する。"""
    m = cfg["mosaic"]
    base = max(int(m["min_cell"]), math.ceil(max(frame_w, frame_h) / float(m["cell_divisor"])))
    mult = m["strength"].get(normalize_pattern(pattern), 1.0)
    return max(int(m["min_cell"]), int(math.ceil(base * mult)))


# ---------------------------------------------------------------------------
# 幾何ヘルパ
# ---------------------------------------------------------------------------

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_containment(a: Box, b: Box) -> float:
    """交差面積 / 小さい方の面積 (片方がほぼ内包される場合に高くなる)"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    mn = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / mn if mn > 0 else 0.0


def box_union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def merge_boxes_union(boxes: List[Box], iou_thr: float = 0.25, cont_thr: float = 0.60) -> List[Box]:
    """重なる箱を和集合に統合する (被覆を減らさないマージ)。"""
    boxes = [tuple(int(v) for v in b) for b in boxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        out: List[Box] = []
        while boxes:
            cur = boxes.pop(0)
            merged_any = False
            for i, other in enumerate(boxes):
                if box_iou(cur, other) > iou_thr or box_containment(cur, other) > cont_thr:
                    boxes[i] = box_union(cur, other)
                    merged_any = True
                    changed = True
                    break
            if not merged_any:
                out.append(cur)
        boxes = out + boxes
    return boxes


def shrink_box(x1: float, y1: float, x2: float, y2: float, cls_name: str, cfg: dict) -> Optional[Box]:
    """検出箱をクラス別縮小率でモザイク領域へ変換する。"""
    m = cfg["mosaic"]
    w, h = x2 - x1, y2 - y1
    if w < m["min_box_px"] or h < m["min_box_px"]:
        return None
    ratios = m["shrink_ratios"]
    rw, rh = ratios.get(cls_name, ratios.get("default", [0.6, 0.5]))
    dx, dy = w * rw / 2.0, h * rh / 2.0
    sx1, sy1, sx2, sy2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    if sx2 - sx1 >= 2 and sy2 - sy1 >= 2:
        return (int(sx1), int(sy1), int(sx2), int(sy2))
    return None


def clip_box(b: Box, w: int, h: int) -> Optional[Box]:
    x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
    x2, y2 = min(int(w), int(b[2])), min(int(h), int(b[3]))
    if x2 - x1 >= 1 and y2 - y1 >= 1:
        return (x1, y1, x2, y2)
    return None


# ---------------------------------------------------------------------------
# 検出
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    cls: str
    conf: float
    box: Box                    # 縮小後 (モザイク適用領域)
    raw: Box                    # 生の検出箱 (マスク精密化のプロンプト等に使用)
    tid: Optional[tuple] = None  # トラックキー ('t', id) / None


# NudeNet ラベル → 縮小率参照用クラス名
_NUDENET_CLS_MAP = {
    "FEMALE_GENITALIA_EXPOSED": "vagina",
    "MALE_GENITALIA_EXPOSED": "penis",
    "ANUS_EXPOSED": "anus",
}


def _nms_dedup(dets: List[Detection], iou_thr: float = 0.55) -> List[Detection]:
    """同一クラスの重複検出を信頼度優先で間引く (TTA/タイル統合用)。"""
    out: List[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        dup = False
        for kept in out:
            if kept.cls == d.cls and box_iou(kept.raw, d.raw) > iou_thr:
                dup = True
                break
        if not dup:
            out.append(d)
    return out


class NsfwDetector:
    """YOLOアンサンブル + NudeNet を束ねる検出器。"""

    def __init__(self, cfg: dict, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.log = log
        d = cfg["detector"]

        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.cuda = torch.cuda.is_available()
        # ultralytics 8.4+ では half= が非推奨 (quantize= に統一)
        self.quantize = "fp16" if (bool(d.get("half", True)) and self.cuda) else None
        self.device = 0 if self.cuda else "cpu"

        # --- YOLO モデル群 ---
        self.models = []
        self.model_class_filters = []  # モデル毎の対象クラスindexリスト(Noneなら全クラス)
        target = set(d["target_classes"])
        for rel in d["yolo_models"]:
            path = rel if os.path.isabs(rel) else os.path.join(BASE_DIR, rel)
            if not os.path.exists(path):
                self.log(f"[INFO] YOLOモデルなし(スキップ): {rel}")
                continue
            try:
                m = YOLO(path)
                names = m.names if isinstance(m.names, dict) else {i: n for i, n in enumerate(m.names)}
                idxs = [i for i, n in names.items() if n in target]
                # 対象クラス名が全く合わない専用モデル(例: 単一nsfwクラス)は全クラス採用
                self.models.append((m, names))
                self.model_class_filters.append(idxs if idxs else None)
                self.log(f"[INFO] YOLOモデル読込: {os.path.basename(path)} classes={list(names.values())}")
            except Exception as e:
                self.log(f"[WARNING] YOLOモデル読込失敗 {rel}: {e}")
        if not self.models:
            raise RuntimeError("有効なYOLOモデルが1つもありません (erax_nsfw_yolo11m.pt を確認してください)")

        self.conf = float(d["yolo_conf"])
        self.iou = float(d["yolo_iou"])
        self.imgsz_cfg = int(d.get("imgsz", 0))

        # --- NudeNet ---
        self.nudenet = None
        self.nn_labels = set(d["nudenet"]["labels"])
        self.nn_conf = float(d["nudenet"]["conf"])
        self.nn_interval = max(1, int(d["nudenet"].get("frame_interval", 1)))
        if d["nudenet"].get("enabled", True):
            self._init_nudenet(d["nudenet"])

        # --- MobileSAM (静止画マスク精密化, 遅延ロード) ---
        self._sam = None
        self._sam_tried = False

        self._frame_no = 0

    # -- NudeNet -----------------------------------------------------------
    def _init_nudenet(self, nn_cfg: dict) -> None:
        try:
            from nudenet import NudeDetector
        except ImportError:
            self.log("[INFO] nudenet 未インストールのためクロスチェック層は無効")
            return
        model_path = nn_cfg.get("model_path", "")
        abs_path = model_path if os.path.isabs(model_path) else os.path.join(BASE_DIR, model_path)
        use_640 = os.path.exists(abs_path) and os.path.getsize(abs_path) > 50_000_000
        try:
            if use_640:
                self.nudenet = NudeDetector(model_path=abs_path, inference_resolution=640)
                which = "640m"
            else:
                self.nudenet = NudeDetector()
                which = "320n(同梱)"
            gpu = self._upgrade_nudenet_session(abs_path if use_640 else None)
            self.log(f"[INFO] NudeNet {which} 読込 ({'CUDA' if gpu else 'CPU'})")
            if not gpu:
                # CPU実行時は動画で自動間引き
                self.nn_interval = max(self.nn_interval, 2 if not use_640 else 3)
        except Exception as e:
            self.nudenet = None
            self.log(f"[WARNING] NudeNet 初期化失敗 (層を無効化): {e}")

    def _upgrade_nudenet_session(self, model_path: Optional[str]) -> bool:
        """nudenet 3.4.2 は providers 指定がコメントアウトされておりCPU実行になるため、
        CUDAが使える場合はセッションを差し替えてGPU化する。"""
        try:
            import onnxruntime as ort
            # torch 同梱の CUDA/cuDNN DLL を探索パスへ (Windows)
            if os.name == "nt":
                try:
                    tl = os.path.join(os.path.dirname(self.torch.__file__), "lib")
                    if os.path.isdir(tl):
                        os.add_dll_directory(tl)
                        os.environ["PATH"] = tl + os.pathsep + os.environ.get("PATH", "")
                except Exception:
                    pass
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                return False
            if model_path is None:
                import nudenet as _nn
                model_path = os.path.join(os.path.dirname(_nn.__file__), "320n.onnx")
            sess = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.nudenet.onnx_session = sess
            return "CUDAExecutionProvider" in sess.get_providers()
        except Exception as e:
            self.log(f"[INFO] NudeNet GPU化は不可 (CPUで続行): {str(e)[:120]}")
            return False

    def _nudenet_dets(self, bgr: np.ndarray) -> List[Detection]:
        if self.nudenet is None:
            return []
        try:
            results = self.nudenet.detect(bgr)  # ndarray直接入力 (一時ファイル不要)
        except Exception as e:
            self.log(f"[WARNING] NudeNet 推論失敗: {str(e)[:120]}")
            return []
        dets: List[Detection] = []
        for det in results:
            label = det.get("class", "")
            score = float(det.get("score", 0.0))
            if label not in self.nn_labels or score < self.nn_conf:
                continue
            b = det.get("box", [])
            if len(b) != 4:
                continue
            # NudeNet は [x, y, w, h] 形式 (旧実装は xyxy と誤解釈していた)
            x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            raw = (int(x), int(y), int(x + w), int(y + h))
            cls = _NUDENET_CLS_MAP.get(label, "default")
            sb = shrink_box(raw[0], raw[1], raw[2], raw[3], cls, self.cfg)
            if sb:
                dets.append(Detection(cls=cls, conf=score, box=sb, raw=raw, tid=None))
        return dets

    # -- YOLO ---------------------------------------------------------------
    def _auto_imgsz(self, w: int, h: int) -> int:
        if self.imgsz_cfg:
            return self.imgsz_cfg
        return 640 if max(w, h) < 1100 else 1280

    def _parse_results(self, res, names: dict, class_filter, with_ids: bool) -> List[Detection]:
        dets: List[Detection] = []
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return dets
        boxes = res[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = None
        if with_ids and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            cls_name = names.get(int(clss[i]), "default")
            if class_filter is not None and int(clss[i]) not in class_filter:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            raw = (int(x1), int(y1), int(x2), int(y2))
            sb = shrink_box(x1, y1, x2, y2, cls_name, self.cfg)
            if sb is None:
                continue
            tid = ("t", int(ids[i])) if ids is not None else None
            dets.append(Detection(cls=cls_name, conf=float(confs[i]), box=sb, raw=raw, tid=tid))
        return dets

    def reset_tracker(self) -> None:
        """動画ごとにトラッカー状態をリセットする。"""
        try:
            if getattr(self.models[0][0], "predictor", None) is not None:
                self.models[0][0].predictor.trackers = None
            self.models[0][0].predictor = None
        except Exception:
            pass
        self._frame_no = 0

    def track_frame(self, bgr: np.ndarray) -> List[Detection]:
        """動画パス1用: プライマリはBoT-SORTトラッキング、他モデル+NudeNetは検出のみ。"""
        h, w = bgr.shape[:2]
        imgsz = self._auto_imgsz(w, h)
        dets: List[Detection] = []
        tracker = self.cfg["video"].get("tracker_yaml", "botsort.yaml")
        tracker_path = tracker if os.path.isabs(tracker) else os.path.join(BASE_DIR, tracker)
        if not os.path.exists(tracker_path):
            tracker_path = "botsort.yaml"  # ultralytics 同梱デフォルト

        m0, names0 = self.models[0]
        try:
            res = m0.track(
                bgr, persist=True, conf=self.conf, iou=self.iou, imgsz=imgsz,
                quantize=self.quantize, device=self.device, tracker=tracker_path,
                classes=self.model_class_filters[0], verbose=False,
            )
            dets += self._parse_results(res, names0, None, with_ids=True)
        except Exception as e:
            self.log(f"[WARNING] トラッキング推論失敗: {str(e)[:150]}")
            # フォールバック: 通常検出
            try:
                res = m0.predict(bgr, conf=self.conf, iou=self.iou, imgsz=imgsz,
                                 quantize=self.quantize, device=self.device,
                                 classes=self.model_class_filters[0], verbose=False)
                dets += self._parse_results(res, names0, None, with_ids=False)
            except Exception:
                pass

        for (m, names), cfilter in list(zip(self.models, self.model_class_filters))[1:]:
            try:
                res = m.predict(bgr, conf=self.conf, iou=self.iou, imgsz=imgsz,
                                quantize=self.quantize, device=self.device,
                                classes=cfilter, verbose=False)
                dets += self._parse_results(res, names, None, with_ids=False)
            except Exception as e:
                self.log(f"[WARNING] サブモデル推論失敗: {str(e)[:120]}")

        if self.nudenet is not None and (self._frame_no % self.nn_interval == 0):
            dets += self._nudenet_dets(bgr)
        self._frame_no += 1
        return dets

    def detect_image(self, bgr: np.ndarray) -> List[Detection]:
        """静止画用: 全モデル + TTA + (高解像度時)タイル推論。"""
        d = self.cfg["detector"]
        h, w = bgr.shape[:2]
        imgsz = self._auto_imgsz(w, h)
        tta = bool(d.get("tta_images", True))
        dets: List[Detection] = []

        for (m, names), cfilter in zip(self.models, self.model_class_filters):
            try:
                res = m.predict(bgr, conf=self.conf, iou=self.iou, imgsz=imgsz,
                                quantize=self.quantize, device=self.device, augment=tta,
                                classes=cfilter, verbose=False)
                dets += self._parse_results(res, names, None, with_ids=False)
            except Exception as e:
                self.log(f"[WARNING] YOLO推論失敗: {str(e)[:120]}")

        # 高解像度タイル推論 (プライマリモデルのみ / 小さな対象の取りこぼし対策)
        tile_min = int(d.get("tile_min_side", 0))
        if tile_min and max(w, h) >= tile_min:
            dets += self._tiled_detect(bgr)

        dets += self._nudenet_dets(bgr)
        return _nms_dedup(dets)

    def _tiled_detect(self, bgr: np.ndarray) -> List[Detection]:
        d = self.cfg["detector"]
        ts = int(d.get("tile_size", 1024))
        ov = float(d.get("tile_overlap", 0.2))
        h, w = bgr.shape[:2]
        step = max(1, int(ts * (1 - ov)))
        m0, names0 = self.models[0]
        out: List[Detection] = []
        ys = list(range(0, max(1, h - ts + 1), step)) or [0]
        xs = list(range(0, max(1, w - ts + 1), step)) or [0]
        if ys[-1] + ts < h:
            ys.append(h - ts)
        if xs[-1] + ts < w:
            xs.append(w - ts)
        for y0 in ys:
            for x0 in xs:
                y1t, x1t = max(0, y0), max(0, x0)
                tile = bgr[y1t:min(h, y1t + ts), x1t:min(w, x1t + ts)]
                if tile.shape[0] < 64 or tile.shape[1] < 64:
                    continue
                try:
                    res = m0.predict(tile, conf=self.conf, iou=self.iou, imgsz=640,
                                     quantize=self.quantize, device=self.device,
                                     classes=self.model_class_filters[0], verbose=False)
                    for det in self._parse_results(res, names0, None, with_ids=False):
                        rx1, ry1, rx2, ry2 = det.raw
                        raw = (rx1 + x1t, ry1 + y1t, rx2 + x1t, ry2 + y1t)
                        sb = shrink_box(raw[0], raw[1], raw[2], raw[3], det.cls, self.cfg)
                        if sb:
                            out.append(Detection(cls=det.cls, conf=det.conf, box=sb, raw=raw))
                except Exception:
                    pass
        return out

    # -- MobileSAM ----------------------------------------------------------
    def refine_masks(self, bgr: np.ndarray, dets: List[Detection]) -> List[Optional[np.ndarray]]:
        """検出箱をプロンプトに MobileSAM でマスク精密化 (静止画向け)。失敗時は None (=矩形)。"""
        if not dets:
            return []
        if not self.cfg["mosaic"].get("mask_mode_image", True):
            return [None] * len(dets)
        if self._sam is None and not self._sam_tried:
            self._sam_tried = True
            path = self.cfg["mosaic"].get("mask_model", "models/mobile_sam.pt")
            abs_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
            if os.path.exists(abs_path):
                try:
                    from ultralytics import SAM
                    self._sam = SAM(abs_path)
                    self.log("[INFO] MobileSAM 読込 (マスク精密モザイク有効)")
                except Exception as e:
                    self.log(f"[INFO] MobileSAM 無効 (矩形モザイクを使用): {str(e)[:120]}")
        if self._sam is None:
            return [None] * len(dets)

        h, w = bgr.shape[:2]
        masks: List[Optional[np.ndarray]] = []
        try:
            bboxes = [list(d.raw) for d in dets]
            res = self._sam(bgr, bboxes=bboxes, verbose=False)
            mdata = res[0].masks.data.cpu().numpy() if res and res[0].masks is not None else None
            for i, det in enumerate(dets):
                mask = None
                if mdata is not None and i < len(mdata):
                    mk = mdata[i].astype(np.uint8)
                    if mk.shape[:2] != (h, w):
                        mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
                    # マスクは検出箱内に限定し、面積が小さすぎる場合は失敗扱い
                    x1, y1, x2, y2 = clip_box(det.raw, w, h) or (0, 0, 0, 0)
                    limited = np.zeros((h, w), dtype=np.uint8)
                    limited[y1:y2, x1:x2] = mk[y1:y2, x1:x2]
                    box_area = max(1, (x2 - x1) * (y2 - y1))
                    if limited.sum() >= 0.10 * box_area:
                        mask = limited
                masks.append(mask)
        except Exception as e:
            self.log(f"[INFO] SAMマスク生成失敗 (矩形を使用): {str(e)[:120]}")
            masks = [None] * len(dets)
        return masks


# ---------------------------------------------------------------------------
# モザイク適用 (BGR ndarray, in-place)
# ---------------------------------------------------------------------------

def _grid_rect(x1: int, y1: int, x2: int, y2: int, cell: int, w: int, h: int) -> Optional[Box]:
    """画像原点に整列したセルグリッドへ矩形を拡張する (動画でのセルちらつき防止)。"""
    gx1 = max(0, (x1 // cell) * cell)
    gy1 = max(0, (y1 // cell) * cell)
    gx2 = min(w, ((x2 + cell - 1) // cell) * cell)
    gy2 = min(h, ((y2 + cell - 1) // cell) * cell)
    if gx2 - gx1 < 1 or gy2 - gy1 < 1:
        return None
    return (gx1, gy1, gx2, gy2)


def _pixelate_roi(roi: np.ndarray, cell: int) -> np.ndarray:
    rh, rw = roi.shape[:2]
    sw, sh = max(1, math.ceil(rw / cell)), max(1, math.ceil(rh / cell))
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def apply_regions(frame: np.ndarray, boxes: List[Box], pattern: str, cell: int,
                  masks: Optional[List[Optional[np.ndarray]]] = None) -> np.ndarray:
    """フレーム(BGR)の指定領域にモザイク/ぼかし/黒塗りを適用する (in-place)。

    masks[i] が与えられた場合は矩形の代わりにマスク形状で適用する
    (マスクはセル半分ぶん膨張させ、被覆の安全マージンを確保)。
    """
    pattern = normalize_pattern(pattern)
    h, w = frame.shape[:2]
    for i, b in enumerate(boxes):
        cb = clip_box(b, w, h)
        if cb is None:
            continue
        x1, y1, x2, y2 = cb
        mask = masks[i] if masks is not None and i < len(masks) else None

        if mask is not None:
            # マスクをセル/2 だけ膨張 + グリッド矩形内で合成
            k = max(3, (cell // 2) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_d = cv2.dilate(mask, kernel)
            ys, xs = np.where(mask_d > 0)
            if len(xs) == 0:
                mask = None
            else:
                mx1, my1 = int(xs.min()), int(ys.min())
                mx2, my2 = int(xs.max()) + 1, int(ys.max()) + 1
                g = _grid_rect(mx1, my1, mx2, my2, cell, w, h)
                if g is None:
                    mask = None
                else:
                    gx1, gy1, gx2, gy2 = g
                    roi = frame[gy1:gy2, gx1:gx2]
                    if pattern in ("モザイク標準", "モザイク強", "モザイク特大"):
                        filt = _pixelate_roi(roi, cell)
                    elif pattern == "ぼかし":
                        sigma = max(3.0, cell * 0.8)
                        kk = int(2 * math.ceil(sigma * 1.5) + 1)
                        filt = cv2.GaussianBlur(roi, (kk, kk), sigma)
                    else:  # 黒塗り
                        filt = np.zeros_like(roi)
                    mm = (mask_d[gy1:gy2, gx1:gx2] > 0)
                    roi[mm] = filt[mm]
                    continue

        # 矩形適用 (マスクなし/失敗時)
        if pattern in ("モザイク標準", "モザイク強", "モザイク特大"):
            g = _grid_rect(x1, y1, x2, y2, cell, w, h)
            if g is None:
                continue
            gx1, gy1, gx2, gy2 = g
            frame[gy1:gy2, gx1:gx2] = _pixelate_roi(frame[gy1:gy2, gx1:gx2], cell)
        elif pattern == "ぼかし":
            sigma = max(3.0, cell * 0.8)
            kk = int(2 * math.ceil(sigma * 1.5) + 1)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (kk, kk), sigma)
        else:  # 黒塗り
            frame[y1:y2, x1:x2] = 0
    return frame


# ---------------------------------------------------------------------------
# シーンカット検出 (軽量: HSVヒストグラム相関)
# ---------------------------------------------------------------------------

class SceneCutDetector:
    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold
        self._prev_hist = None

    def is_cut(self, bgr: np.ndarray) -> bool:
        small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self._prev_hist is not None:
            corr = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
            cut = corr < self.threshold
        self._prev_hist = hist
        return cut


# ---------------------------------------------------------------------------
# 時系列後処理 (動画 2パスの中核)
# ---------------------------------------------------------------------------

def build_timeline(n_frames: int,
                   per_frame: List[List[Detection]],
                   scene_cuts: set,
                   cfg: dict) -> List[List[Box]]:
    """フレーム毎の検出結果から、補間・平滑化済みのモザイク適用タイムラインを構築する。

    scene_cuts: フレーム f が「f-1 と f の間にカットがある」集合。補間・延長はカットを跨がない。
    """
    v = cfg["video"]
    max_gap = int(v["max_gap_frames"])
    pad = int(v["pad_frames"])
    win = max(1, int(v["smooth_window"]))
    if win % 2 == 0:
        win += 1

    # --- 1) トラック集約: YOLO track id を優先し、それ以外はIoUで時系列チェーン化 ---
    tracks: Dict[tuple, Dict[int, Box]] = {}
    anon_active: List[List] = []  # [key, last_frame, last_box]
    anon_seq = 0
    ANON_GAP = 6
    ANON_IOU = 0.30

    for f in range(n_frames):
        if f in scene_cuts:
            anon_active = []  # カットを跨ぐチェーンを禁止
        for det in per_frame[f]:
            if det.tid is not None:
                slot = tracks.setdefault(det.tid, {})
                slot[f] = box_union(slot[f], det.box) if f in slot else det.box
            else:
                # 既存トラック(直近数フレーム)への吸着を試みる
                attached = False
                for key, slot in tracks.items():
                    for bf in range(f, max(-1, f - ANON_GAP - 1), -1):
                        if bf in slot:
                            if box_iou(slot[bf], det.box) > ANON_IOU:
                                slot[f] = box_union(slot[f], det.box) if f in slot else det.box
                                attached = True
                            break
                    if attached:
                        break
                if attached:
                    continue
                # 匿名チェーンへ
                best = None
                for rec in anon_active:
                    if f - rec[1] <= ANON_GAP and box_iou(rec[2], det.box) > ANON_IOU:
                        best = rec
                        break
                if best is None:
                    key = ("a", anon_seq)
                    anon_seq += 1
                    tracks[key] = {f: det.box}
                    anon_active.append([key, f, det.box])
                else:
                    key = best[0]
                    slot = tracks[key]
                    slot[f] = box_union(slot[f], det.box) if f in slot else det.box
                    best[1], best[2] = f, det.box
        anon_active = [r for r in anon_active if f - r[1] <= ANON_GAP]

    # カット位置リスト (昇順)
    cuts_sorted = sorted(scene_cuts)

    def crosses_cut(f_from: int, f_to: int) -> bool:
        """f_from < f_to 間 (f_from+1 .. f_to) にカットがあるか"""
        for c in cuts_sorted:
            if f_from < c <= f_to:
                return True
        return False

    timeline: List[List[Box]] = [[] for _ in range(n_frames)]

    # --- 2) トラック毎: セグメント分割 → 補間 → 延長 → 平滑化(和集合保証) ---
    for slot in tracks.values():
        frames_sorted = sorted(slot.keys())
        if not frames_sorted:
            continue
        # セグメント分割
        segments: List[List[int]] = [[frames_sorted[0]]]
        for f in frames_sorted[1:]:
            prev = segments[-1][-1]
            if f - prev <= max_gap and not crosses_cut(prev, f):
                segments[-1].append(f)
            else:
                segments.append([f])

        for seg in segments:
            series: Dict[int, Box] = {}
            # 線形補間
            for i, f in enumerate(seg):
                series[f] = slot[f]
                if i + 1 < len(seg):
                    nf = seg[i + 1]
                    if nf - f > 1:
                        b0, b1 = slot[f], slot[nf]
                        for g in range(f + 1, nf):
                            t = (g - f) / (nf - f)
                            series[g] = tuple(int(round(b0[k] + (b1[k] - b0[k]) * t)) for k in range(4))
            # リードイン/アウト延長
            f0, f1 = seg[0], seg[-1]
            for g in range(f0 - 1, max(-1, f0 - 1 - pad), -1):
                if g < 0 or crosses_cut(g, f0):
                    break
                series[g] = slot[f0]
            for g in range(f1 + 1, min(n_frames, f1 + 1 + pad)):
                if crosses_cut(f1, g):
                    break
                series[g] = slot[f1]

            # 移動平均平滑化 (中心窓) → 生の箱との和集合で被覆を保証
            sf = sorted(series.keys())
            half_w = win // 2
            for idx, f in enumerate(sf):
                lo, hi = max(0, idx - half_w), min(len(sf), idx + half_w + 1)
                cx = sum((series[sf[j]][0] + series[sf[j]][2]) / 2 for j in range(lo, hi)) / (hi - lo)
                cy = sum((series[sf[j]][1] + series[sf[j]][3]) / 2 for j in range(lo, hi)) / (hi - lo)
                bw = sum(series[sf[j]][2] - series[sf[j]][0] for j in range(lo, hi)) / (hi - lo)
                bh = sum(series[sf[j]][3] - series[sf[j]][1] for j in range(lo, hi)) / (hi - lo)
                sm = (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2))
                timeline[f].append(box_union(sm, series[f]))

    # --- 3) フレーム内の重複箱を和集合統合 ---
    for f in range(n_frames):
        if timeline[f]:
            timeline[f] = merge_boxes_union(timeline[f])
    return timeline


# ---------------------------------------------------------------------------
# ffmpeg 出力 (高品質エンコード / 二重再エンコード排除)
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg が見つかりません。PATH に ffmpeg.exe を追加してください。")
    return exe


def probe_audio_codec(path: str) -> Optional[str]:
    """音声ストリームのコーデック名 (無ければ None)。"""
    try:
        import ffmpeg as ffm
        info = ffm.probe(path)
        for s in info.get("streams", []):
            if s.get("codec_type") == "audio":
                return s.get("codec_name", "unknown")
    except Exception:
        pass
    return None


class FFmpegVideoWriter:
    """rawvideo を ffmpeg パイプへ流し込み、libx264 で一発エンコードする。

    audio_source を与えると そのファイルの音声を同時に mux する
    (aac/mp3 は copy、それ以外は AAC 再エンコード)。
    """

    def __init__(self, out_path: str, width: int, height: int, fps: float,
                 audio_source: Optional[str] = None, crf: int = 18,
                 preset: str = "medium", audio_bitrate: str = "192k"):
        self.out_path = out_path
        exe = find_ffmpeg()
        if not fps or fps != fps or fps <= 0:  # 0/NaN ガード
            fps = 30.0
        args = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-framerate", f"{fps:.6f}",
            "-i", "pipe:0",
        ]
        acodec = probe_audio_codec(audio_source) if audio_source else None
        if audio_source and acodec:
            # 注: -shortest は「音声が動画より短い」場合に映像まで切り詰めるため使わない
            args += ["-i", audio_source, "-map", "0:v:0", "-map", "1:a:0?"]
            if acodec in ("aac", "mp3"):
                args += ["-c:a", "copy"]
            else:
                args += ["-c:a", "aac", "-b:a", audio_bitrate]
        else:
            args += ["-map", "0:v:0"]
        args += [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, creationflags=creationflags,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        err = b""
        try:
            err = self.proc.stderr.read()
        except Exception:
            pass
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg エンコード失敗 (code {rc}): {err.decode(errors='replace')[:400]}")

    def abort(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        if os.path.exists(self.out_path):
            try:
                os.remove(self.out_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 動画処理パイプライン (2パス)
# ---------------------------------------------------------------------------

def out_container_for(input_path: str) -> str:
    ext = os.path.splitext(input_path)[1].lower()
    return ".mov" if ext == ".mov" else ".mp4"


def process_video(video_path: str,
                  out_path: str,
                  pattern: str,
                  cfg: dict,
                  detector: NsfwDetector,
                  keep_audio: bool = True,
                  progress: Optional[Callable[[str, int, int], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None,
                  log: Callable[[str], None] = print) -> dict:
    """動画を2パスで処理する。

    Pass1: 全フレーム検出 (トラッキング) + シーンカット検出
    後処理: build_timeline (補間/延長/平滑化)
    Pass2: モザイク適用 + ffmpegパイプエンコード
    """
    t0 = time.time()
    pattern = normalize_pattern(pattern)
    v = cfg["video"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    detector.reset_tracker()
    scene = SceneCutDetector(float(v["scene_cut_threshold"]))

    per_frame: List[List[Detection]] = []
    scene_cuts: set = set()

    # ---- Pass 1: 検出 ----
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scene.is_cut(frame) and idx > 0:
            scene_cuts.add(idx)
        per_frame.append(detector.track_frame(frame))
        idx += 1
        if progress and (idx % 5 == 0 or idx == 1):
            progress("detect", idx, max(total_est, idx))
        if cancel and cancel():
            cap.release()
            return {"cancelled": True}
    cap.release()
    n_frames = idx
    if n_frames == 0:
        raise RuntimeError(f"フレームを読み込めません: {video_path}")

    det_frames = sum(1 for d in per_frame if d)
    log(f"[INFO] Pass1完了: {n_frames}フレーム / 検出あり{det_frames}フレーム / カット{len(scene_cuts)}箇所")

    # ---- 時系列後処理 ----
    timeline = build_timeline(n_frames, per_frame, scene_cuts, cfg)
    covered = sum(1 for b in timeline if b)
    cell = compute_cell(width, height, pattern, cfg)
    log(f"[INFO] タイムライン構築: モザイク適用{covered}フレーム / セル寸法{cell}px")

    # ---- Pass 2: 適用 + エンコード ----
    cap = cv2.VideoCapture(video_path)
    writer = FFmpegVideoWriter(
        out_path, width, height, fps,
        audio_source=video_path if keep_audio else None,
        crf=int(v["crf"]), preset=str(v["preset"]),
        audio_bitrate=str(v.get("audio_bitrate", "192k")),
    )
    try:
        f = 0
        while True:
            ret, frame = cap.read()
            if not ret or f >= n_frames:
                break
            if timeline[f]:
                apply_regions(frame, timeline[f], pattern, cell)
            writer.write(frame)
            f += 1
            if progress and (f % 10 == 0 or f == 1 or f == n_frames):
                progress("encode", f, n_frames)
            if cancel and cancel():
                cap.release()
                writer.abort()
                return {"cancelled": True}
        cap.release()
        writer.close()
    except Exception:
        writer.abort()
        raise

    return {
        "cancelled": False,
        "frames": n_frames,
        "detected_frames": det_frames,
        "covered_frames": covered,
        "scene_cuts": len(scene_cuts),
        "cell": cell,
        "elapsed_sec": round(time.time() - t0, 1),
        "out_path": out_path,
    }


def verify_video(video_path: str,
                 cfg: dict,
                 detector: NsfwDetector,
                 progress: Optional[Callable[[str, int, int], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None,
                 log: Callable[[str], None] = print,
                 verify_conf: float = 0.25) -> int:
    """出力動画を再スキャンし、検出が残っているフレーム数を返す (書き換えはしない)。

    verify_conf: 検証用の信頼度閾値。モザイク適用済み領域への低confidence誤反応を
    抑えるため、通常検出 (既定0.10) より高めに設定する。実際の未処理領域は
    高confidenceで検出されるため漏れ検出能力への影響は小さい。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    detector.reset_tracker()
    orig_conf = detector.conf
    detector.conf = max(orig_conf, verify_conf)
    leftover = 0
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            dets = detector.track_frame(frame)
            if dets:
                leftover += 1
            if progress and (idx % 10 == 0 or idx == 1):
                progress("verify", idx, max(total, idx))
            if cancel and cancel():
                return leftover
    finally:
        detector.conf = orig_conf
        cap.release()
    log(f"[INFO] 再スキャン: 検出残存 {leftover}/{idx} フレーム")
    return leftover


# ---------------------------------------------------------------------------
# 静止画処理
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def process_image_bgr(bgr: np.ndarray, pattern: str, cfg: dict, detector: NsfwDetector,
                      use_masks: bool = True) -> Tuple[np.ndarray, int]:
    """BGR ndarray にモザイクを適用して返す。戻り値: (画像, 適用領域数)"""
    pattern = normalize_pattern(pattern)
    h, w = bgr.shape[:2]
    dets = detector.detect_image(bgr)
    if not dets:
        return bgr, 0
    masks = detector.refine_masks(bgr, dets) if use_masks else [None] * len(dets)
    # 矩形はマージ、マスク付きはそのまま
    rect_boxes = [d.box for d, m in zip(dets, masks) if m is None]
    masked = [(d.box, m) for d, m in zip(dets, masks) if m is not None]
    cell = compute_cell(w, h, pattern, cfg)
    boxes: List[Box] = merge_boxes_union(rect_boxes) if rect_boxes else []
    all_boxes = boxes + [b for b, _ in masked]
    all_masks: List[Optional[np.ndarray]] = [None] * len(boxes) + [m for _, m in masked]
    apply_regions(bgr, all_boxes, pattern, cell, all_masks)
    return bgr, len(all_boxes)


def process_image_file(in_path: str, out_path: str, pattern: str, cfg: dict,
                       detector: NsfwDetector, log: Callable[[str], None] = print) -> int:
    """画像ファイルを処理して保存する。GIFアニメ/EXIF/ICC/透過を保持。戻り値: 適用領域数"""
    from PIL import Image, ImageOps, ImageSequence

    ext = os.path.splitext(in_path)[1].lower()
    im = Image.open(in_path)

    # --- アニメGIF: 全フレーム処理 ---
    if ext == ".gif" and getattr(im, "is_animated", False):
        frames_out = []
        durations = []
        n_regions = 0
        for frame in ImageSequence.Iterator(im):
            durations.append(frame.info.get("duration", im.info.get("duration", 100)))
            rgb = frame.convert("RGB")
            bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            bgr, n = process_image_bgr(bgr, pattern, cfg, detector, use_masks=False)
            n_regions += n
            out_f = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frames_out.append(out_f.quantize(colors=256, method=Image.MEDIANCUT))
        frames_out[0].save(
            out_path, save_all=True, append_images=frames_out[1:],
            duration=durations, loop=im.info.get("loop", 0), disposal=2, optimize=False,
        )
        return n_regions

    # --- 静止画 ---
    im = ImageOps.exif_transpose(im)  # EXIF回転を画素に反映 (検出と保存の向き不一致を防止)
    exif = im.info.get("exif")
    icc = im.info.get("icc_profile")

    alpha = None
    if im.mode in ("RGBA", "LA", "PA"):
        im_rgba = im.convert("RGBA")
        alpha = im_rgba.getchannel("A")
        rgb = im_rgba.convert("RGB")
    else:
        rgb = im.convert("RGB")

    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    bgr, n_regions = process_image_bgr(bgr, pattern, cfg, detector, use_masks=True)
    out_im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if alpha is not None:
        out_im = out_im.convert("RGBA")
        out_im.putalpha(alpha)

    save_kwargs: dict = {}
    out_ext = os.path.splitext(out_path)[1].lower()
    if out_ext in (".jpg", ".jpeg"):
        if out_im.mode == "RGBA":
            out_im = out_im.convert("RGB")
        save_kwargs.update(quality=95, subsampling=1)
        if exif:
            save_kwargs["exif"] = exif
        if icc:
            save_kwargs["icc_profile"] = icc
    elif out_ext == ".png":
        if icc:
            save_kwargs["icc_profile"] = icc
    elif out_ext == ".webp":
        save_kwargs.update(quality=95)
        if icc:
            save_kwargs["icc_profile"] = icc
    out_im.save(out_path, **save_kwargs)
    return n_regions
