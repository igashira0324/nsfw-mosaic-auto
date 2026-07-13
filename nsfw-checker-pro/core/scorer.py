# -*- coding: utf-8 -*-
"""
nsfw-checker-pro - Scorer
Unified scoring logic that combines results from all engines into a consensus verdict.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CATEGORY_MAP, THRESHOLDS, VERDICT_ICONS, STYLE_TAG_MAP,
    ENGINE_WEIGHTS, ANIME_TAGS, REAL_TAGS
)


@dataclass
class CategoryScore:
    display_score: float
    max_score: float
    label_info: str = ""


@dataclass
class ScoringResult:
    # Per-category scores (from NudeNet)
    categories: Dict[str, CategoryScore] = field(default_factory=dict)
    # Total consensus score (0-100)
    total_score: float = 0.0
    # Final verdict
    verdict: str = "SAFE"
    verdict_icon: str = "✅"
    # Summary label
    labels_summary: str = ""
    # Primary clothing style
    primary_style: str = "着衣"
    # All WD14 tags
    all_tags: str = ""
    # WD14 content rating (general/sensitive/questionable/explicit)
    wd14_rating: Dict[str, float] = field(default_factory=dict)
    # Photo Tagger (idolsankaku, 実写) content rating
    photo_tagger_rating: Dict[str, float] = field(default_factory=dict)
    # Per-engine raw scores
    engine_scores: Dict[str, float] = field(default_factory=dict)
    # Anime/Real style
    image_style: str = "不明"
    # Vision API SafeSearch
    safe_search: Dict[str, str] = field(default_factory=dict)
    # ViT NSFW result (4段階: neutral/low/medium/high)
    vit_label: str = ""
    vit_nsfw_score: float = 0.0
    vit_severity: Dict[str, float] = field(default_factory=dict)
    # LFM2.5-VL result
    lfm_safety_level: str = ""
    lfm_nsfw_score: float = 0.0
    lfm_description: str = ""
    # Gender
    gender: str = "不明"


class Scorer:
    """マルチエンジン統合スコアラー"""

    def __init__(self):
        self.category_map = CATEGORY_MAP
        self.thresholds = THRESHOLDS

    def score(self, analysis_result: Dict[str, Any]) -> ScoringResult:
        """
        Score analysis results from all engines.

        Args:
            analysis_result: Combined result from MultiEngineAnalyzer

        Returns:
            ScoringResult with consensus verdict
        """
        result = ScoringResult()
        engine_scores = {}
        engine_ok = {}  # コンセンサス投票に参加させるか (エンジンが正常動作したか)

        # ─── 1. NudeNet Scoring ───
        nudenet_data = analysis_result.get('nudenet', {})
        detections = nudenet_data.get('detections', [])
        nudenet_score = self._score_nudenet(detections, result)
        engine_scores['nudenet'] = nudenet_score
        engine_ok['nudenet'] = 'error' not in nudenet_data

        # ─── 2. WD14 Tag Analysis (+ Content Rating) ───
        wd14_data = analysis_result.get('wd14', {})
        tags = wd14_data.get('tags', {})
        rating = wd14_data.get('rating', {})
        result.wd14_rating = rating
        wd14_score = self._score_wd14(tags, result, rating)
        engine_scores['wd14'] = wd14_score
        engine_ok['wd14'] = 'error' not in wd14_data

        # ─── 2b. Photo Tagger (idolsankaku, 実写) Rating Analysis ───
        photo_data = analysis_result.get('photo_tagger', {})
        photo_tags = photo_data.get('tags', {})
        photo_rating = photo_data.get('rating', {})
        result.photo_tagger_rating = photo_rating
        photo_score = self._score_photo_tagger(photo_tags, photo_rating, result)
        engine_scores['photo_tagger'] = photo_score
        engine_ok['photo_tagger'] = 'error' not in photo_data

        # ─── 3. Anime/Real Classification ───
        anime_data = analysis_result.get('anime_cls', {})
        style = anime_data.get('style', {})
        if style.get('anime', 0) > style.get('real', 0):
            result.image_style = "アニメ"
        elif style.get('real', 0) > 0.5:
            result.image_style = "実写"
        engine_scores['anime_cls'] = 0  # Not a risk score

        # WD14 tag override for style
        if tags:
            anime_tag_hits = sum(1 for t in tags if t.lower() in ANIME_TAGS)
            real_tag_hits = sum(1 for t in tags if t.lower() in REAL_TAGS)
            if anime_tag_hits > real_tag_hits:
                result.image_style = "アニメ"
            elif real_tag_hits > anime_tag_hits:
                result.image_style = "実写"

        # ─── 4. Vision API Score ───
        vision_data = analysis_result.get('vision_api', {})
        if 'error' not in vision_data:
            vision_score = vision_data.get('score', 0.0)
            result.safe_search = vision_data.get('safe_search', {})
            engine_scores['vision_api'] = vision_score
            engine_ok['vision_api'] = True
        else:
            engine_scores['vision_api'] = 0
            engine_ok['vision_api'] = False

        # ─── 5. ViT NSFW Score (4段階重大度) ───
        vit_data = analysis_result.get('vit_nsfw', {})
        if 'error' not in vit_data:
            vit_nsfw = vit_data.get('nsfw_score', 0.0)
            result.vit_label = vit_data.get('label', '')
            result.vit_nsfw_score = vit_nsfw
            result.vit_severity = vit_data.get('severity', {})
            engine_scores['vit_nsfw'] = vit_nsfw * 100  # Convert to 0-100 scale
            engine_ok['vit_nsfw'] = True
        else:
            engine_scores['vit_nsfw'] = 0
            engine_ok['vit_nsfw'] = False

        # ─── 6. LFM2.5-VL Score ───
        lfm_data = analysis_result.get('lfm_vl', {})
        if 'error' not in lfm_data:
            lfm_nsfw = lfm_data.get('nsfw_score', 0.0)
            result.lfm_safety_level = lfm_data.get('safety_level', '')
            result.lfm_nsfw_score = lfm_nsfw
            result.lfm_description = lfm_data.get('description', '')
            engine_scores['lfm_vl'] = lfm_nsfw * 100
            engine_ok['lfm_vl'] = True
        else:
            engine_scores['lfm_vl'] = 0
            engine_ok['lfm_vl'] = False

        # ─── Gender Detection (from NudeNet FACE labels) ───
        self._detect_gender(detections, tags, result, extra_tags=photo_tags)

        # ─── Consensus Score ───
        # 正常動作した「リスク評価エンジン」全てを重み付き平均に参加させる。
        # 旧実装はスコア0のエンジンを除外していたため、安全側の票が反映されず
        # スコアが上振れするバイアスがあった。anime_cls はスタイル分類のため対象外。
        total_weight = 0.0
        weighted_sum = 0.0
        for eng_name, score in engine_scores.items():
            if eng_name == 'anime_cls':
                continue
            weight = ENGINE_WEIGHTS.get(eng_name, 0.0)
            if weight > 0 and engine_ok.get(eng_name, False):
                weighted_sum += score * weight
                total_weight += weight

        if total_weight > 0:
            result.total_score = round(weighted_sum / total_weight, 2)
        else:
            result.total_score = nudenet_score  # Fallback to NudeNet only

        result.engine_scores = engine_scores

        # ─── Special Overrides ───
        # Nipples singularity override
        for det in detections:
            if det.get('label') in ('FEMALE_BREAST_EXPOSED',) and det.get('score', 0) >= 0.93:
                result.total_score = max(result.total_score, 85.0)
                result.primary_style = '裸'
                break

        # タガー(WD14 + 実写PhotoTagger)の nude タグ override
        for tag_name, tag_score in list(tags.items()) + list(photo_tags.items()):
            if tag_name.lower() in ('nipples', 'pussy', 'penis') and tag_score >= 0.90:
                result.total_score = max(result.total_score, 85.0)
                result.primary_style = '裸'
                break

        # レーティング override: WD14 と PhotoTagger どちらかが explicit を高確度で
        # 示せば格上げする (旧実装はWD14側のみを見ており、実写画像で
        # PhotoTaggerのみがexplicitを検出したケースを見逃していた)
        if max(float(rating.get('explicit', 0.0)), float(photo_rating.get('explicit', 0.0))) >= 0.85:
            result.total_score = max(result.total_score, 85.0)

        # ViT override
        if result.vit_nsfw_score >= 0.90:
            result.total_score = max(result.total_score, 80.0)

        # LFM override - high confidence from VLM is very reliable
        if result.lfm_nsfw_score >= 0.85:
            result.total_score = max(result.total_score, 82.0)
        elif result.lfm_safety_level == 'UNSAFE':
            result.total_score = max(result.total_score, 80.0)

        # ─── Final Verdict ───
        result.verdict = self._determine_verdict(result.total_score)
        result.verdict_icon = VERDICT_ICONS.get(result.verdict, '❓')

        return result

    def _score_nudenet(self, detections: List[dict], result: ScoringResult) -> float:
        """Score NudeNet detections and populate category scores."""
        category_max = {}
        labels = []

        for det in detections:
            label = det.get('label', '')
            score = det.get('score', 0.0)

            for cat_name, cat_labels in self.category_map.items():
                if label in cat_labels:
                    if cat_name not in category_max or score > category_max[cat_name]:
                        category_max[cat_name] = score
                    labels.append(f"{label}({score:.0%})")

        # Populate per-category scores
        for cat_name in ['FEMALE_BREAST', 'GENITALIA', 'BUTTOCKS', 'ANUS']:
            max_s = category_max.get(cat_name, 0.0)
            result.categories[cat_name] = CategoryScore(
                display_score=round(max_s * 100, 1),
                max_score=max_s
            )

        result.labels_summary = ', '.join(labels) if labels else '特になし'

        # NudeNet score: weighted max of critical categories
        critical_scores = []
        for cat in ['FEMALE_BREAST', 'GENITALIA', 'ANUS']:
            if cat in category_max:
                critical_scores.append(category_max[cat])
        buttocks = category_max.get('BUTTOCKS', 0)

        if critical_scores:
            nn_score = max(critical_scores) * 100
        else:
            nn_score = buttocks * 50  # Lower weight for buttocks alone

        return round(nn_score, 2)

    def _score_wd14(self, tags: Dict[str, float], result: ScoringResult,
                    rating: Dict[str, float] = None) -> float:
        """Score WD14 tags + content rating and determine primary style.

        レーティングヘッド (general/sensitive/questionable/explicit) はWD14の
        最重要NSFW指標。タグベースのスコアと組み合わせて最大値を採用する。
        """
        rating = rating or {}
        rating_score = max(
            float(rating.get('explicit', 0.0)) * 100.0,
            float(rating.get('questionable', 0.0)) * 60.0,
        )
        if not tags:
            result.primary_style = '着衣'
            result.all_tags = ''
            return round(rating_score, 2)

        top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:15]
        result.all_tags = ', '.join(f"{t}({s:.0%})" for t, s in top_tags)

        # Determine primary style
        style_scores = {}
        for style_name, style_tags in STYLE_TAG_MAP.items():
            s = 0.0
            for tag, score in tags.items():
                if tag.lower() in [t.lower() for t in style_tags]:
                    s = max(s, score)
            if s > 0.15:
                style_scores[style_name] = s

        if style_scores:
            result.primary_style = max(style_scores, key=style_scores.get)
        else:
            result.primary_style = '着衣'

        # WD14 risk score: nude/explicitタグ と レーティングヘッド の最大値
        nude_tags = STYLE_TAG_MAP.get('裸', [])
        max_nude = 0.0
        for tag, score in tags.items():
            if tag.lower() in [t.lower() for t in nude_tags]:
                max_nude = max(max_nude, score)

        return round(max(max_nude * 100, rating_score), 2)

    def _score_photo_tagger(self, tags: Dict[str, float], rating: Dict[str, float],
                            result: ScoringResult) -> float:
        """Score the real-photo tagger (idolsankaku) rating + nude-tag overlap.

        primary_style/all_tags は WD14 (アニメタガー) 側が担うため、ここでは
        上書きしない。レーティングヘッドと裸タグの重なりのみでrisk scoreを返す。
        """
        rating_score = max(
            float(rating.get('explicit', 0.0)) * 100.0,
            float(rating.get('questionable', 0.0)) * 60.0,
        )
        if not tags:
            return round(rating_score, 2)

        nude_tags = STYLE_TAG_MAP.get('裸', [])
        max_nude = 0.0
        for tag, score in tags.items():
            if tag.lower() in [t.lower() for t in nude_tags]:
                max_nude = max(max_nude, score)

        return round(max(max_nude * 100, rating_score), 2)

    def _detect_gender(self, detections: List[dict], tags: Dict[str, float], result: ScoringResult,
                       extra_tags: Dict[str, float] = None):
        """Detect gender from NudeNet face labels and tagger (WD14 + photo_tagger) tags."""
        for det in detections:
            if det.get('label') == 'FACE_FEMALE' and det.get('score', 0) > 0.5:
                result.gender = '女性'
                return
            elif det.get('label') == 'FACE_MALE' and det.get('score', 0) > 0.5:
                result.gender = '男性'
                return

        # Fallback to tagger tags (WD14 + 実写タガーを加点合算)
        girl_score = tags.get('1girl', 0) + tags.get('female focus', 0)
        boy_score = tags.get('1boy', 0) + tags.get('male focus', 0)
        if extra_tags:
            girl_score += extra_tags.get('1girl', 0) + extra_tags.get('female focus', 0)
            boy_score += extra_tags.get('1boy', 0) + extra_tags.get('male focus', 0)
        if girl_score > boy_score and girl_score > 0.3:
            result.gender = '女性'
        elif boy_score > girl_score and boy_score > 0.3:
            result.gender = '男性'

    def _determine_verdict(self, score: float) -> str:
        """Determine verdict from total score."""
        if score >= self.thresholds['UNSAFE'] * 100:
            return 'UNSAFE'
        elif score >= self.thresholds['HIGH_RISK'] * 100:
            return 'HIGH_RISK'
        elif score >= self.thresholds['MODERATE'] * 100:
            return 'MODERATE'
        elif score >= self.thresholds['LOW_RISK'] * 100:
            return 'LOW_RISK'
        else:
            return 'SAFE'
