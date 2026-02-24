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
    # Per-engine raw scores
    engine_scores: Dict[str, float] = field(default_factory=dict)
    # Anime/Real style
    image_style: str = "不明"
    # Vision API SafeSearch
    safe_search: Dict[str, str] = field(default_factory=dict)
    # ViT NSFW result
    vit_label: str = ""
    vit_nsfw_score: float = 0.0
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

        # ─── 1. NudeNet Scoring ───
        nudenet_data = analysis_result.get('nudenet', {})
        detections = nudenet_data.get('detections', [])
        nudenet_score = self._score_nudenet(detections, result)
        engine_scores['nudenet'] = nudenet_score

        # ─── 2. WD14 Tag Analysis ───
        wd14_data = analysis_result.get('wd14', {})
        tags = wd14_data.get('tags', {})
        wd14_score = self._score_wd14(tags, result)
        engine_scores['wd14'] = wd14_score

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
        else:
            engine_scores['vision_api'] = 0

        # ─── 5. ViT NSFW Score ───
        vit_data = analysis_result.get('vit_nsfw', {})
        if 'error' not in vit_data:
            vit_nsfw = vit_data.get('nsfw_score', 0.0)
            result.vit_label = vit_data.get('label', '')
            result.vit_nsfw_score = vit_nsfw
            engine_scores['vit_nsfw'] = vit_nsfw * 100  # Convert to 0-100 scale
        else:
            engine_scores['vit_nsfw'] = 0

        # ─── 6. LFM2.5-VL Score ───
        lfm_data = analysis_result.get('lfm_vl', {})
        if 'error' not in lfm_data:
            lfm_nsfw = lfm_data.get('nsfw_score', 0.0)
            result.lfm_safety_level = lfm_data.get('safety_level', '')
            result.lfm_nsfw_score = lfm_nsfw
            result.lfm_description = lfm_data.get('description', '')
            engine_scores['lfm_vl'] = lfm_nsfw * 100
        else:
            engine_scores['lfm_vl'] = 0

        # ─── Gender Detection (from NudeNet FACE labels) ───
        self._detect_gender(detections, tags, result)

        # ─── Consensus Score ───
        total_weight = 0.0
        weighted_sum = 0.0
        for eng_name, score in engine_scores.items():
            weight = ENGINE_WEIGHTS.get(eng_name, 0.0)
            if weight > 0 and score > 0:
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

        # WD14 nude tag override
        for tag_name, tag_score in tags.items():
            if tag_name.lower() in ('nipples', 'pussy', 'penis') and tag_score >= 0.90:
                result.total_score = max(result.total_score, 85.0)
                result.primary_style = '裸'
                break

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

    def _score_wd14(self, tags: Dict[str, float], result: ScoringResult) -> float:
        """Score WD14 tags and determine primary style."""
        if not tags:
            result.primary_style = '着衣'
            result.all_tags = ''
            return 0.0

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

        # WD14 risk score based on nude/explicit tags
        nude_tags = STYLE_TAG_MAP.get('裸', [])
        max_nude = 0.0
        for tag, score in tags.items():
            if tag.lower() in [t.lower() for t in nude_tags]:
                max_nude = max(max_nude, score)

        return round(max_nude * 100, 2)

    def _detect_gender(self, detections: List[dict], tags: Dict[str, float], result: ScoringResult):
        """Detect gender from NudeNet face labels and WD14 tags."""
        for det in detections:
            if det.get('label') == 'FACE_FEMALE' and det.get('score', 0) > 0.5:
                result.gender = '女性'
                return
            elif det.get('label') == 'FACE_MALE' and det.get('score', 0) > 0.5:
                result.gender = '男性'
                return

        # Fallback to WD14 tags
        girl_score = tags.get('1girl', 0) + tags.get('female focus', 0)
        boy_score = tags.get('1boy', 0) + tags.get('male focus', 0)
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
