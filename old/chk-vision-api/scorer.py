# -*- coding: utf-8 -*-
"""
NSFW Image Checker - Scorer
スコアリングロジック
"""

from typing import Dict, Tuple
from dataclasses import dataclass

from config import (
    LIKELIHOOD_SCORES,
    CATEGORY_WEIGHTS,
    VERDICT_THRESHOLDS,
    VERDICT_ICONS
)


@dataclass
class CategoryResult:
    """カテゴリ別の結果"""
    likelihood: str
    score: int
    weighted_score: float


@dataclass
class ScoringResult:
    """スコアリング結果"""
    categories: Dict[str, CategoryResult]
    total_score: float
    verdict: str
    verdict_icon: str
    description: str = ""


class Scorer:
    """SafeSearch結果のスコアリング"""
    
    def __init__(self, 
                 likelihood_scores: Dict[str, int] = None,
                 category_weights: Dict[str, float] = None,
                 verdict_thresholds: Dict[str, Tuple[int, int]] = None):
        """
        Args:
            likelihood_scores: likelihood値のスコアマッピング
            category_weights: カテゴリ別重み係数
            verdict_thresholds: 判定閾値
        """
        self.likelihood_scores = likelihood_scores or LIKELIHOOD_SCORES
        self.category_weights = category_weights or CATEGORY_WEIGHTS
        self.verdict_thresholds = verdict_thresholds or VERDICT_THRESHOLDS
    
    def _get_likelihood_score(self, likelihood: str) -> int:
        """likelihood文字列をスコアに変換"""
        return self.likelihood_scores.get(likelihood, 0)
    
    def _calculate_total_score(self, category_results: Dict[str, CategoryResult]) -> float:
        """
        総合スコアを計算
        
        計算式: Σ(カテゴリスコア × 重み) / Σ(最大スコア × 重み) × 100
        """
        max_score = max(self.likelihood_scores.values())
        
        weighted_sum = sum(cr.weighted_score for cr in category_results.values())
        max_weighted_sum = sum(
            max_score * self.category_weights.get(cat, 1.0)
            for cat in category_results.keys()
        )
        
        if max_weighted_sum == 0:
            return 0.0
        
        return round((weighted_sum / max_weighted_sum) * 100, 1)
    
    def _determine_verdict(self, total_score: float) -> str:
        """総合スコアから判定を決定"""
        for verdict, (low, high) in self.verdict_thresholds.items():
            if low <= total_score <= high:
                return verdict
        return 'UNKNOWN'
    
    def score(self, analysis_result: Dict[str, any]) -> ScoringResult:
        """
        Vision APIの包括分析結果（analyze_image）をスコアリング
        
        Args:
            analysis_result: VisionClient.analyze_image()からの結果
            {
                'safe_search': {...},
                'description': 'cat, animal, pet',
                'labels': {...}
            }
            
        Returns:
            ScoringResult オブジェクト
        """
        # analyze_imageの結果からsafe_search部分を取得
        safe_search_result = analysis_result.get('safe_search', {})
        
        # 説明文を取得
        description = analysis_result.get('description', '')
        
        category_results = {}
        
        for category, likelihood in safe_search_result.items():
            score = self._get_likelihood_score(likelihood)
            weight = self.category_weights.get(category, 1.0)
            weighted_score = score * weight
            
            category_results[category] = CategoryResult(
                likelihood=likelihood,
                score=score,
                weighted_score=weighted_score
            )
        
        total_score = self._calculate_total_score(category_results)
        verdict = self._determine_verdict(total_score)
        verdict_icon = VERDICT_ICONS.get(verdict, '❓')
        
        return ScoringResult(
            categories=category_results,
            total_score=total_score,
            verdict=verdict,
            verdict_icon=verdict_icon,
            description=description
        )
