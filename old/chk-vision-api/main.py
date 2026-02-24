# -*- coding: utf-8 -*-
"""
NSFW Image Checker - Main CLI
Google Cloud Vision API SafeSearch を使った画像NSFWチェックツール
"""

import argparse
import json
import csv
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from colorama import init, Fore, Style
    init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    class Fore:
        RED = GREEN = YELLOW = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''

from config import BATCH_DELAY, VERDICT_ICONS, API_KEY
from vision_client import VisionClient, VisionAPIError
from scorer import Scorer, ScoringResult
from file_handler import FileHandler
from gui import launch_gui


def color_verdict(verdict: str, icon: str) -> str:
    """判定結果に色を付ける"""
    if not HAS_COLORAMA:
        return f"{icon} {verdict}"
    
    colors = {
        'SAFE': Fore.GREEN,
        'LOW_RISK': Fore.YELLOW,
        'MODERATE': Fore.YELLOW,
        'HIGH_RISK': Fore.RED,
        'UNSAFE': Fore.RED + Style.BRIGHT
    }
    color = colors.get(verdict, Fore.WHITE)
    return f"{color}{icon} {verdict}{Style.RESET_ALL}"


def format_category_line(categories: dict) -> str:
    """カテゴリ結果を1行にフォーマット"""
    parts = []
    for cat in ['adult', 'racy', 'violence']:
        if cat in categories:
            cr = categories[cat]
            parts.append(f"{cat.capitalize()}: {cr.likelihood} ({cr.score})")
    return " | ".join(parts)


def print_result(file_path: Path, result: ScoringResult, index: int, total: int, labels: Dict[str, float] = None):
    """結果をコンソールに表示"""
    print(f"\n[{index}/{total}] {Fore.CYAN}{file_path.name}{Style.RESET_ALL}")
    print(f"  {format_category_line(result.categories)}")
    print(f"  Score: {result.total_score}/100 | Verdict: {color_verdict(result.verdict, result.verdict_icon)}")
    
    # 画像の説明文を表示
    if labels:
        top_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:3]  # 上位3つ
        label_text = ", ".join([f"{label}({score:.1%})" for label, score in top_labels])
        print(f"  📝 Labels: {label_text}")


def process_images(
    images: List[Path],
    client: VisionClient,
    scorer: Scorer,
    threshold: Optional[float] = None,
    quiet: bool = False
) -> List[Dict[str, Any]]:
    """
    画像リストを処理
    
    Args:
        images: 画像パスのリスト
        client: Vision APIクライアント
        scorer: スコアラー
        threshold: 結果に含める最小スコア（Noneなら全て）
        quiet: 進捗表示を抑制
        
    Returns:
        結果のリスト
    """
    results = []
    total = len(images)
    
    if HAS_TQDM and not quiet:
        iterator = tqdm(enumerate(images, 1), total=total, desc="Processing")
    else:
        iterator = enumerate(images, 1)
    
    for i, image_path in iterator:
        try:
            # 包括的分析を実行 (SafeSearch + Label Detection)
            analysis_result = client.analyze_image(image_path)
            
            # スコアリング
            score_result = scorer.score(analysis_result)
            
            # 閾値フィルタリング
            if threshold is not None and score_result.total_score < threshold:
                continue
            
            # 結果を保存
            result_data = {
                'file': str(image_path),
                'filename': image_path.name,
                'categories': {
                    cat: {
                        'likelihood': cr.likelihood,
                        'score': cr.score
                    }
                    for cat, cr in score_result.categories.items()
                },
                'total_score': score_result.total_score,
                'verdict': score_result.verdict,
                'labels': analysis_result.get('labels', {}),
                'description': score_result.description
            }
            results.append(result_data)
            
            # コンソール出力
            if not quiet and not HAS_TQDM:
                print_result(image_path, score_result, i, total, analysis_result.get('labels'))
            elif not quiet and HAS_TQDM:
                # tqdmのpostfixで最新結果を表示
                iterator.set_postfix({
                    'file': image_path.name[:20],
                    'score': score_result.total_score,
                    'verdict': score_result.verdict
                })
            
            # レート制限対策
            if i < total:
                time.sleep(BATCH_DELAY)
                
        except VisionAPIError as e:
            if not quiet:
                print(f"{Fore.RED}Error processing {image_path.name}: {e}{Style.RESET_ALL}")
            results.append({
                'file': str(image_path),
                'filename': image_path.name,
                'error': str(e)
            })
        except Exception as e:
            if not quiet:
                print(f"{Fore.RED}Unexpected error for {image_path.name}: {e}{Style.RESET_ALL}")
            results.append({
                'file': str(image_path),
                'filename': image_path.name,
                'error': str(e)
            })
    
    return results


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """結果のサマリーを生成"""
    summary = {
        'total': len(results),
        'safe': 0,
        'low_risk': 0,
        'moderate': 0,
        'high_risk': 0,
        'unsafe': 0,
        'errors': 0
    }
    
    for r in results:
        if 'error' in r:
            summary['errors'] += 1
        else:
            verdict = r.get('verdict', '').lower().replace('-', '_')
            if verdict in summary:
                summary[verdict] += 1
    
    return summary


def print_summary(summary: Dict[str, int]):
    """サマリーを表示"""
    print(f"\n{'='*60}")
    print(f"{Style.BRIGHT}📊 Summary{Style.RESET_ALL}")
    print(f"  Total: {summary['total']}")
    print(f"  {VERDICT_ICONS['SAFE']} Safe: {summary['safe']}")
    print(f"  {VERDICT_ICONS['LOW_RISK']} Low Risk: {summary['low_risk']}")
    print(f"  {VERDICT_ICONS['MODERATE']} Moderate: {summary['moderate']}")
    print(f"  {VERDICT_ICONS['HIGH_RISK']} High Risk: {summary['high_risk']}")
    print(f"  {VERDICT_ICONS['UNSAFE']} Unsafe: {summary['unsafe']}")
    if summary['errors'] > 0:
        print(f"  ❌ Errors: {summary['errors']}")
    print(f"{'='*60}")


def save_json(results: List[Dict], summary: Dict, output_path: Path):
    """JSON形式で保存"""
    data = {
        'summary': summary,
        'results': results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(results: List[Dict], output_path: Path):
    """CSV形式で保存"""
    if not results:
        return
    
    # ヘッダー
    headers = ['filename', 'adult', 'racy', 'violence', 'medical', 'spoof', 'total_score', 'verdict', 'file']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for r in results:
            if 'error' in r:
                row = [r['filename'], '', '', '', '', '', '', 'ERROR', r['file']]
            else:
                cats = r.get('categories', {})
                row = [
                    r['filename'],
                    cats.get('adult', {}).get('likelihood', ''),
                    cats.get('racy', {}).get('likelihood', ''),
                    cats.get('violence', {}).get('likelihood', ''),
                    cats.get('medical', {}).get('likelihood', ''),
                    cats.get('spoof', {}).get('likelihood', ''),
                    r.get('total_score', ''),
                    r.get('verdict', ''),
                    r['file']
                ]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='NSFW Image Checker - Google Cloud Vision API SafeSearch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py image.jpg
  python main.py ./images --recursive --output results.json
  python main.py --gui
        '''
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        help='Image file or directory path (Launch GUI if omitted)'
    )
    parser.add_argument(
        '--api-key', '-k',
        default=API_KEY,
        help='Google Cloud Vision API key (Overrides config.py)'
    )
    parser.add_argument(
        '--gui', '-g',
        action='store_true',
        help='Launch GUI interface'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process subdirectories recursively'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file path (.json or .csv)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help='Minimum score threshold to include in results'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()

    # GUIを起動する場合:
    # 1. パスが指定されていない
    # 2. --gui フラグが設定されている
    if args.path is None or args.gui:
        launch_gui()
        return
    
    # ファイルハンドラー
    file_handler = FileHandler()
    
    try:
        target_path = file_handler.validate_path(args.path)
    except FileNotFoundError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    # 画像収集
    images = file_handler.collect_images(target_path, args.recursive)
    
    if not images:
        print(f"{Fore.YELLOW}No image files found.{Style.RESET_ALL}")
        sys.exit(0)
    
    print(f"\n📁 Found {len(images)} image(s) to process")
    
    # クライアント・スコアラー初期化
    client = VisionClient(args.api_key)
    scorer = Scorer()
    
    # 処理実行
    results = process_images(
        images,
        client,
        scorer,
        threshold=args.threshold,
        quiet=args.quiet
    )
    
    # サマリー生成・表示
    summary = generate_summary(results)
    if not args.quiet:
        print_summary(summary)
    
    # 結果保存
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == '.json':
            save_json(results, summary, output_path)
            print(f"\n💾 Results saved to: {output_path}")
        elif output_path.suffix.lower() == '.csv':
            save_csv(results, output_path)
            print(f"\n💾 Results saved to: {output_path}")
        else:
            # デフォルトはJSON
            output_path = output_path.with_suffix('.json')
            save_json(results, summary, output_path)
            print(f"\n💾 Results saved to: {output_path}")


if __name__ == '__main__':
    main()
