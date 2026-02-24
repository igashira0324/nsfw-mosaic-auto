# -*- coding: utf-8 -*-
"""
nsfw-checker-pro - Report Exporters
CSV, JSON, and HTML report generation.
"""

import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def export_csv(results: List[Dict[str, Any]], output_path: str):
    """Export results to CSV file."""
    fieldnames = [
        'ファイル', '判定', 'スコア', 'スタイル', '性別', '画風',
        'NudeNet', 'WD14', 'VisionAPI', 'ViT', '詳細ラベル', 'タグ'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'ファイル': r.get('filename', ''),
                '判定': r.get('verdict', ''),
                'スコア': r.get('total_score', 0),
                'スタイル': r.get('primary_style', ''),
                '性別': r.get('gender', ''),
                '画風': r.get('image_style', ''),
                'NudeNet': r.get('engine_scores', {}).get('nudenet', 0),
                'WD14': r.get('engine_scores', {}).get('wd14', 0),
                'VisionAPI': r.get('engine_scores', {}).get('vision_api', 0),
                'ViT': r.get('engine_scores', {}).get('vit_nsfw', 0),
                '詳細ラベル': r.get('labels_summary', ''),
                'タグ': r.get('all_tags', '')
            })


def export_json(results: List[Dict[str, Any]], output_path: str):
    """Export results to JSON file."""
    export = {
        'generated_at': datetime.now().isoformat(),
        'total_files': len(results),
        'summary': _generate_summary(results),
        'results': results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)


def export_html(results: List[Dict[str, Any]], output_path: str):
    """Export results to a beautiful HTML report."""
    summary = _generate_summary(results)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    verdict_colors = {
        'SAFE': '#2ecc71', 'LOW_RISK': '#f1c40f', 'MODERATE': '#f39c12',
        'HIGH_RISK': '#e67e22', 'UNSAFE': '#e74c3c', 'ERROR': '#95a5a6'
    }

    rows_html = ""
    for r in results:
        verdict = r.get('verdict', 'ERROR')
        color = verdict_colors.get(verdict, '#95a5a6')
        rows_html += f"""
        <tr>
            <td class="filename">{r.get('filename', '')}</td>
            <td><span class="badge" style="background:{color}">{r.get('verdict_icon', '')} {verdict}</span></td>
            <td class="score">{r.get('total_score', 0):.1f}</td>
            <td>{r.get('primary_style', '')}</td>
            <td>{r.get('gender', '')}</td>
            <td>{r.get('image_style', '')}</td>
            <td class="score">{r.get('engine_scores', {}).get('nudenet', 0):.1f}</td>
            <td class="score">{r.get('engine_scores', {}).get('wd14', 0):.1f}</td>
            <td class="score">{r.get('engine_scores', {}).get('vision_api', 0):.1f}</td>
            <td class="score">{r.get('engine_scores', {}).get('vit_nsfw', 0):.1f}</td>
            <td class="tags">{r.get('labels_summary', '')}</td>
        </tr>"""

    # Summary cards
    summary_cards = ""
    for verdict, count in summary.items():
        color = verdict_colors.get(verdict, '#95a5a6')
        summary_cards += f'<div class="card" style="border-left: 4px solid {color}"><h3>{verdict}</h3><p class="count">{count}</p></div>'

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSFW Checker Pro - Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Segoe UI', 'Meiryo', sans-serif; background: #0d1117; color: #c9d1d9; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #58a6ff; font-size: 28px; margin-bottom: 5px; }}
.subtitle {{ color: #8b949e; margin-bottom: 20px; }}
.summary {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
.card {{ background: #161b22; padding: 16px 20px; border-radius: 8px; min-width: 120px; }}
.card h3 {{ font-size: 13px; color: #8b949e; margin-bottom: 4px; }}
.card .count {{ font-size: 28px; font-weight: bold; color: #f0f6fc; }}
table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
th {{ background: #21262d; color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
     padding: 12px 10px; text-align: left; position: sticky; top: 0; }}
td {{ padding: 10px; border-bottom: 1px solid #21262d; font-size: 13px; }}
tr:hover {{ background: #1c2128; }}
.filename {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.score {{ text-align: center; font-family: 'Consolas', monospace; }}
.tags {{ max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; color: #8b949e; }}
.badge {{ padding: 3px 10px; border-radius: 12px; color: #fff; font-size: 12px; font-weight: 600; white-space: nowrap; }}
</style>
</head>
<body>
<div class="container">
<h1>📊 NSFW Checker Pro Report</h1>
<p class="subtitle">Generated: {timestamp} | Total: {len(results)} files</p>
<div class="summary">{summary_cards}</div>
<table>
<thead><tr>
<th>ファイル</th><th>判定</th><th>スコア</th><th>スタイル</th><th>性別</th><th>画風</th>
<th>NudeNet</th><th>WD14</th><th>Vision</th><th>ViT</th><th>詳細</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def _generate_summary(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Generate verdict count summary."""
    summary = {'SAFE': 0, 'LOW_RISK': 0, 'MODERATE': 0, 'HIGH_RISK': 0, 'UNSAFE': 0}
    for r in results:
        v = r.get('verdict', 'ERROR')
        if v in summary:
            summary[v] += 1
    return summary
