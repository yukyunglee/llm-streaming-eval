#!/usr/bin/env python3
"""
Unified export script for experiment results.
Handles: Clustering, Temporal QA, and Summarization (with metric calculation)

Year-based directory mapping:
  - 2025 (story 1,2,3): experiments_structured_multistage
  - 2016 (story 25,218,227): experiments_structured

Usage:
    python export_multistage_results.py --task all --year 2025
    python export_multistage_results.py --task all --year 2016
    python export_multistage_results.py --task all --year all
    python export_multistage_results.py --task summarization --year 2025 --use-bertscore
"""

import json
import os
import re
import glob
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


# ============================================================================
# Common Utilities
# ============================================================================

# Year-based configuration
YEAR_CONFIG = {
    '2025': {
        'base_dir': Path('./experiments_structured_multistage'),
        'stories': [1, 2, 3]
    },
    '2016': {
        'base_dir': Path('./experiments_structured_2016'),
        'stories': [25, 218, 227]
    }
}

def get_all_models(base_dir: Path) -> List[str]:
    """Get all model directories"""
    models = []
    if not base_dir.exists():
        return models
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('experiment_summary'):
            models.append(item.name)
    return models


def get_year_configs(year: str) -> List[Dict]:
    """Get configurations based on year selection"""
    if year == 'all':
        return [{'year': '2025', **YEAR_CONFIG['2025']}, {'year': '2016', **YEAR_CONFIG['2016']}]
    elif year in YEAR_CONFIG:
        return [{'year': year, **YEAR_CONFIG[year]}]
    else:
        raise ValueError(f"Unknown year: {year}. Use '2025', '2016', or 'all'")


# ============================================================================
# Clustering Export
# ============================================================================

def parse_clustering_experiment(exp_folder: str) -> Optional[Dict]:
    """Parse clustering experiment folder name: cluster_218_t1_no_struct_20251015_172726"""
    pattern = r'cluster_(\d+)_t(\d+)_(no_struct|with_struct)_'
    match = re.search(pattern, exp_folder)
    
    if match:
        return {
            'story': int(match.group(1)),
            'text_per_event': int(match.group(2)),
            'structure': 'no' if match.group(3) == 'no_struct' else 'yes'
        }
    return None


def export_clustering(year_configs: List[Dict], output_dir: Path) -> List[Dict]:
    """Export clustering results to CSV"""
    print("\n" + "=" * 80)
    print("📊 CLUSTERING Results Export")
    print("=" * 80)
    
    rows = []
    
    for config in year_configs:
        year = config['year']
        base_dir = config['base_dir']
        valid_stories = config['stories']
        
        print(f"\n  📁 Processing {year} from {base_dir}...")
        
        for model_name in get_all_models(base_dir):
            clustering_dir = base_dir / model_name / 'clustering'
            if not clustering_dir.exists():
                continue
            
            for exp_dir in sorted(clustering_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                
                json_file = exp_dir / 'clustering_summary.json'
                if not json_file.exists():
                    continue
                
                metadata = parse_clustering_experiment(exp_dir.name)
                if not metadata:
                    continue
                
                # Filter by valid stories for this year
                if metadata['story'] not in valid_stories:
                    continue
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metrics
                    if 'evaluation_metrics' in data and 'original_task1_metrics' in data['evaluation_metrics']:
                        metrics = data['evaluation_metrics']['original_task1_metrics']
                        
                        row = {
                            'year': year,
                            'model': model_name,
                            'story': metadata['story'],
                            'text_per_event': metadata['text_per_event'],
                            'structure': metadata['structure'],
                            'ami': metrics.get('ami', None),
                            'nmi': metrics.get('nmi', None),
                            'b3_f1': metrics.get('b3_f1', None)
                        }
                        rows.append(row)
                except Exception as e:
                    print(f"  ⚠️  Error reading {json_file}: {e}")
    
    # Sort
    rows.sort(key=lambda x: (x['year'], x['model'], x['story'], x['text_per_event'], x['structure']))
    
    # Save per year
    for config in year_configs:
        year = config['year']
        year_rows = [r for r in rows if r['year'] == year]
        if not year_rows:
            continue
        
        save_dir = output_dir / year
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / 'clustering_results.csv'
        
        fieldnames = ['year', 'model', 'story', 'text_per_event', 'structure', 'ami', 'nmi', 'b3_f1']
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(year_rows)
        
        print(f"\n✅ Saved: {output_file} ({len(year_rows)} rows)")
    
    if rows:
        # Stats
        no_struct = [r for r in rows if r['structure'] == 'no' and r['b3_f1'] is not None]
        yes_struct = [r for r in rows if r['structure'] == 'yes' and r['b3_f1'] is not None]
        
        if no_struct and yes_struct:
            print(f"\n   📈 Stats:")
            print(f"      No structure:   {len(no_struct)} rows, mean B³-F1 = {sum(r['b3_f1'] for r in no_struct)/len(no_struct):.4f}")
            print(f"      With structure: {len(yes_struct)} rows, mean B³-F1 = {sum(r['b3_f1'] for r in yes_struct)/len(yes_struct):.4f}")
    else:
        print("  ⚠️  No clustering results found")
    
    return rows


# ============================================================================
# Temporal QA Export
# ============================================================================

def parse_qa_experiment(exp_folder: str) -> Optional[Dict]:
    """Parse QA experiment folder name: qa_1_t1_no_struct_20260115_074557"""
    pattern = r'qa_(\d+)_t(\d+)_(no_struct|with_struct)_'
    match = re.search(pattern, exp_folder)
    
    if match:
        return {
            'story': int(match.group(1)),
            'text_per_event': int(match.group(2)),
            'struct_type': match.group(3)
        }
    return None


def export_temporal_qa(year_configs: List[Dict], output_dir: Path) -> List[Dict]:
    """Export temporal QA results with intersection-based accuracy"""
    print("\n" + "=" * 80)
    print("📊 TEMPORAL QA Results Export")
    print("=" * 80)
    
    rows = []
    
    for config in year_configs:
        year = config['year']
        base_dir = config['base_dir']
        valid_stories = config['stories']
        
        print(f"\n  📁 Processing {year} from {base_dir}...")
        
        for model_name in get_all_models(base_dir):
            qa_dir = base_dir / model_name / 'temporal_qa'
            if not qa_dir.exists():
                continue
            
            # Group experiments by (story, text_per_event)
            experiments = defaultdict(lambda: {'no_struct': None, 'with_struct': None})
            
            for exp_dir in sorted(qa_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                
                json_file = exp_dir / 'qa_detailed_results.json'
                if not json_file.exists():
                    continue
                
                metadata = parse_qa_experiment(exp_dir.name)
                if not metadata:
                    continue
                
                # Filter by valid stories for this year
                if metadata['story'] not in valid_stories:
                    continue
                
                key = (metadata['story'], metadata['text_per_event'])
                experiments[key][metadata['struct_type']] = json_file
            
            # Process each experiment pair
            for (story, tpe), paths in experiments.items():
                if not paths['no_struct'] or not paths['with_struct']:
                    # Still add individual results if only one exists
                    for struct_type, json_path in paths.items():
                        if json_path:
                            try:
                                with open(json_path, 'r') as f:
                                    data = json.load(f)
                                qa_results = data.get('qa_results', [])
                                
                                correct = sum(1 for r in qa_results if r.get('is_correct', False))
                                total = len(qa_results)
                                
                                rows.append({
                                    'year': year,
                                    'model': model_name,
                                    'story': story,
                                    'text_per_event': tpe,
                                    'structure': 'no' if struct_type == 'no_struct' else 'yes',
                                    'accuracy': correct / total if total > 0 else 0,
                                    'correct': correct,
                                    'total': total,
                                    'intersection': False
                                })
                            except Exception as e:
                                print(f"  ⚠️  Error reading {json_path}: {e}")
                    continue
                
                # Both exist - calculate intersection
                try:
                    with open(paths['no_struct'], 'r') as f:
                        no_struct_data = json.load(f)
                    with open(paths['with_struct'], 'r') as f:
                        with_struct_data = json.load(f)
                    
                    no_struct_results = no_struct_data.get('qa_results', [])
                    with_struct_results = with_struct_data.get('qa_results', [])
                    
                    # Create question maps
                    no_struct_map = {r.get('question', ''): r for r in no_struct_results}
                    with_struct_map = {r.get('question', ''): r for r in with_struct_results}
                    
                    # Find intersection
                    common_questions = set(no_struct_map.keys()) & set(with_struct_map.keys())
                    
                    if common_questions:
                        no_struct_correct = sum(1 for q in common_questions if no_struct_map[q].get('is_correct', False))
                        with_struct_correct = sum(1 for q in common_questions if with_struct_map[q].get('is_correct', False))
                        total = len(common_questions)
                        
                        rows.append({
                            'year': year,
                            'model': model_name,
                            'story': story,
                            'text_per_event': tpe,
                            'structure': 'no',
                            'accuracy': no_struct_correct / total if total > 0 else 0,
                            'correct': no_struct_correct,
                            'total': total,
                            'intersection': True
                        })
                        
                        rows.append({
                            'year': year,
                            'model': model_name,
                            'story': story,
                            'text_per_event': tpe,
                            'structure': 'yes',
                            'accuracy': with_struct_correct / total if total > 0 else 0,
                            'correct': with_struct_correct,
                            'total': total,
                            'intersection': True
                        })
                except Exception as e:
                    print(f"  ⚠️  Error processing QA pair for {model_name}/story_{story}/t{tpe}: {e}")
    
    # Sort
    rows.sort(key=lambda x: (x['year'], x['model'], x['story'], x['text_per_event'], x['structure']))
    
    # Save per year
    for config in year_configs:
        year = config['year']
        year_rows = [r for r in rows if r['year'] == year]
        if not year_rows:
            continue
        
        save_dir = output_dir / year
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / 'temporal_qa_results.csv'
        
        fieldnames = ['year', 'model', 'story', 'text_per_event', 'structure', 'accuracy', 'correct', 'total', 'intersection']
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(year_rows)
        
        print(f"\n✅ Saved: {output_file} ({len(year_rows)} rows)")
    
    if rows:
        # Stats (intersection only)
        intersection_rows = [r for r in rows if r['intersection']]
        no_struct = [r for r in intersection_rows if r['structure'] == 'no']
        yes_struct = [r for r in intersection_rows if r['structure'] == 'yes']
        
        if no_struct and yes_struct:
            print(f"\n   📈 Stats (intersection questions):")
            print(f"      No structure:   {len(no_struct)} rows, mean acc = {sum(r['accuracy'] for r in no_struct)/len(no_struct):.4f}")
            print(f"      With structure: {len(yes_struct)} rows, mean acc = {sum(r['accuracy'] for r in yes_struct)/len(yes_struct):.4f}")
    else:
        print("  ⚠️  No temporal QA results found")
    
    return rows


# ============================================================================
# Summarization Export (with metric calculation)
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_rouge_scores(generated: str, reference: str, scorer) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    try:
        scores = scorer.score(reference, generated)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    except:
        return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}


def calculate_bleu_score(generated: str, reference: str) -> float:
    """Calculate BLEU score"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
        
        gen_clean = clean_text(generated)
        ref_clean = clean_text(reference)
        
        if not gen_clean or not ref_clean:
            return 0.0
        
        gen_tokens = word_tokenize(gen_clean)
        ref_tokens = word_tokenize(ref_clean)
        
        smoothing = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
    except:
        return 0.0



def calculate_meteor_score(generated: str, reference: str) -> float:
    """Calculate METEOR score"""
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        
        gen_clean = clean_text(generated)
        ref_clean = clean_text(reference)
        
        if not gen_clean or not ref_clean:
            return 0.0
        
        gen_tokens = word_tokenize(gen_clean)
        ref_tokens = word_tokenize(ref_clean)
        
        return meteor_score([ref_tokens], gen_tokens)
    except:
        return 0.0

def parse_summarization_experiment(exp_folder: str) -> Optional[Dict]:
    """Parse summarization experiment folder: summ_1_t1_abstract_no_struct_20260106_111516"""
    pattern = r'summ_(\d+)_t(\d+)_(\w+)_(no_struct|with_struct)_'
    match = re.search(pattern, exp_folder)
    
    if match:
        return {
            'story': int(match.group(1)),
            'text_per_event': int(match.group(2)),
            'label_type': match.group(3),
            'struct_type': match.group(4)
        }
    return None


def export_summarization(year_configs: List[Dict], output_dir: Path, use_bertscore: bool = False, device: str = 'cuda:0') -> List[Dict]:
    """Export summarization results with metric calculation"""
    print("\n" + "=" * 80)
    print("📊 SUMMARIZATION Results Export (with metric calculation)")
    print("=" * 80)
    
    # Import required packages
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    except ImportError:
        print("  ⚠️  Installing rouge-score...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'rouge-score', '-q'])
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        import nltk
        nltk.download('punkt', quiet=True)
    except:
        pass
    
    # BERTScore setup
    bert_scorer = None
    if use_bertscore:
        try:
            from bert_score import BERTScorer
            print(f"  🔧 Loading BERTScore model on {device}...")
            bert_scorer = BERTScorer(lang='en', device=device, rescale_with_baseline=True)
            print("  ✅ BERTScore ready")
        except Exception as e:
            print(f"  ⚠️  BERTScore not available: {e}")
            use_bertscore = False
    
    rows = []
    
    for config in year_configs:
        year = config['year']
        base_dir = config['base_dir']
        valid_stories = config['stories']
        
        print(f"\n  📁 Processing {year} from {base_dir}...")
        
        for model_name in get_all_models(base_dir):
            summ_dir = base_dir / model_name / 'summarization'
            if not summ_dir.exists():
                continue
            
            print(f"\n    Processing {model_name}...")
            
            for exp_dir in sorted(summ_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                
                # Find summaries JSON file
                json_files = list(exp_dir.glob('summaries_*.json'))
                if not json_files:
                    continue
                
                json_file = json_files[0]  # Use first match
                
                metadata = parse_summarization_experiment(exp_dir.name)
                if not metadata:
                    continue
                
                # Filter by valid stories for this year
                if metadata['story'] not in valid_stories:
                    continue
                
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    summaries = data.get('summaries', [])
                    if not summaries:
                        continue
                    
                    # Calculate metrics for each summary
                    all_scores = []
                    generated_list = []
                    reference_list = []
                    
                    for summary in summaries:
                        generated = summary.get('generated_summary', '')
                        reference = summary.get('reference_summary', '')
                        
                        scores = calculate_rouge_scores(generated, reference, scorer)
                        scores['bleu'] = calculate_bleu_score(generated, reference)
                        scores['meteor'] = calculate_meteor_score(generated, reference)
                        all_scores.append(scores)
                        
                        if use_bertscore:
                            generated_list.append(generated)
                            reference_list.append(reference)
                    
                    # BERTScore batch calculation
                    if use_bertscore and bert_scorer and generated_list:
                        try:
                            P, R, F1 = bert_scorer.score(generated_list, reference_list)
                            for i, f1 in enumerate(F1.tolist()):
                                if i < len(all_scores):
                                    all_scores[i]['bertscore'] = f1
                        except Exception as e:
                            print(f"    ⚠️  BERTScore error: {e}")
                    
                    # Average scores
                    avg_scores = {}
                    if all_scores:
                        for key in all_scores[0].keys():
                            avg_scores[key] = sum(s.get(key, 0) for s in all_scores) / len(all_scores)
                    
                    row = {
                        'year': year,
                        'model': model_name,
                        'story': metadata['story'],
                        'text_per_event': metadata['text_per_event'],
                        'label_type': metadata['label_type'],
                        'structure': 'no' if metadata['struct_type'] == 'no_struct' else 'yes',
                        'num_summaries': len(summaries),
                        **avg_scores
                    }
                    rows.append(row)
                    
                except Exception as e:
                    print(f"    ⚠️  Error processing {json_file}: {e}")
    
    # Sort
    rows.sort(key=lambda x: (x['year'], x['model'], x['story'], x['text_per_event'], x['label_type'], x['structure']))
    
    # Save per year
    bertscore_suffix = '_with_bertscore' if use_bertscore else ''
    fieldnames = ['year', 'model', 'story', 'text_per_event', 'label_type', 'structure', 
                  'num_summaries', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'meteor']
    if use_bertscore:
        fieldnames.append('bertscore')
    
    for config in year_configs:
        year = config['year']
        year_rows = [r for r in rows if r['year'] == year]
        if not year_rows:
            continue
        
        save_dir = output_dir / year
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / f'summarization_results{bertscore_suffix}.csv'
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in year_rows:
                filtered_row = {k: row.get(k) for k in fieldnames}
                writer.writerow(filtered_row)
        
        print(f"\n✅ Saved: {output_file} ({len(year_rows)} rows)")
    
    if rows:
        # Stats
        no_struct = [r for r in rows if r['structure'] == 'no']
        yes_struct = [r for r in rows if r['structure'] == 'yes']
        
        if no_struct and yes_struct:
            print(f"\n   📈 Stats:")
            print(f"      No structure:   {len(no_struct)} rows, mean ROUGE-L = {sum(r['rougeL_f'] for r in no_struct)/len(no_struct):.4f}")
            print(f"      With structure: {len(yes_struct)} rows, mean ROUGE-L = {sum(r['rougeL_f'] for r in yes_struct)/len(yes_struct):.4f}")
    else:
        print("  ⚠️  No summarization results found")
    
    return rows


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Export results by year')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'clustering', 'qa', 'summarization'],
                       help='Task to export (default: all)')
    parser.add_argument('--year', type=str, default='all',
                       choices=['2025', '2016', 'all'],
                       help='Year to export: 2025 (story 1,2,3), 2016 (story 25,218,227), or all')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for CSV files')
    parser.add_argument('--use-bertscore', action='store_true',
                       help='Calculate BERTScore for summarization (slow, GPU recommended)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for BERTScore (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()
    
    # Get year configurations
    year_configs = get_year_configs(args.year)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 Unified Results Export")
    print("=" * 80)
    print(f"   Year: {args.year}")
    print(f"   Task: {args.task}")
    print(f"   Output: {output_dir}")
    for config in year_configs:
        print(f"   - {config['year']}: {config['base_dir']} (stories: {config['stories']})")
    
    if args.task in ['all', 'clustering']:
        export_clustering(year_configs, output_dir)
    
    if args.task in ['all', 'qa']:
        export_temporal_qa(year_configs, output_dir)
    
    if args.task in ['all', 'summarization']:
        export_summarization(year_configs, output_dir, args.use_bertscore, args.device)
    
    print("\n" + "=" * 80)
    print("✅ Export completed!")
    print(f"   Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
