#!/usr/bin/env python3
"""
CheckEval for Paired Samples (NO TRUNCATION)
Properly paired: same (model, story, window, tpe) with struct & no_struct
"""

import os
import json
import argparse
from openai import OpenAI

CHECKEVAL_QUESTIONS = {
    "faithfulness": "Does the summary contain only facts that are present in or inferable from the reference? (No hallucinated information)",
    "coverage": "Does the summary cover the key information from the reference?",
    "non_redundancy": "Does the summary avoid unnecessary repetition?",
    "relevance": "Does the summary focus on the main topic and include only relevant information?",
    "coherence": "Does the summary have logical flow and good organization?",
}

def evaluate_single(client, reference: str, generated: str, model: str = "gpt-4o-mini"):
    """Evaluate a single summary"""
    questions_text = "\n".join([f"- {k}: {v}" for k, v in CHECKEVAL_QUESTIONS.items()])
    
    prompt = f"""## Reference Summary:
{reference}

## Generated Summary:
{generated}

## Evaluation (Answer Yes or No for each):
{questions_text}

Output as JSON only:
{{"faithfulness": "Yes/No", "coverage": "Yes/No", "non_redundancy": "Yes/No", "relevance": "Yes/No", "coherence": "Yes/No"}}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip()
        if answer.startswith('```'):
            answer = answer.split('```')[1]
            if answer.startswith('json'):
                answer = answer[4:]
        
        parsed = json.loads(answer)
        results = {}
        for k in CHECKEVAL_QUESTIONS.keys():
            val = parsed.get(k, 'No')
            results[k] = 1 if str(val).lower().startswith('yes') else 0
        
        results['score'] = sum(results.values()) / len(CHECKEVAL_QUESTIONS)
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        return {k: 0 for k in CHECKEVAL_QUESTIONS.keys()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--limit', type=int, default=100, help='Number of samples to evaluate')
    args = parser.parse_args()
    
    client = OpenAI(api_key=args.api_key)
    
    # Load samples
    sample_file = '../dataset/checkeval_samples_paired_7k.json'
    with open(sample_file) as f:
        samples = json.load(f)
    
    if args.limit:
        samples = samples[:args.limit]
    print(f"Evaluating {len(samples)} samples...")
    
    results = []
    output_file = '../dataset/checkeval_results_paired_7k.json'
    
    # Resume if exists
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)}")
        samples = samples[len(results):]
    
    total = len(results) + len(samples)
    for i, s in enumerate(samples):
        print(f"[{len(results)+1}/{total}] {s['model'][:20]}...", end=" ")
        
        scores = evaluate_single(client, s['reference'], s['generated'])
        
        results.append({
            **s,
            **scores
        })
        print(f"score={scores.get('score', 0):.2f}")
        
        # Save every 100
        if (len(results)) % 100 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f)
            print(f"  [Saved {len(results)}]")
    
    # Save results
    output_file = '../dataset/checkeval_results_paired_7k.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for k in CHECKEVAL_QUESTIONS.keys():
        avg = sum(r[k] for r in results) / len(results)
        print(f"  {k}: {avg*100:.1f}%")
    
    overall = sum(r['score'] for r in results) / len(results)
    print(f"\n  OVERALL: {overall*100:.1f}%")
    
    # By struct vs no_struct
    struct = [r for r in results if r['is_struct']]
    no_struct = [r for r in results if not r['is_struct']]
    
    if struct and no_struct:
        struct_avg = sum(r['score'] for r in struct) / len(struct)
        no_struct_avg = sum(r['score'] for r in no_struct) / len(no_struct)
        print(f"\n  With Struct: {struct_avg*100:.1f}% (n={len(struct)})")
        print(f"  No Struct: {no_struct_avg*100:.1f}% (n={len(no_struct)})")
    
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    main()
