import os
import json
import difflib
import pandas as pd
from collections import Counter

def jaccard_similarity(a, b):
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def heading_tuple(h):
    # Normalize for comparison
    return (h.get('level', '').strip().upper(), h.get('text', '').strip().lower(), int(h.get('page', 0)))

def outline_score(desired, actual):
    # Use set-based matching for headings (level, text, page)
    desired_set = set(heading_tuple(h) for h in desired)
    actual_set = set(heading_tuple(h) for h in actual)
    if not desired_set and not actual_set:
        return 1.0
    if not desired_set or not actual_set:
        return 0.0
    intersection = len(desired_set & actual_set)
    union = len(desired_set | actual_set)
    return intersection / union

def title_score(desired, actual):
    # Use normalized string similarity
    desired = (desired or '').strip().lower()
    actual = (actual or '').strip().lower()
    if not desired and not actual:
        return 1.0
    if not desired or not actual:
        return 0.0
    return difflib.SequenceMatcher(None, desired, actual).ratio()

def main():
    outputs_dir = os.path.join(os.path.dirname(__file__), 'sample_dataset', 'outputs')
    test_outputs_dir = os.path.join(os.path.dirname(__file__), 'output')
    files = [f for f in os.listdir(outputs_dir) if f.endswith('.json')]
    results = []
    for fname in files:
        desired_path = os.path.join(outputs_dir, fname)
        actual_path = os.path.join(test_outputs_dir, fname)
        if not os.path.exists(actual_path):
            print(f"Missing actual output for {fname}")
            continue
        with open(desired_path, encoding='utf-8') as f:
            desired = json.load(f)
        with open(actual_path, encoding='utf-8') as f:
            actual = json.load(f)
        t_score = title_score(desired.get('title', ''), actual.get('title', ''))
        o_score = outline_score(desired.get('outline', []), actual.get('outline', []))
        total = 0.5 * t_score + 0.5 * o_score
        results.append({
            'file': fname,
            'title_score': round(t_score, 3),
            'outline_score': round(o_score, 3),
            'total_score': round(total, 3)
        })
    df = pd.DataFrame(results)
    avg_title = df['title_score'].mean() if not df.empty else 0.0
    avg_outline = df['outline_score'].mean() if not df.empty else 0.0
    avg_total = df['total_score'].mean() if not df.empty else 0.0
    print(df.to_string(index=False))
    print(f"\nAverage Title Score: {avg_title:.3f}")
    print(f"Average Outline Score: {avg_outline:.3f}")
    print(f"Average Total Score: {avg_total:.3f}")
    df.to_csv(os.path.join(os.path.dirname(__file__), 'accuracy_scores.csv'), index=False)

if __name__ == "__main__":
    main() 