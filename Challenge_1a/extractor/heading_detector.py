from extractor.utils import clean_text, cluster_font_sizes
import re
from collections import Counter, defaultdict
import statistics
from difflib import SequenceMatcher

HEADING_LEVELS = {0: "H1", 1: "H2", 2: "H3"}

COMMON_FORM_LABELS = set(label.lower() for label in [
    "name", "date", "age", "relationship", "designation", "service", "pay", "amount", "rs.", "block", "single", "fare", "from", "the", "headquarters", "to", "home town", "place", "visit", "route", "persons", "s.no", "signature", "temporary", "permanent", "advance", "required"
])

def normalize(text):
    return clean_text(text).strip().lower()

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() > threshold

def detect_title(blocks):
    page1_blocks = [b for b in blocks if b['page'] == 1]
    if not page1_blocks:
        return ""
    max_size = max(b['size'] for b in page1_blocks)
    candidates = [b for b in page1_blocks if abs((b['bbox'][0] + b['bbox'][2]) / 2 - 300) < 80 and b['size'] == max_size]
    lines = {}
    for b in candidates:
        y = round(b['bbox'][1] / 5) * 5
        lines.setdefault(y, []).append(b)
    title_lines = []
    for y in sorted(lines):
        line_blocks = sorted(lines[y], key=lambda b: b['bbox'][0])
        title_lines.append(' '.join(clean_text(b['text']) for b in line_blocks))
    return ' '.join(title_lines).strip()

def merge_adjacent_blocks(blocks, max_vertical_gap=5):
    merged = []
    current = None
    for b in sorted(blocks, key=lambda x: (x['page'], x['bbox'][1])):
        if current and b['page'] == current['page'] and abs(b['bbox'][1] - current['bbox'][3]) < max_vertical_gap and b['size'] == current['size']:
            current['text'] += " " + b['text']
            current['bbox'] = (current['bbox'][0], current['bbox'][1], b['bbox'][2], b['bbox'][3])
        else:
            if current:
                merged.append(current)
            current = b.copy()
    if current:
        merged.append(current)
    return merged

# Already imported: from extractor.utils import cluster_font_sizes
import statistics

def map_font_size_to_levels(font_sizes):
    return cluster_font_sizes(font_sizes, n_clusters=3)

def is_heading_candidate(text, is_bold, font_size, median_size):
    text_clean = text.strip().lower()
    # Exclude only pure numbers (e.g., "3.")
    if re.fullmatch(r"\d+\.?", text_clean):
        return False
    # Skip things that match common form labels exactly
    if text_clean in COMMON_FORM_LABELS:
        return False
    # Skip mostly punctuation
    if re.fullmatch(r"[\W_]+", text_clean):
        return False
    # Accept bold or large
    if is_bold or font_size > median_size + 1.5:
        return True
    # Accept ALL CAPS with up to 5 words
    if text_clean.isupper() and len(text_clean.split()) <= 5:
        return True
    return False

def detect_headings(blocks, title=None, max_level=3, debug=False):
    font_sizes = [b['size'] for b in blocks]
    size_to_level = map_font_size_to_levels(font_sizes)
    if not size_to_level or len(set(size_to_level.values())) < 2:
        return []
    median_size = statistics.median(font_sizes) if font_sizes else 0
    title_norm = normalize(title or "")
    merged_blocks = merge_adjacent_blocks(blocks, max_vertical_gap=5)
    candidates = []
    for b in merged_blocks:
        level_idx = size_to_level.get(b['size'])
        if level_idx is not None and level_idx in HEADING_LEVELS and level_idx < max_level:
            is_bold = bool(b.get('flags', 0) & 2)
            text = clean_text(b['text']).strip()
            if is_heading_candidate(text, is_bold, b['size'], median_size):
                candidates.append((text, level_idx, b['page'], b['bbox'], is_bold, b['size']))
                if debug:
                    print(f"Candidate: {text} | Size: {b['size']} | Bold: {is_bold} | Level: {level_idx}")
    filtered = []
    seen_texts = []
    for text, level_idx, page, bbox, is_bold, size in candidates:
        text_norm = normalize(text)
        if is_similar(text_norm, title_norm):
            if debug:
                print(f"Filtered (title match): {text} | Level: {HEADING_LEVELS[level_idx]} | Page: {page} | BBox: {bbox}")
            continue
        if any(is_similar(text_norm, t) for t in seen_texts):
            if debug:
                print(f"Filtered (duplicate): {text} | Level: {HEADING_LEVELS[level_idx]} | Page: {page} | BBox: {bbox}")
            continue
        seen_texts.append(text_norm)
        filtered.append({
            "level": HEADING_LEVELS[level_idx],
            "text": text,
            "page": page
        })
    seen = set()
    unique_outline = []
    level_order = {"H1": 0, "H2": 1, "H3": 2}
    seen_bboxes = set()
    for h in filtered:
        key = (h['level'], h['text'], h['page'], tuple(h.get('bbox', ())))
        if key not in seen_bboxes:
            unique_outline.append(h)
            seen_bboxes.add(key)
    unique_outline.sort(key=lambda h: (h['page'], level_order[h['level']]))
    return unique_outline
