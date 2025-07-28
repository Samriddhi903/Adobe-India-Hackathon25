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
    font_sizes = sorted({block["size"] for block in blocks})
    if len(font_sizes) < 2:
        return ""

    title_size = font_sizes[-1]  # Largest size
    if len(font_sizes) >= 2:
        body_size = font_sizes[0]
        title_size = font_sizes[-1] if font_sizes[-1] != body_size else None

    if not title_size:
        return ""

    first_page = min(b['page'] for b in blocks)

    # Pick all blocks with that size and bold flag on the first page only
    title_blocks = [b for b in blocks if b["size"] == title_size and (b.get("flags", 0) & 2) and b['page'] == first_page]
    if not title_blocks:
        # fallback to all blocks with that size on the first page only
        title_blocks = [b for b in blocks if b["size"] == title_size and b['page'] == first_page]
    if not title_blocks:
        return ""

    # Sort by page and descending bbox[1] (top coordinate)
    title_blocks = sorted(title_blocks, key=lambda b: (b["page"], -b["bbox"][1]))

    # Concatenate texts of first 2 title blocks separated by space
    title_text = " ".join(b["text"] for b in title_blocks[:2])
    return title_text

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

from collections import Counter

def map_font_size_to_levels(font_sizes):
    font_counts = Counter(font_sizes)
    sorted_by_size = sorted(font_counts.items(), key=lambda x: x[0], reverse=True)

    # Step 1: Assign most frequent size as Body
    body_font = max(font_counts.items(), key=lambda x: x[1])[0]
    mapping = {body_font: 'Body'}

    # Step 2: Assign Title (if any font is both large and rare)
    title_font = None
    for size, count in sorted_by_size:
        if size != body_font and count <= 2 and size > body_font + 2:
            title_font = size
            mapping[title_font] = 'Title'
            break

    # Step 3: Assign H1, H2, H3 from the next largest remaining sizes (not Body or Title)
    heading_level = 1
    for size, _ in sorted_by_size:
        if size in (body_font, title_font):
            continue
        if heading_level <= 3:
            mapping[size] = f'H{heading_level}'
            heading_level += 1

    return mapping

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
    # Heuristic: candidate headings are typically 1-liners, short length, and bold or large
    if len(text_clean.split()) > 10:
        return False
    if '\n' in text:
        return False
    if text.count(' ') > 8:
        return False  # long sentence, likely paragraph not heading
    if is_bold or font_size > median_size + 1.0:
        return True
    # Accept ALL CAPS with up to 5 words
    if text_clean.isupper() and len(text_clean.split()) <= 5:
        return True
    return False

import re

def detect_headings(blocks, title=None, max_level=3, debug=False):
    font_sizes = [b['size'] for b in blocks]
    if debug:
        print(f"Font sizes: {font_sizes}")
    size_to_level = map_font_size_to_levels(font_sizes)
    if debug:
        print(f"Size to level mapping: {size_to_level}")
    if not size_to_level or len(set(size_to_level.values())) < 2:
        if debug:
            print("No sufficient heading levels found, returning empty list.")
        return []
    median_size = statistics.median(font_sizes) if font_sizes else 0
    title_norm = normalize(title or "")
    merged_blocks = merge_adjacent_blocks(blocks, max_vertical_gap=5)
    candidates = []
    # Group blocks by font size
    blocks_by_size = defaultdict(list)
    for b in merged_blocks:
        blocks_by_size[b['size']].append(b)
    # For each font size, sort blocks by horizontal center proximity to 300 (page center)
    for size in blocks_by_size:
        blocks_by_size[size].sort(key=lambda b: abs(((b['bbox'][0] + b['bbox'][2]) / 2) - 300))
    seen_texts = []
    candidates = []
    numbering_pattern = re.compile(r'^(\d+)(\.\d+)*\.?')
    for size in blocks_by_size:
        level_str = size_to_level.get(size)
        if level_str is not None and level_str.startswith('H'):
            level_idx = int(level_str[1]) - 1  # Convert 'H1' to 0, 'H2' to 1, etc.
            if level_idx < max_level:
                for b in blocks_by_size[size]:
                    is_bold = bool(b.get('flags', 0) & 2)
                    text = clean_text(b['text']).strip()
                    if debug:
                        print(f"[HEADINGS] Checking: {text} | Font: {b['size']} | Bold: {is_bold} | Page: {b['page']}")
                    # Check numbering pattern for heading level override
                    match = numbering_pattern.match(text)
                    if match:
                        parts = [p for p in re.split(r'\.+', match.group(0)) if p.strip()]
                        if len(parts) == 1:
                            level_idx = 0  # H1
                        elif len(parts) == 2:
                            level_idx = 1  # H2
                        elif len(parts) >= 3:
                            level_idx = 2  # H3
                    if is_heading_candidate(text, is_bold, b['size'], median_size):
                        candidates.append((text, level_idx, b['page'], b['bbox'], is_bold, b['size']))
                        if debug:
                            print(f"Candidate: {text} | Size: {b['size']} | Bold: {is_bold} | Level: {level_idx}")
    filtered = []
    seen_texts = []
    for text, level_idx, page, bbox, is_bold, size in candidates:
        text_norm = normalize(text)
        if is_similar(text_norm, title_norm) and page == 1:
            if debug:
                print(f"Filtered (title match): {text} | Level: {HEADING_LEVELS[level_idx]} | Page: {page} | BBox: {bbox}")
            continue
        if any(is_similar(text_norm, t) and p == page for t, p in seen_texts):
            if debug:
                print(f"Filtered (duplicate): {text} | Level: {HEADING_LEVELS[level_idx]} | Page: {page} | BBox: {bbox}")
            continue
        seen_texts.append((text_norm, page))
        filtered.append({
            "level": HEADING_LEVELS[level_idx],
            "text": text,
            "page": page,
            "bbox": bbox
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
    unique_outline.sort(key=lambda h: (h['page'], -h['bbox'][1], level_order[h['level']]))
    return unique_outline
