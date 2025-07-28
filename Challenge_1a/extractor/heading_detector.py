from extractor.utils import clean_text, cluster_font_sizes, group_textlines
import re
from collections import Counter, defaultdict
import statistics
from difflib import SequenceMatcher

HEADING_LEVELS = {0: "H1", 1: "H2", 2: "H3"}
LEVEL_ORDER = {"H1": 0, "H2": 1, "H3": 2}

COMMON_FORM_LABELS = set(label.lower() for label in [
    "name", "date", "age", "relationship", "designation", "service", "pay", "amount", "rs.", "block", "single", "fare", "from", "the", "headquarters", "to", "home town", "place", "visit", "route", "persons", "s.no", "signature", "temporary", "permanent", "advance", "required"
])

def normalize(text):
    return clean_text(text).strip().lower()

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() > threshold

def detect_title(blocks):
    if not blocks:
        return "", None
    
    font_sizes = [b['size'] for b in blocks]
    if not font_sizes:
        return "", None
    
    max_size = max(font_sizes)
    max_blocks = [b for b in blocks if b['size'] == max_size]
    
    # New: Skip title detection if largest text is in bottom 50% of page
    for b in max_blocks:
        page_height = b['bbox'][3]  # Get page height from this block
        position_ratio = (page_height - b['bbox'][1]) / page_height
        if position_ratio > 0.5:  # More aggressive threshold (bottom 50%)
            print(f"Excluding bottom-positioned text from title: position_ratio={position_ratio:.2f}")
            excluded_text = " ".join(b["text"] for b in max_blocks)
            return "", excluded_text
    
    # Original title detection for other cases
    first_page = min(b['page'] for b in blocks)
    title_blocks = [b for b in max_blocks if b['page'] == first_page and (b.get('flags', 0) & 2)]
    if not title_blocks:
        title_blocks = max_blocks[:1]
    
    title_text = " ".join(b["text"] for b in title_blocks[:2])
    # Print position ratio for debug
    if title_blocks:
        position_ratio = round((page_height - title_blocks[0]['bbox'][1]) / page_height, 3)
        print(f"Title position ratio (height on page): {position_ratio}")
    return title_text, None

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
    text_clean = normalize(text)
    
    # Enhanced exclusion patterns (no hardcoding)
    if (re.fullmatch(r"\d+\.?", text_clean) or
        text_clean in COMMON_FORM_LABELS or
        re.fullmatch(r"^[\W_]+$", text_clean) or
        any(t in text_clean for t in ["www.", "http", "://", ".com"]) or
        re.search(r"^-+$", text)):  # Lines of hyphens
        return False
    
    # Dynamic length checks
    word_count = len(text_clean.split())
    char_count = len(text_clean)
    
    return (word_count <= 10 and
            char_count/word_count < 15 and  # Avoid long unbroken strings
            '\n' not in text and
            (is_bold or font_size > median_size * 1.3))

import re

def detect_headings(blocks, title=None, excluded_title_text=None, max_level=3, debug=False):
    # Initial setup and validation
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

    # Calculate document statistics
    median_size = statistics.median(font_sizes) if font_sizes else 0
    max_size = max(font_sizes) if font_sizes else 0

    # Ensure title and excluded_title_text are strings before normalization
    if title is None:
        title = ""
    elif not isinstance(title, str):
        title = str(title)

    if excluded_title_text is None:
        excluded_title_text = ""
    elif not isinstance(excluded_title_text, str):
        excluded_title_text = str(excluded_title_text)

    title_norm = normalize(title)
    page_height = blocks[0]['bbox'][3] if blocks else 792  # Default to letter size
    # Use group_textlines from utils to merge text lines
    # Convert blocks to expected format for group_textlines
    text_elements = []
    for b in blocks:
        text_elements.append({
            "text": b["text"],
            "font_size": b["size"],
            "page_num": b["page"],
            "x0": b["bbox"][0],
            "y0": b["bbox"][1],
            "x1": b["bbox"][2],
            "y1": b["bbox"][3],
        })
    merged_text_elements = group_textlines(text_elements)
    # Convert back to blocks format
    merged_blocks = []
    for e in merged_text_elements:
        merged_blocks.append({
            "text": e["text"],
            "size": e["font_size"],
            "page": e["page_num"],
            "bbox": e["bbox"],
        })

    # Handle single font case first
    font_counts = Counter(b['font'] for b in blocks)
    if len(font_counts) == 1:
        headings = [
            {
                'level': 'H1',
                'text': b['text'],
                'page': b['page'],
                'bbox': b['bbox']
            }
            for b in blocks if 'Bold' in b['font']
        ]
        if debug:
            print("All text is of one font. Bolded text considered as headings.")
        return headings

    # Process blocks by size
    blocks_by_size = defaultdict(list)
    for b in merged_blocks:
        blocks_by_size[b['size']].append(b)
    
    # Sort blocks by center proximity
    for size in blocks_by_size:
        blocks_by_size[size].sort(key=lambda b: abs(((b['bbox'][0] + b['bbox'][2]) / 2) - 300))

    # Detect heading candidates
    candidates = []
    numbering_pattern = re.compile(r'^(\d+)(\.\d+)*\.?')
    
    for size in blocks_by_size:
        level_str = size_to_level.get(size)
        if level_str is None or not level_str.startswith('H'):
            continue
            
        level_idx = int(level_str[1]) - 1
        if level_idx >= max_level:
            continue
            
        for b in blocks_by_size[size]:
            is_bold = bool(b.get('flags', 0) & 2)
            text = clean_text(b['text']).strip()
            
            if debug:
                print(f"[HEADINGS] Checking: {text} | Font: {b['size']} | Bold: {is_bold} | Page: {b['page']}")

            # Handle numbered headings
            match = numbering_pattern.match(text)
            if match:
                parts = [p for p in re.split(r'\.+', match.group(0)) if p.strip()]
                level_idx = min(len(parts) - 1, 2)  # Cap at H3 (level 2)

            # Special promotion for largest text at bottom
            if size == max_size:
                position_ratio = (page_height - b['bbox'][1]) / page_height
                if position_ratio > 0.5:  # More aggressive bottom detection (50%)
                    level_idx = 0  # Force H1
                    if debug:
                        print(f"Promoting bottom-positioned text to H1 (ratio={position_ratio:.2f}): {text}")
                    # Add candidate forcibly as H1 even if it fails is_heading_candidate
                    candidates.append((text, level_idx, b['page'], b['bbox'], is_bold, b['size']))
                    continue

            if is_heading_candidate(text, is_bold, b['size'], median_size):
                candidates.append((text, level_idx, b['page'], b['bbox'], is_bold, b['size']))
                if debug:
                    print(f"Candidate: {text} | Size: {b['size']} | Bold: {is_bold} | Level: {level_idx}")

    # Add forcibly excluded title text as H1 heading if provided
    if excluded_title_text:
        if debug:
            print(f"Adding excluded title text as H1 heading: {excluded_title_text}")
        # Add with page 1 and dummy bbox (0,0,0,0) or could be improved if bbox info available
        candidates.append((excluded_title_text, 0, 1, (0, 0, 0, 0), True, max_size))

    # Filter and deduplicate results
    seen_texts = set()
    unique_outline = []
    
    for text, level_idx, page, bbox, is_bold, size in candidates:
        text_norm = normalize(text)
        text_key = (text_norm, page, tuple(bbox))
        
        if (is_similar(text_norm, title_norm) and page == 1) or text_key in seen_texts:
            if debug:
                print(f"Filtered {'(title match)' if is_similar(text_norm, title_norm) else '(duplicate)'}: {text}")
            continue
            
        seen_texts.add(text_key)
        unique_outline.append({
            "level": HEADING_LEVELS[level_idx],
            "text": text,
            "page": page,
            "bbox": bbox
        })

    # Sort results by page, vertical position (higher first), and heading level
    unique_outline.sort(key=lambda h: (
        h['page'],
        -h['bbox'][1],  # Higher on page first
        LEVEL_ORDER[h['level']]  # H1 before H2, etc.
    ))
    
    return unique_outline
