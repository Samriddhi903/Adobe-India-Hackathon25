import numpy as np
from sklearn.cluster import KMeans
import unicodedata
import statistics

def clean_text(text):
    # Normalize Unicode, strip control chars, and clean whitespace
    def strip_unicode(t):
        return ''.join(c for c in t if unicodedata.category(c)[0] != 'C')
    return strip_unicode(text).strip().replace('\n', ' ')

def cluster_font_sizes(font_sizes, n_clusters=3):
    """
    Cluster font sizes into n_clusters (for H1, H2, H3) using KMeans.
    Returns: dict mapping font size to cluster label (0=largest, 2=smallest)
    """
    unique_sizes = list(set(font_sizes))
    font_sizes_np = np.array(unique_sizes).reshape(-1, 1)
    if len(font_sizes_np) < n_clusters:
        n_clusters = len(font_sizes_np)
    if n_clusters < 2:
        # Fallback: treat all as H1
        return {size: 0 for size in unique_sizes}
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(font_sizes_np)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(-cluster_centers)
        label_map = {orig: rank for rank, orig in enumerate(sorted_indices)}
        font_to_cluster = {}
        for size, label in zip(font_sizes_np.flatten(), kmeans.labels_):
            font_to_cluster[size] = label_map[label]
        return font_to_cluster
    except Exception:
        # Fallback: treat all as H1
        return {size: 0 for size in unique_sizes}

def group_textlines(text_elements, y_thresh=3, x_thresh=10):
    """
    Merge text spans on same line with similar font size.
    Each text_element is a dict: {text, font_size, page_num, x0, y0, x1, y1}
    """
    grouped = []
    current_line = []

    # Sort top-down, then left-right
    text_elements.sort(key=lambda b: (b['page_num'], -b['y0'], b['x0']))

    for elem in text_elements:
        if not current_line:
            current_line.append(elem)
            continue

        prev = current_line[-1]
        same_line = abs(prev['y0'] - elem['y0']) <= y_thresh and elem['page_num'] == prev['page_num']
        close_horizontally = abs(elem['x0'] - prev['x1']) <= x_thresh
        similar_font = abs(elem['font_size'] - prev['font_size']) <= 0.5

        if same_line and close_horizontally and similar_font:
            current_line.append(elem)
        else:
            # Merge current_line into one block
            merged_text = " ".join([e['text'] for e in current_line])
            avg_font = max([e['font_size'] for e in current_line])  # pick largest font in line
            bbox = (
                min([e['x0'] for e in current_line]),
                min([e['y0'] for e in current_line]),
                max([e['x1'] for e in current_line]),
                max([e['y1'] for e in current_line]),
            )
            grouped.append({
                "text": merged_text,
                "font_size": avg_font,
                "page_num": current_line[0]["page_num"],
                "bbox": bbox,
            })
            current_line = [elem]

    # Handle last line
    if current_line:
        merged_text = " ".join([e['text'] for e in current_line])
        avg_font = max([e['font_size'] for e in current_line])
        bbox = (
            min([e['x0'] for e in current_line]),
            min([e['y0'] for e in current_line]),
            max([e['x1'] for e in current_line]),
            max([e['y1'] for e in current_line]),
        )
        grouped.append({
            "text": merged_text,
            "font_size": avg_font,
            "page_num": current_line[0]["page_num"],
            "bbox": bbox,
        })

    return grouped
