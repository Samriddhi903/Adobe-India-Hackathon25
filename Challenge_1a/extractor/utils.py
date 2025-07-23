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