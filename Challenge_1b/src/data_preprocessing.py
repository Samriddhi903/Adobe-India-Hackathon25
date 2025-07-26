import json
import os
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
import re
from typing import List, Dict, Tuple
import pandas as pd

class PDFTextExtractor:
    def __init__(self):
        self.text_elements = []
    
    def extract_text_with_features(self, pdf_path: str) -> List[Dict]:
        """Extract text with formatting features from PDF"""
        text_elements = []
        
        for page_num, page_layout in enumerate(extract_pages(pdf_path)):
            page_height = page_layout.height
            
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if not text or len(text.split()) < 1:
                        continue
                    
                    # Get font information
                    chars = list(element._objs[0]._objs) if element._objs else []
                    font_info = self._extract_font_info(chars)
                    
                    # Calculate position features
                    y_position = page_height - element.y1  # Normalize y position
                    
                    features = {
                        'text': text,
                        'page': page_num + 1,
                        'x_position': element.x0,
                        'y_position': y_position,
                        'width': element.width,
                        'height': element.height,
                        'font_size': font_info['font_size'],
                        'font_name': font_info['font_name'],
                        'is_bold': font_info['is_bold'],
                        'is_italic': font_info['is_italic'],
                        'char_count': len(text),
                        'word_count': len(text.split()),
                        'is_uppercase': text.isupper(),
                        'has_numbers': bool(re.search(r'\d', text)),
                        'line_spacing': self._calculate_line_spacing(element),
                        'is_centered': self._is_centered(element, page_layout.width),
                    }
                    
                    text_elements.append(features)
        
        return text_elements
    
    def _extract_font_info(self, chars) -> Dict:
        """Extract font information from characters"""
        if not chars:
            return {'font_size': 12, 'font_name': 'default', 'is_bold': False, 'is_italic': False}
        
        font_sizes = []
        font_names = []
        is_bold = False
        is_italic = False
        
        for char in chars:
            if isinstance(char, LTChar):
                font_sizes.append(char.height)
                font_names.append(char.fontname)
                if 'bold' in char.fontname.lower():
                    is_bold = True
                if 'italic' in char.fontname.lower():
                    is_italic = True
        
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        most_common_font = max(set(font_names), key=font_names.count) if font_names else 'default'
        
        return {
            'font_size': avg_font_size,
            'font_name': most_common_font,
            'is_bold': is_bold,
            'is_italic': is_italic
        }
    
    def _calculate_line_spacing(self, element) -> float:
        """Calculate line spacing"""
        return element.height / max(1, len(element.get_text().split('\n')))
    
    def _is_centered(self, element, page_width: float) -> bool:
        """Check if text element is centered"""
        center_x = element.x0 + element.width / 2
        page_center = page_width / 2
        return abs(center_x - page_center) < page_width * 0.1

class DatasetCreator:
    def __init__(self):
        self.label_mapping = {
            'TITLE': 0,
            'H1': 1,
            'H2': 2,
            'H3': 3,
            'BODY': 4
        }
        
    def create_training_data(self, pdf_dir: str, annotation_dir: str) -> pd.DataFrame:
        """Create training dataset from PDFs and annotations"""
        training_data = []
        extractor = PDFTextExtractor()
        
        for pdf_file in os.listdir(pdf_dir):
            if not pdf_file.endswith('.pdf'):
                continue
                
            pdf_path = os.path.join(pdf_dir, pdf_file)
            annotation_path = os.path.join(annotation_dir, pdf_file.replace('.pdf', '.json'))
            
            if not os.path.exists(annotation_path):
                continue
            
            # Extract text features
            text_elements = extractor.extract_text_with_features(pdf_path)
            
            # Load annotations
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            # Create labels
            labeled_data = self._create_labels(text_elements, annotations)
            training_data.extend(labeled_data)
        
        return pd.DataFrame(training_data)
    
    def _create_labels(self, text_elements: List[Dict], annotations: Dict) -> List[Dict]:
        """Create labels for text elements based on annotations"""
        labeled_data = []
        
        # Create mapping from annotation text to label
        annotation_map = {}
        
        # Add title
        if 'title' in annotations:
            annotation_map[annotations['title'].lower().strip()] = 'TITLE'
        
        # Add outline elements
        for item in annotations.get('outline', []):
            annotation_map[item['text'].lower().strip()] = item['level']
        
        # Label text elements
        for element in text_elements:
            text_lower = element['text'].lower().strip()
            
            # Check for exact match
            label = 'BODY'  # default
            for ann_text, ann_label in annotation_map.items():
                if self._text_similarity(text_lower, ann_text) > 0.9:
                    label = ann_label
                    break
            
            element['label'] = self.label_mapping.get(label, 4)
            labeled_data.append(element)
        
        return labeled_data
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        # Simple Jaccard similarity
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0