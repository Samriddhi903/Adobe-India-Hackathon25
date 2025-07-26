import torch
import json
import os
from typing import List, Dict
from .model import HeadingClassifier, FeatureExtractor
from .data_preprocessing import PDFTextExtractor
import numpy as np

class HeadingPredictor:
    def __init__(self, model_path: str, model_name='distilbert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HeadingClassifier(model_name, num_classes=5).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.feature_extractor = FeatureExtractor()
        self.pdf_extractor = PDFTextExtractor()
        
        self.label_mapping = {
            0: 'TITLE',
            1: 'H1',
            2: 'H2',
            3: 'H3',
            4: 'BODY'
        }
        
    def predict_pdf(self, pdf_path: str) -> Dict:
        """Predict headings for a PDF file"""
        # Extract text elements
        text_elements = self.pdf_extractor.extract_text_with_features(pdf_path)
        
        if not text_elements:
            return {"title": "", "outline": []}
        
        # Predict labels
        predictions = self._predict_elements(text_elements)
        
        # Post-process predictions
        result = self._post_process_predictions(text_elements, predictions)
        
        return result
    
    def _predict_elements(self, text_elements: List[Dict]) -> List[str]:
        """Predict labels for text elements"""
        predictions = []
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(text_elements), batch_size):
            batch_elements = text_elements[i:i+batch_size]
            batch_predictions = self._predict_batch(batch_elements, text_elements)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def _predict_batch(self, batch_elements: List[Dict], all_elements: List[Dict]) -> List[str]:
        """Predict labels for a batch of elements"""
        texts = [elem['text'] for elem in batch_elements]
        features = []
        
        for elem in batch_elements:
            feature_vector = self.feature_extractor.extract_features(elem, all_elements)
            features.append(feature_vector)
        
        features = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            text_inputs = self.model.encode_text(texts)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            outputs = self.model(text_inputs, features)
            _, predicted = torch.max(outputs, 1)
            
            predictions = [self.label_mapping[pred.item()] for pred in predicted]
        
        return predictions
    
    def _post_process_predictions(self, text_elements: List[Dict], predictions: List[str]) -> Dict:
        """Post-process predictions into final format"""
        title = ""
        outline = []
        
        # Find title (first TITLE prediction or fallback)
        title_candidates = []
        for i, (elem, pred) in enumerate(zip(text_elements, predictions)):
            if pred == 'TITLE':
                title_candidates.append((elem, i))
        
        if title_candidates:
            # Choose the first title or the one with largest font
            title_elem = max(title_candidates, key=lambda x: x[0].get('font_size', 0))[0]
            title = title_elem['text']
        else:
            # Fallback: use first heading or first text
            for elem, pred in zip(text_elements, predictions):
                if pred in ['H1', 'H2', 'H3']:
                    title = elem['text']
                    break
            if not title and text_elements:
                title = text_elements[0]['text']
        
        # Collect headings
        for elem, pred in zip(text_elements, predictions):
            if pred in ['H1', 'H2', 'H3']:
                outline.append({
                    "level": pred,
                    "text": elem['text'],
                    "page": elem['page']
                })
        
        # Sort by page and position
        outline.sort(key=lambda x: (x['page'], x.get('y_position', 0)))
        
        return {
            "title": title,
            "outline": outline
        }

def process_directory(input_dir: str, output_dir: str, model_path: str):
    """Process all PDFs in a directory"""
    predictor = HeadingPredictor(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))
            
            print(f"Processing {filename}...")
            result = predictor.predict_pdf(pdf_path)
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Saved results to {output_path}")