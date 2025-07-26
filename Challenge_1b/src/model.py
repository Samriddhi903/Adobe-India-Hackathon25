import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class HeadingClassifier(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=5, feature_dim=15):
        super(HeadingClassifier, self).__init__()
        
        # Text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # Feature dimensions
        text_dim = self.text_encoder.config.hidden_size
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, text_inputs, features):
        # Encode text
        text_outputs = self.text_encoder(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        # Process features
        feature_embeddings = self.feature_processor(features)
        
        # Combine embeddings
        combined = torch.cat([text_embeddings, feature_embeddings], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        return logits
    
    def encode_text(self, texts, max_length=128):
        """Encode text inputs"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

class FeatureExtractor:
    @staticmethod
    def extract_features(text_element: dict, context_elements: list = None) -> np.ndarray:
        """Extract numerical features from text element"""
        
        # Basic features
        features = [
            text_element.get('font_size', 12) / 20.0,  # Normalized font size
            text_element.get('x_position', 0) / 1000.0,  # Normalized x position
            text_element.get('y_position', 0) / 1000.0,  # Normalized y position
            text_element.get('width', 0) / 1000.0,  # Normalized width
            text_element.get('height', 0) / 100.0,  # Normalized height
            1.0 if text_element.get('is_bold', False) else 0.0,
            1.0 if text_element.get('is_italic', False) else 0.0,
            text_element.get('char_count', 0) / 100.0,  # Normalized char count
            text_element.get('word_count', 0) / 20.0,  # Normalized word count
            1.0 if text_element.get('is_uppercase', False) else 0.0,
            1.0 if text_element.get('has_numbers', False) else 0.0,
            text_element.get('line_spacing', 12) / 20.0,  # Normalized line spacing
            1.0 if text_element.get('is_centered', False) else 0.0,
            text_element.get('page', 1) / 50.0,  # Normalized page number
        ]
        
        # Context feature (relative font size)
        if context_elements:
            avg_font_size = np.mean([elem.get('font_size', 12) for elem in context_elements])
            relative_font_size = text_element.get('font_size', 12) / max(avg_font_size, 1.0)
            features.append(relative_font_size)
        else:
            features.append(1.0)
        
        return np.array(features, dtype=np.float32)