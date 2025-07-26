import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from .model import HeadingClassifier, FeatureExtractor

class HeadingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = FeatureExtractor()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Text
        text = str(row['text'])
        
        # Features
        features = self.feature_extractor.extract_features(row.to_dict())
        
        # Label
        label = int(row['label'])
        
        return {
            'text': text,
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class HeadingTrainer:
    def __init__(self, model_name='distilbert-base-uncased', num_classes=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HeadingClassifier(model_name, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
    def prepare_data(self, df: pd.DataFrame, test_size=0.2):
        """Prepare training and validation datasets"""
        train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
        
        train_dataset = HeadingDataset(train_df, self.model.tokenizer)
        val_dataset = HeadingDataset(val_df, self.model.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            texts = batch['text']
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Encode texts
            text_inputs = self.model.encode_text(texts)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(text_inputs, features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                texts = batch['text']
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                text_inputs = self.model.encode_text(texts)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                
                outputs = self.model(text_inputs, features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        return total_loss / len(val_loader), accuracy
    
    def train(self, train_loader, val_loader, epochs=10):
        """Train the model"""
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'models/heading_classifier.pth')
                print(f"New best model saved with accuracy: {best_val_acc:.4f}")
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()