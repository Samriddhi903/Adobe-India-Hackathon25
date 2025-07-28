import os
os.environ['TORCH_HOME'] = 'C:/models/torch'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'C:/models/sentence_transformers'
os.environ['TRANSFORMERS_CACHE'] = 'C:/models/huggingface/hub'
import json
import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet, stopwords
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from src.inference import HeadingPredictor
from extractor.parser import extract_blocks

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DocumentProcessor:
    def __init__(self, model_path: str, embedding_model_name='all-distilroberta-v1'):
        self.heading_predictor = HeadingPredictor(model_path)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.stopwords = set(stopwords.words('english'))
        self.classifier = pipeline("zero-shot-classification", 
                                 model="typeform/distilbert-base-uncased-mnli", 
                                 device=-1,
                                 model_kwargs={"cache_dir": os.environ['TRANSFORMERS_CACHE']})
        
        # Configuration parameters
        self.min_paragraph_length = 50  # Minimum characters for valid content
        self.max_sections = 10          # Maximum sections to return
        self.max_subsections = 10       # Maximum subsections to return
        self.similarity_threshold = 0.2 # Minimum similarity score
        
        # Persona configurations
        self.persona_config = {
            'travel_planner': {
                'keywords': ['itinerary', 'visit', 'hotel', 'restaurant', 'activity', 'guide', 'trip'],
                'antonyms': ['history', 'theory', 'research', 'permanent', 'student'],
                'context': "travel planning and tourism information"
            },
            'researcher': {
                'keywords': ['study', 'methodology', 'data', 'analysis', 'results', 'research'],
                'antonyms': ['tourism', 'itinerary', 'vacation', 'holiday'],
                'context': "academic research and technical information"
            },
            'architect': {
                'keywords': ['design', 'structure', 'materials', 'blueprint', 'construction', 'building'],
                'antonyms': ['temporary', 'itinerary', 'travel', 'event'],
                'context': "architectural details and structural information"
            }
        }

    def load_input(self, input_json_path):
        with open(input_json_path, 'r', encoding='utf-8') as f:
            self.input_data = json.load(f)
        self.documents = self.input_data.get('documents', [])
        self.persona = self.input_data.get('persona', {}).get('role', '').lower()
        self.job = self.input_data.get('job_to_be_done', {}).get('task', '')
        
        # Get persona-specific configuration
        self.config = self.persona_config.get(self.persona, {
            'keywords': [],
            'antonyms': [],
            'context': f"information relevant to {self.persona}"
        })

    def clean_text(self, text):
        """Clean and normalize text content"""
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        return text

    def is_relevant_content(self, text):
        """Check if content is relevant to current persona"""
        text_lower = text.lower()
        
        # Length check
        if len(text) < self.min_paragraph_length:
            return False
            
        # Keyword check
        if self.config['keywords']:
            has_keywords = any(kw in text_lower for kw in self.config['keywords'])
            if not has_keywords:
                return False
                
        # Antonym check
        if any(term in text_lower for term in self.config['antonyms']):
            return False
            
        return True

    def group_sentences(self, sentences):
        """Group sentences into meaningful paragraphs"""
        paragraphs = []
        current_para = []
        
        for sent in sentences:
            current_para.append(sent)
            if len(current_para) >= 2:  # Group every 2-3 sentences
                para_text = ' '.join(current_para)
                if len(para_text) >= self.min_paragraph_length:
                    paragraphs.append(para_text)
                current_para = []
                
        # Add remaining sentences
        if current_para:
            para_text = ' '.join(current_para)
            if len(para_text) >= self.min_paragraph_length:
                paragraphs.append(para_text)
                
        return paragraphs

    def extract_keywords(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = set(w for w in words if w not in self.stopwords and len(w) > 2)
        return keywords

    def expand_query(self, keywords):
        expanded = set(keywords)
        for kw in keywords:
            try:
                for syn in wordnet.synsets(kw):
                    for lemma in syn.lemmas():
                        name = lemma.name().replace('_', ' ').lower()
                        if name not in self.stopwords and len(name) > 2:
                            expanded.add(name)
            except:
                continue
        return expanded

    def process_document(self, pdf_dir, document):
        pdf_path = os.path.join(pdf_dir, document['filename'])
        print(f"Processing document: {document['filename']}")
        
        heading_result = self.heading_predictor.predict_pdf(pdf_path)
        title = heading_result.get('title', '')
        outline = heading_result.get('outline', [])
        blocks = extract_blocks(pdf_path)
        
        # Filter and clean blocks
        filtered_blocks = []
        for block in blocks:
            clean_text = self.clean_text(block['text'])
            if self.is_relevant_content(clean_text):
                block['text'] = clean_text
                filtered_blocks.append(block)
                
        return title, outline, filtered_blocks

    def rank_content(self, content_items, context_text):
        """Rank content by relevance to context"""
        if not content_items:
            return []
            
        # Prepare texts for embedding
        texts = [self.clean_text(item['text']) for item in content_items]
        
        # Get embeddings
        content_embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        context_embedding = self.embedding_model.encode([context_text], convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(context_embedding, content_embeddings)[0]
        
        # Combine with keyword relevance
        expanded_keywords = self.expand_query(self.extract_keywords(context_text))
        ranked_items = []
        
        for item, text, score in zip(content_items, texts, similarities):
            # Keyword relevance
            kw_score = 0.3 if any(kw in text.lower() for kw in expanded_keywords) else 0
            
            # Zero-shot classification
            candidate_labels = ['relevant', 'irrelevant']
            relevance_score = self.classifier(
                text, 
                candidate_labels, 
                hypothesis_template="This text is about {}."
            )['scores'][0]
            
            combined_score = 0.4 * score + 0.3 * kw_score + 0.3 * relevance_score
            
            if combined_score >= self.similarity_threshold:
                ranked_items.append((item, combined_score))
                
        return sorted(ranked_items, key=lambda x: x[1], reverse=True)

    def generate_output(self, pdf_dir):
        context_text = f"{self.config['context']}. Task: {self.job}"
        
        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in self.documents],
                "persona": self.persona,
                "job_to_be_done": self.job,
                "processing_timestamp": datetime.datetime.utcnow().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

        seen_sections = set()
        seen_subsections = set()
        
        for document in self.documents:
            title, outline, blocks = self.process_document(pdf_dir, document)
            
            # Rank sections
            ranked_sections = self.rank_content(
                [{'text': h['text'], 'page': h['page']} for h in outline],
                context_text
            )[:3]  # Top 3 sections per document
            
            for section, score in ranked_sections:
                section_key = f"{document['filename']}:{section['text']}"
                if section_key not in seen_sections:
                    output["extracted_sections"].append({
                        "document": document['filename'],
                        "section_title": section['text'],
                        "importance_rank": len(output["extracted_sections"]) + 1,
                        "page_number": section['page']
                    })
                    seen_sections.add(section_key)
                    
                    # Get relevant blocks for this section
                    section_blocks = [b for b in blocks if b['page'] == section['page']]
                    
                    # Extract and group sentences
                    all_sentences = []
                    for block in section_blocks:
                        all_sentences.extend(sent_tokenize(block['text']))
                    
                    # Create meaningful paragraphs
                    paragraphs = self.group_sentences(all_sentences)
                    
                    # Rank paragraphs
                    ranked_paragraphs = self.rank_content(
                        [{'text': p} for p in paragraphs],
                        context_text
                    )[:2]  # Top 2 paragraphs per section
                    
                    for para, para_score in ranked_paragraphs:
                        para_key = f"{document['filename']}:{para['text'][:100]}"
                        if para_key not in seen_subsections:
                            output["subsection_analysis"].append({
                                "document": document['filename'],
                                "refined_text": para['text'],
                                "page_number": section['page']
                            })
                            seen_subsections.add(para_key)

        # Enforce maximum output size
        output["extracted_sections"] = output["extracted_sections"][:self.max_sections]
        output["subsection_analysis"] = output["subsection_analysis"][:self.max_subsections]
        
        return output

def main():
    input_json_path = 'Challenge_1b/Collection 1/challenge1b_input.json'
    pdf_dir = 'Challenge_1b/Collection 1/PDFs'
    model_path = 'Challenge_1b/models/heading_classifier.pth'
    
    processor = DocumentProcessor(model_path)
    processor.load_input(input_json_path)
    output = processor.generate_output(pdf_dir)
    
    output_path = 'Challenge_1b/Collection 1/generated_output.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    main()