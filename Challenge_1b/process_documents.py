import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Set environment variables for model cache paths if not already set
os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', '/app/models/torch')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/app/models/sentence_transformers')
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', '/app/models/huggingface/hub')

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
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DocumentProcessor:
    def __init__(self, model_path: str, embedding_model_name='all-MiniLM-L6-v2'):
        """Initialize the document processor with ML models and configurations."""
        self.heading_predictor = HeadingPredictor(model_path)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.stopwords = set(stopwords.words('english'))
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="typeform/distilbert-base-uncased-mnli", 
            device=-1,
            model_kwargs={"cache_dir": os.environ['TRANSFORMERS_CACHE']}
        )
        
        # Configuration parameters
        self.min_paragraph_length = 50
        self.max_sections = 10
        self.max_subsections = 10
        self.similarity_threshold = 0.2
        self.line_spacing_threshold = 10
        
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
        
        # Initialize instance variables
        self.input_data = None
        self.documents = []
        self.persona = ''
        self.job = ''
        self.config = {}

    def load_input(self, input_json_path: str):
        """Load input configuration from JSON file."""
        try:
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
            
            print(f"Loaded configuration for persona: {self.persona}")
            print(f"Task: {self.job}")
            
        except Exception as e:
            raise Exception(f"Error loading input file: {e}")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.,;:!?()-]', '', text)
        
        return text

    def is_relevant_content(self, text: str) -> bool:
        """Check if content is relevant to current persona."""
        if not text or len(text) < self.min_paragraph_length:
            return False
            
        text_lower = text.lower()
        
        # Check for antonyms first (exclusion criteria)
        if self.config['antonyms'] and any(term in text_lower for term in self.config['antonyms']):
            return False
            
        # Check for keywords (inclusion criteria)
        if self.config['keywords']:
            has_keywords = any(kw in text_lower for kw in self.config['keywords'])
            if not has_keywords:
                return False
                
        return True

    def group_blocks_into_paragraphs(self, blocks):
        """Group line blocks into paragraph blocks by merging lines close vertically on the same page."""
        if not blocks:
            return []
            
        # Sort blocks by page and vertical position (y1 descending)
        blocks = sorted(blocks, key=lambda b: (b['page'], -b['bbox'][3]))
        paragraphs = []
        
        current_para = {
            'text': blocks[0]['text'],
            'page': blocks[0]['page'],
            'bbox': blocks[0]['bbox'][:]
        }
        
        for i, block in enumerate(blocks[1:], 1):
            last_block = blocks[i-1]
            
            # Check if block is on the same page
            if block['page'] != current_para['page']:
                # Save current paragraph and start new one
                if current_para['text'].strip():
                    paragraphs.append(current_para)
                current_para = {
                    'text': block['text'],
                    'page': block['page'],
                    'bbox': block['bbox'][:]
                }
                continue
                
            # Check vertical distance between blocks
            vertical_gap = last_block['bbox'][1] - block['bbox'][3]
            
            if vertical_gap < self.line_spacing_threshold:
                # Merge text with space
                current_para['text'] += ' ' + block['text']
                # Expand bbox to include current block
                current_para['bbox'][0] = min(current_para['bbox'][0], block['bbox'][0])
                current_para['bbox'][1] = min(current_para['bbox'][1], block['bbox'][1])
                current_para['bbox'][2] = max(current_para['bbox'][2], block['bbox'][2])
                current_para['bbox'][3] = max(current_para['bbox'][3], block['bbox'][3])
            else:
                # Save current paragraph and start new one
                if current_para['text'].strip():
                    paragraphs.append(current_para)
                current_para = {
                    'text': block['text'],
                    'page': block['page'],
                    'bbox': block['bbox'][:]
                }
        
        # Append last paragraph
        if current_para['text'].strip():
            paragraphs.append(current_para)
            
        return paragraphs

    def extract_meaningful_paragraphs(self, blocks):
        """Extract and create meaningful paragraphs from document blocks."""
        if not blocks:
            return []
            
        # Group blocks into paragraphs first
        paragraph_blocks = self.group_blocks_into_paragraphs(blocks)
        
        # Process each paragraph block to create coherent text
        meaningful_paragraphs = []
        
        for para_block in paragraph_blocks:
            text = self.clean_text(para_block['text'])
            
            # Skip if too short or not relevant
            if not self.is_relevant_content(text):
                continue
                
            # Split into sentences and reconstruct meaningful paragraphs
            sentences = sent_tokenize(text)
            
            # Group sentences into coherent paragraphs (3-5 sentences per paragraph)
            current_paragraph = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Skip very short sentences
                    current_paragraph.append(sentence)
                    
                    # Create paragraph when we have 3-5 sentences or reach natural break
                    if len(current_paragraph) >= 3:
                        para_text = ' '.join(current_paragraph)
                        if len(para_text) >= self.min_paragraph_length:
                            meaningful_paragraphs.append({
                                'text': para_text,
                                'page': para_block['page'],
                                'bbox': para_block['bbox']
                            })
                        current_paragraph = []
            
            # Add remaining sentences as a paragraph if substantial
            if current_paragraph:
                para_text = ' '.join(current_paragraph)
                if len(para_text) >= self.min_paragraph_length:
                    meaningful_paragraphs.append({
                        'text': para_text,
                        'page': para_block['page'],
                        'bbox': para_block['bbox']
                    })
        
        return meaningful_paragraphs

    def extract_keywords(self, text: str) -> set:
        """Extract keywords from text, excluding stopwords."""
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = set(w for w in words if w not in self.stopwords and len(w) > 2)
        return keywords

    def expand_query(self, keywords: set) -> set:
        """Expand keywords using WordNet synonyms."""
        expanded = set(keywords)
        for kw in list(keywords)[:10]:  # Limit to prevent excessive expansion
            try:
                for syn in wordnet.synsets(kw)[:2]:  # Limit synsets
                    for lemma in syn.lemmas()[:2]:  # Limit lemmas
                        name = lemma.name().replace('_', ' ').lower()
                        if name not in self.stopwords and len(name) > 2:
                            expanded.add(name)
            except:
                continue
        return expanded

    def process_document(self, pdf_dir: str, document: dict):
        """Process a single PDF document and extract relevant content."""
        pdf_path = os.path.join(pdf_dir, document['filename'])
        print(f"Processing document: {document['filename']}")
        
        try:
            # Get document structure
            heading_result = self.heading_predictor.predict_pdf(pdf_path)
            title = heading_result.get('title', '')
            outline = heading_result.get('outline', [])
            
            # Extract text blocks
            blocks = extract_blocks(pdf_path)
            
            # Create meaningful paragraphs
            paragraphs = self.extract_meaningful_paragraphs(blocks)
            
            return title, outline, paragraphs
            
        except Exception as e:
            print(f"Error processing {document['filename']}: {e}")
            return '', [], []

    def rank_content(self, content_items, context_text: str):
        """Rank content by relevance to context using multiple signals."""
        if not content_items:
            return []
            
        # Prepare texts for embedding
        texts = [self.clean_text(item['text']) for item in content_items]
        
        try:
            # Get embeddings
            content_embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            context_embedding = self.embedding_model.encode([context_text], convert_to_tensor=True)
            
            # Calculate semantic similarities
            similarities = util.cos_sim(context_embedding, content_embeddings)[0]
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            similarities = [0.0] * len(texts)
        
        # Expand query keywords
        context_keywords = self.extract_keywords(context_text)
        expanded_keywords = self.expand_query(context_keywords)
        
        ranked_items = []
        
        for item, text, semantic_score in zip(content_items, texts, similarities):
            try:
                # Keyword relevance score
                text_words = set(re.findall(r'\b\w+\b', text.lower()))
                keyword_matches = len(expanded_keywords.intersection(text_words))
                keyword_score = min(keyword_matches / max(len(expanded_keywords), 1), 1.0)
                
                # Zero-shot classification for relevance
                try:
                    relevance_result = self.classifier(
                        text[:512],  # Limit text length for efficiency
                        ['relevant', 'irrelevant'],
                        hypothesis_template="This text is {} to the given context."
                    )
                    classification_score = relevance_result['scores'][0]
                except:
                    classification_score = 0.5  # Default neutral score
                
                # Combine scores with weights
                combined_score = (
                    0.4 * float(semantic_score) + 
                    0.3 * keyword_score + 
                    0.3 * classification_score
                )
                
                # Only include items above threshold
                if combined_score >= self.similarity_threshold:
                    ranked_items.append((item, combined_score))
                    
            except Exception as e:
                print(f"Error ranking item: {e}")
                continue
                
        return sorted(ranked_items, key=lambda x: x[1], reverse=True)

    def generate_output(self, pdf_dir: str) -> dict:
        """Generate the final output with extracted sections and subsections."""
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
            title, outline, paragraphs = self.process_document(pdf_dir, document)
            
            # Rank and select top sections
            if outline:
                ranked_sections = self.rank_content(
                    [{'text': h['text'], 'page': h['page']} for h in outline],
                    context_text
                )[:3]  # Top 3 sections per document
                
                for section, score in ranked_sections:
                    section_key = f"{document['filename']}:{section['text']}"
                    if section_key not in seen_sections and len(output["extracted_sections"]) < self.max_sections:
                        output["extracted_sections"].append({
                            "document": document['filename'],
                            "section_title": section['text'],
                            "importance_rank": len(output["extracted_sections"]) + 1,
                            "page_number": section['page']
                        })
                        seen_sections.add(section_key)
                        
                        # Get relevant paragraphs for this section
                        section_paragraphs = [p for p in paragraphs if p['page'] == section['page']]
                        
                        # If no paragraphs on same page, get nearby paragraphs
                        if not section_paragraphs:
                            section_paragraphs = [p for p in paragraphs 
                                                if abs(p['page'] - section['page']) <= 1][:3]
                        
                        # Rank paragraphs for this section
                        if section_paragraphs:
                            ranked_paragraphs = self.rank_content(section_paragraphs, context_text)[:2]
                            
                            for para, para_score in ranked_paragraphs:
                                para_key = f"{document['filename']}:{para['text'][:100]}"
                                if (para_key not in seen_subsections and 
                                    len(output["subsection_analysis"]) < self.max_subsections):
                                    
                                    output["subsection_analysis"].append({
                                        "document": document['filename'],
                                        "refined_text": para['text'],
                                        "page_number": para['page']
                                    })
                                    seen_subsections.add(para_key)
            
            # If no outline sections found, use top paragraphs directly
            else:
                ranked_paragraphs = self.rank_content(paragraphs, context_text)[:5]
                for para, score in ranked_paragraphs:
                    para_key = f"{document['filename']}:{para['text'][:100]}"
                    if (para_key not in seen_subsections and 
                        len(output["subsection_analysis"]) < self.max_subsections):
                        
                        output["subsection_analysis"].append({
                            "document": document['filename'],
                            "refined_text": para['text'],
                            "page_number": para['page']
                        })
                        seen_subsections.add(para_key)

        return output



def main():
    """Main execution function."""
    import os
    base_dir = '.'
    model_path = 'models/heading_classifier.pth'
    
    try:
        # Initialize processor
        processor = DocumentProcessor(model_path)
        
        # Identify all collection directories dynamically
        collections = [d for d in os.listdir(base_dir) if os.path.isdir(d) and d.lower().startswith('collection')]
        
        for collection_name in collections:
            input_json_path = os.path.join(collection_name, 'challenge1b_input.json')
            pdf_dir = os.path.join(collection_name, 'PDFs')
            output_path = os.path.join(collection_name, 'generated_output.json')
            
            print(f"Processing collection: {collection_name}")
            print(f"Input JSON: {input_json_path}")
            print(f"PDF directory: {pdf_dir}")
            print(f"Output path: {output_path}")
            
            # Load input and generate output
            processor.load_input(input_json_path)
            output = processor.generate_output(pdf_dir)
            
            # Save output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"Processing completed for {collection_name}")
            print(f"Output saved to {output_path}")
            print(f"Extracted {len(output['extracted_sections'])} sections")
            print(f"Generated {len(output['subsection_analysis'])} subsection analyses")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise



if __name__ == '__main__':
    main()
