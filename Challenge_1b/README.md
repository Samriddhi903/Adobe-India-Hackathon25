# Challenge 1b: Multi-Collection PDF Analysis

## Overview

This project provides an advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases. The approach leverages machine learning models, natural language processing, and semantic ranking to identify important sections and subsections from diverse PDF documents.

The core of the solution is a custom-trained heading classification model that predicts document structure (titles, headings, body text) using a DistilBERT-based classifier enhanced with layout and font features. This model was trained on a custom dataset created by hand-annotating PDFs provided by the hackathon organizers, combined with additional annotations to improve accuracy.

Persona-based content filtering and semantic ranking using sentence embeddings and zero-shot classification ensure that extracted content is highly relevant to the user's task.

## Project Structure

```
Challenge_1b/
├── Collection 1/                    # Travel Planning documents and configs
├── Collection 2/                    # Adobe Acrobat Learning documents and configs
├── Collection 3/                    # Recipe Collection documents and configs
├── Dockerfile                      # Docker environment setup
├── extractor/                     # PDF text extraction utilities
├── models/                        # Trained ML models (e.g., heading_classifier.pth)
├── process_documents.py           # Main document processing pipeline
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── src/                          # Model training and inference code
└── test_model.py                 # Model testing utilities
```
## How to Run

The main processing script is `process_documents.py`. It loads input configuration JSON files specifying documents, persona, and task, processes the PDFs to extract relevant sections, and outputs structured JSON results.

### Navigate to Challenge_1b directory

```bash
cd Challenge_1b
```
**Ensure that all input scenarios follow the same directory struture, i.e. make a new Collection x and save the pdfs in Collection x/PDFs and input json path is Collection x/challenge1b_input.json**
### Build Docker
```bash
docker build -t challenge1b-python-slim .
```
### Run Docker
```bash
docker run --rm -it --network none -w /app challenge1b-python-slim python process_documents.py 
```

## Main Components

- **DocumentProcessor (`process_documents.py`)**: Orchestrates the document processing pipeline including loading inputs, extracting text blocks, predicting headings, filtering content based on persona, ranking sections semantically, and generating output JSON.

- **HeadingPredictor (`src/inference.py`)**: Loads the custom-trained heading classification model and predicts document structure labels (TITLE, H1, H2, H3, BODY) for text elements extracted from PDFs.

- **Text Extraction (`extractor/parser.py`)**: Uses `pdfminer.six` to extract text blocks with font and layout features from PDF pages, which are then used for heading prediction and paragraph formation.

## Input and Output Formats

### Input JSON

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [{ "filename": "doc.pdf", "title": "Title" }],
  "persona": { "role": "User Persona" },
  "job_to_be_done": { "task": "Use case description" }
}
```

### Output JSON

```json
{
  "metadata": {
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "processing_timestamp": "ISO timestamp"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```

## Notes

- The heading classification model (`heading_classifier.pth`) was trained using a custom dataset created by hand-annotating PDFs provided by the hackathon organizers, combined with additional PDFs and annotations. The training pipeline is implemented in the `src/` directory.

- Persona-based filtering uses keyword and antonym lists to include or exclude content relevant to the user's role and task.

- Semantic ranking combines sentence embeddings from `sentence-transformers` and zero-shot classification from `transformers` to score and select the most relevant sections and paragraphs.

- The output JSON provides a structured summary of important sections and detailed subsection analyses for downstream use.

---
