# PDF Outline Extractor

## Overview
This solution extracts a structured outline (title, H1, H2, H3 headings) from PDF files and outputs them as JSON, as required for the Adobe India Hackathon 2025 (Round 1A).

## Features
- Accepts PDFs from `/app/input`, outputs JSONs to `/app/output`
- Extracts title and headings (H1, H2, H3) with page numbers
- No internet required, CPU-only, ≤10s for 50-page PDF
- Modular code: easy to extend for future rounds

## Folder Structure
```
.
├── Dockerfile
├── requirements.txt
├── main.py
├── extractor/
│   ├── __init__.py
│   ├── parser.py
│   ├── heading_detector.py
│   └── utils.py
├── input/         # (mounted by Docker)
├── output/        # (mounted by Docker)
└── README.md
```

## Approach
- **Text Extraction:** Uses PDFminer to extract text, font size, font flags, and position from each page.
- **Title Detection:** Picks the largest, top-most, centered text on page 1.
- **Heading Detection:** Clusters font sizes (KMeans) to assign H1/H2/H3, uses boldness/position heuristics.
- **No Hardcoding:** All logic is dynamic and generalizes to any PDF.

## How to Build and Run
### Navigate to Challenge_1a
```bash
  cd Challenge_1a
```
### Build Docker Image
```bash
docker build --platform=linux/amd64 -t pdf-outline-extractor:latest .
```
### Run Docker Container
```bash
docker run --rm   -v "$(pwd)/input:/app/input"   -v "$(pwd)/output:/app/output"   --network none   mysolutionname:somerandomidentifier
```

## Output Format
Each output JSON matches the schema in `sample_dataset/schema/output_schema.json`:
```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

## Dependencies
- pdfminer.six
- numpy
- scikit-learn

## Notes
- No internet access required or used
- No hardcoded heading logic
- All processing is offline and CPU-only
- Modular code for easy extension

## For Hackathon Judges
- Place PDFs in `/app/input` (mounted)
- Output JSONs will appear in `/app/output`
- Processing completes in <10s for 50-page PDFs
- Output strictly matches the required schema 
