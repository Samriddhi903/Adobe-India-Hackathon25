import os
import fitz  # PyMuPDF
import json
from extractor.heading_detector import detect_title, detect_headings

def extract_blocks(pdf_path):
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTChar

    blocks = []
    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    line_text = ""
                    font_sizes = []
                    fonts = []
                    # text_line might be LTChar, LTAnno, or iterable of LTChar/LTAnno
                    from pdfminer.layout import LTAnno
                    if isinstance(text_line, LTChar):
                        char = text_line
                        line_text += char.get_text()
                        font_sizes.append(char.size)
                        fonts.append(char.fontname)
                    elif isinstance(text_line, LTAnno):
                        # LTAnno represents a space or annotation, skip or add space
                        line_text += " "
                    else:
                        for char in text_line:
                            if isinstance(char, LTChar):
                                line_text += char.get_text()
                                font_sizes.append(char.size)
                                fonts.append(char.fontname)
                    if line_text.strip() and font_sizes:
                        # Use the most common font size and font name in the line
                        size = max(set(font_sizes), key=font_sizes.count)
                        font = max(set(fonts), key=fonts.count)
                        is_bold = 'Bold' in font
                        blocks.append({
                            'text': line_text.strip(),
                            'size': size,
                            'font': font,
                            'flags': 2 if is_bold else 0,
                            'page': page_num,
                            'bbox': [element.x0, element.y0, element.x1, element.y1]
                        })
    return blocks

def process_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        blocks = extract_blocks(pdf_path)
        title = detect_title(blocks)
        outline = detect_headings(blocks, title=title)
        output = {
            "title": title,
            "outline": outline
        }
        out_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0] + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Processed {pdf_file} -> {os.path.basename(out_path)}") 