from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

def extract_with_font_sizes(pdf_path):
    for page_layout in extract_pages(pdf_path):
        print("\n--- New Page ---\n")
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if not text:
                    continue

                font_sizes = [char.size for line in element for char in line if isinstance(char, LTChar)]
                if not font_sizes:
                    continue

                avg_size = sum(font_sizes) / len(font_sizes)
                print(f"[{avg_size:.2f}] {text}")

extract_with_font_sizes("input/file02.pdf")
