import os
from extractor.parser import process_pdfs

if __name__ == "__main__":
    print("Starting PDF outline extraction...")
    process_pdfs(input_dir="input", output_dir="output")
    print("Completed PDF outline extraction.") 