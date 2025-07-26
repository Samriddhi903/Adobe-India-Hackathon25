#!/usr/bin/env python3

import os
import json
import time
from pathlib import Path
import argparse
from typing import Dict, List
from src.inference import HeadingPredictor
import pandas as pd

class ModelTester:
    def __init__(self, model_path: str):
        """Initialize the tester with model path"""
        self.model_path = model_path
        self.predictor = None
        self.results = []
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        try:
            self.predictor = HeadingPredictor(self.model_path)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_single_pdf(self, pdf_path: str, output_path: str = None) -> Dict:
        """Test a single PDF file"""
        if not self.predictor:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\n📄 Processing: {os.path.basename(pdf_path)}")
        
        # Record start time
        start_time = time.time()
        
        try:
            # Predict
            result = self.predictor.predict_pdf(pdf_path)
            processing_time = time.time() - start_time
            
            # Add metadata
            result['metadata'] = {
                'processing_time_seconds': round(processing_time, 2),
                'file_size_mb': round(os.path.getsize(pdf_path) / (1024*1024), 2),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 Results saved to: {output_path}")
            
            # Print summary
            self.print_result_summary(result, pdf_path)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return None
    
    def test_directory(self, input_dir: str, output_dir: str):
        """Test all PDFs in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in {input_dir}")
            return
        
        print(f"🔍 Found {len(pdf_files)} PDF files to process")
        print("=" * 60)
        
        # Process each PDF
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]", end=" ")
            
            # Generate output filename
            output_file = output_path / f"{pdf_file.stem}.json"
            
            # Process PDF
            result = self.test_single_pdf(str(pdf_file), str(output_file))
            
            if result:
                self.results.append({
                    'filename': pdf_file.name,
                    'success': True,
                    'processing_time': result['metadata']['processing_time_seconds'],
                    'file_size_mb': result['metadata']['file_size_mb'],
                    'title': result['title'],
                    'num_headings': len(result['outline'])
                })
            else:
                self.results.append({
                    'filename': pdf_file.name,
                    'success': False,
                    'processing_time': 0,
                    'file_size_mb': round(pdf_file.stat().st_size / (1024*1024), 2),
                    'title': '',
                    'num_headings': 0
                })
        
        # Generate summary report
        self.generate_summary_report(output_path)
    
    def print_result_summary(self, result: Dict, pdf_path: str):
        """Print a summary of the results"""
        print(f"⏱️  Processing time: {result['metadata']['processing_time_seconds']}s")
        print(f"📊 File size: {result['metadata']['file_size_mb']} MB")
        print(f"📋 Title: '{result['title']}'")
        print(f"🏷️  Found {len(result['outline'])} headings:")
        
        for heading in result['outline']:
            indent = "  " * (int(heading['level'][1]) - 1)  # H1=0, H2=1, H3=2 indents
            print(f"    {indent}{heading['level']}: {heading['text']} (page {heading['page']})")
    
    def generate_summary_report(self, output_dir: Path):
        """Generate a summary report of all tests"""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("📊 SUMMARY REPORT")
        print("="*60)
        
        # Overall statistics
        total_files = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total_files - successful
        
        total_time = sum(r['processing_time'] for r in self.results)
        avg_time = total_time / max(1, successful)
        
        total_size = sum(r['file_size_mb'] for r in self.results)
        
        print(f"📁 Total files processed: {total_files}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"⏱️  Average processing time: {avg_time:.2f}s")
        print(f"📊 Total data processed: {total_size:.2f} MB")
        
        if successful > 0:
            total_headings = sum(r['num_headings'] for r in self.results if r['success'])
            avg_headings = total_headings / successful
            print(f"🏷️  Total headings found: {total_headings}")
            print(f"🏷️  Average headings per document: {avg_headings:.1f}")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS")
        print("-" * 30)
        
        if successful > 0:
            times = [r['processing_time'] for r in self.results if r['success']]
            sizes = [r['file_size_mb'] for r in self.results if r['success']]
            
            print(f"⚡ Fastest processing: {min(times):.2f}s")
            print(f"🐌 Slowest processing: {max(times):.2f}s")
            print(f"📄 Smallest file: {min(sizes):.2f} MB")
            print(f"📚 Largest file: {max(sizes):.2f} MB")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS")
        print("-" * 80)
        print(f"{'Filename':<25} {'Status':<10} {'Time(s)':<8} {'Size(MB)':<10} {'Headings':<10}")
        print("-" * 80)
        
        for result in self.results:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{result['filename']:<25} {status:<10} {result['processing_time']:<8.2f} "
                  f"{result['file_size_mb']:<10.2f} {result['num_headings']:<10}")
        
        # Save detailed report
        report_file = output_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_files': total_files,
                    'successful': successful,
                    'failed': failed,
                    'total_processing_time': total_time,
                    'average_processing_time': avg_time,
                    'total_size_mb': total_size
                },
                'detailed_results': self.results
            }, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Test PDF heading extraction model')
    parser.add_argument('--model', '-m', 
                       default='models/heading_classifier.pth',
                       help='Path to trained model (default: models/heading_classifier.pth)')
    parser.add_argument('--input', '-i',
                       default='data/test/pdfs',
                       help='Input directory with test PDFs (default: data/test/pdfs)')
    parser.add_argument('--output', '-o',
                       default='data/test/annotations',
                       help='Output directory for results (default: data/test/annotations)')
    parser.add_argument('--single', '-s',
                       help='Test a single PDF file instead of directory')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Make sure you have trained the model first by running: python train.py")
        return
    
    # Initialize tester
    tester = ModelTester(args.model)
    
    # Load model
    if not tester.load_model():
        return
    
    print(f"🚀 Starting PDF heading extraction test...")
    print(f"📁 Model: {args.model}")
    
    if args.single:
        # Test single file
        print(f"📁 Input file: {args.single}")
        output_file = os.path.join(args.output, 
                                 os.path.basename(args.single).replace('.pdf', '.json'))
        tester.test_single_pdf(args.single, output_file)
    else:
        # Test directory
        print(f"📁 Input directory: {args.input}")
        print(f"📁 Output directory: {args.output}")
        tester.test_directory(args.input, args.output)
    
    print(f"\n🎉 Testing completed!")

if __name__ == "__main__":
    main()