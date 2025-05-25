#!/usr/bin/env python3
"""
Script để chạy đánh giá mô hình RAG
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from eval import RAGEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

def main():
    """
    Chạy đánh giá RAG với error handling
    """
    try:
        logging.info("Bắt đầu đánh giá mô hình RAG...")
        
        # Kiểm tra file config
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            logging.error(f"Không tìm thấy file config: {config_path}")
            return
        
        # Kiểm tra thư mục wiki data
        wiki_data_path = "data/wiki_data"
        if not os.path.exists(wiki_data_path):
            logging.error(f"Không tìm thấy thư mục wiki data: {wiki_data_path}")
            return
        
        # Khởi tạo evaluator
        evaluator = RAGEvaluator(config_path)
        
        # Chạy đánh giá
        results = evaluator.run_full_evaluation(
            wiki_data_path=wiki_data_path,
            output_path="evaluation_results.json"
        )
        
        logging.info("Đánh giá hoàn thành thành công!")
        logging.info(f"Kết quả được lưu tại: evaluation_results.json")
        logging.info(f"Dataset được tạo tại: generated_qa_dataset.json")
        
    except Exception as e:
        logging.error(f"Lỗi trong quá trình đánh giá: {str(e)}")
        logging.error("Chi tiết lỗi:", exc_info=True)

if __name__ == "__main__":
    main() 