#!/usr/bin/env python3
"""
Demo script để test hệ thống đánh giá RAG
Chạy với một file wiki nhỏ để kiểm tra functionality
"""

import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def demo_qa_generation():
    """
    Demo tạo câu hỏi-đáp từ một file wiki
    """
    print("=== DEMO: Tạo câu hỏi-đáp từ dữ liệu wiki ===")
    
    # Đọc một file wiki mẫu
    wiki_file = Path("data/wiki_data/Đại_học_Quốc_gia_Hà_Nội.json")
    
    if not wiki_file.exists():
        print(f"Không tìm thấy file: {wiki_file}")
        return
    
    with open(wiki_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    title = data["raw_content"]["title"]
    content_paragraphs = data["raw_content"]["content"]
    
    print(f"Title: {title}")
    print(f"Số đoạn văn: {len(content_paragraphs)}")
    
    # Tạo câu hỏi mẫu
    question_templates = [
        f"{title} là gì?",
        f"Khi nào {title} được thành lập?",
        f"Địa chỉ của {title} ở đâu?",
        f"Vai trò của {title} trong giáo dục Việt Nam?",
    ]
    
    # Lọc đoạn văn có thông tin
    informative_paragraphs = [p for p in content_paragraphs if len(p.strip()) > 50]
    print(f"Số đoạn văn có thông tin: {len(informative_paragraphs)}")
    
    # Tạo Q&A pairs
    qa_pairs = []
    for question in question_templates[:2]:  # Chỉ lấy 2 câu hỏi đầu
        # Tìm đoạn văn phù hợp nhất
        if informative_paragraphs:
            relevant_paragraph = informative_paragraphs[0]  # Đơn giản hóa
            
            # Tạo câu trả lời
            sentences = relevant_paragraph.split('. ')
            answer = '. '.join(sentences[:2]) + '.' if len(sentences) >= 2 else relevant_paragraph
            
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "context": relevant_paragraph
            })
    
    # In kết quả
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n--- Q&A Pair {i} ---")
        print(f"Question: {qa['question']}")
        print(f"Answer: {qa['answer'][:200]}...")
        print(f"Context length: {len(qa['context'])} chars")
    
    return qa_pairs

def demo_similarity_calculation():
    """
    Demo tính toán similarity giữa các văn bản
    """
    print("\n=== DEMO: Tính toán similarity ===")
    
    # Khởi tạo sentence transformer
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Loaded sentence transformer model")
    except Exception as e:
        print(f"✗ Lỗi load model: {e}")
        return
    
    # Test texts
    text1 = "Đại học Quốc gia Hà Nội là một trong hai hệ thống đại học quốc gia của Việt Nam"
    text2 = "VNU Hanoi là trường đại học hàng đầu Việt Nam"
    text3 = "Hôm nay trời đẹp"
    
    texts = [text1, text2, text3]
    
    # Tính embeddings
    embeddings = model.encode(texts)
    
    # Tính similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    print("\nSimilarity Matrix:")
    for i, text_i in enumerate(texts):
        for j, text_j in enumerate(texts):
            if i <= j:
                sim = similarity_matrix[i][j]
                print(f"Text {i+1} vs Text {j+1}: {sim:.4f}")
    
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")

def demo_metrics_calculation():
    """
    Demo tính toán các metrics đánh giá
    """
    print("\n=== DEMO: Tính toán metrics ===")
    
    # Sample data
    reference = "Đại học Quốc gia Hà Nội là một trong hai hệ thống đại học quốc gia của Việt Nam, có trụ sở tại Hà Nội."
    candidate = "VNU Hanoi là trường đại học quốc gia của Việt Nam tại Hà Nội."
    
    print(f"Reference: {reference}")
    print(f"Candidate: {candidate}")
    
    # Simple word overlap (giả lập BLEU)
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())
    
    overlap = len(ref_words.intersection(cand_words))
    precision = overlap / len(cand_words) if cand_words else 0
    recall = overlap / len(ref_words) if ref_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nWord-level metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Semantic similarity
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([reference, candidate])
        semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"Semantic similarity: {semantic_sim:.4f}")
    except Exception as e:
        print(f"Không thể tính semantic similarity: {e}")

def main():
    """
    Chạy tất cả demos
    """
    print("RAG EVALUATION SYSTEM DEMO")
    print("=" * 50)
    
    try:
        # Demo 1: Tạo Q&A
        qa_pairs = demo_qa_generation()
        
        # Demo 2: Similarity calculation
        demo_similarity_calculation()
        
        # Demo 3: Metrics calculation
        demo_metrics_calculation()
        
        print("\n" + "=" * 50)
        print("Demo hoàn thành! Hệ thống đánh giá đã sẵn sàng.")
        print("Để chạy đánh giá đầy đủ, sử dụng: python eval.py")
        
    except Exception as e:
        print(f"Lỗi trong demo: {e}")
        print("Đảm bảo đã cài đặt dependencies: pip install -r eval_requirements.txt")

if __name__ == "__main__":
    main() 