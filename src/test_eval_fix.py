
"""
Test script để kiểm tra hệ thống đánh giá sau khi sửa lỗi
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        from eval import RAGEvaluator
        print("RAGEvaluator import successful")
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False

def test_config_loading():
    """Test config loading"""
    print("Testing config loading...")
    try:
        from eval import RAGEvaluator
        evaluator = RAGEvaluator("config.yaml")
        print("Config loading successful")
        return True
    except Exception as e:
        print(f"Config loading error: {e}")
        return False

def test_qa_generation():
    """Test Q&A generation with a small sample"""
    print("Testing Q&A generation...")
    try:
        from eval import RAGEvaluator
        evaluator = RAGEvaluator("config.yaml")
        
        # Test với một file nhỏ
        wiki_data_path = "data/wiki_data"
        if not os.path.exists(wiki_data_path):
            print(f"Wiki data path not found: {wiki_data_path}")
            return False
        
        # Lấy chỉ 1 file để test
        wiki_files = list(Path(wiki_data_path).glob("*.json"))
        if not wiki_files:
            print("No wiki files found")
            return False
        
        print(f"Found {len(wiki_files)} wiki files")
        
        # Test tạo Q&A với 1 file
        qa_dataset = []
        test_file = wiki_files[0]
        print(f"Testing with file: {test_file.name}")
        
        import json
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        title = data["raw_content"]["title"]
        content_paragraphs = data["raw_content"]["content"]
        
        questions = evaluator._generate_questions_from_content(title, content_paragraphs, 2)
        print(f"Generated {len(questions)} questions")
        
        if questions:
            print("Q&A generation successful")
            print(f"Sample question: {questions[0]['question']}")
            return True
        else:
            print("No questions generated")
            return False
            
    except Exception as e:
        print(f"Q&A generation error: {e}")
        return False

def test_rag_engine():
    """Test RAG engine"""
    print("Testing RAG engine...")
    try:
        from engine.rag_engine import RAGEngine
        from indexer.pinecone import PineconeIndex
        from generator.groq_model import GroqModel
        import yaml
        
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        pinecone_config = config.get("pinecone")
        generator_config = config.get("generator")
        
        indexer = PineconeIndex(
            index_name=pinecone_config["index_name"],
            model_name=pinecone_config["model_name"],
            dimension=pinecone_config["dimension"],
        )
        generator = GroqModel(
            model_name=generator_config["model_name"],
        )
        engine = RAGEngine(indexer=indexer, generator=generator)
        
        # Test search
        test_query = "Đại học Quốc gia Hà Nội là gì?"
        answer = engine.generate_answer(test_query)
        
        print(f"RAG engine test successful")
        print(f"Query: {test_query}")
        print(f"Answer: {answer[:100]}...")
        return True
        
    except Exception as e:
        print(f"RAG engine error: {e}")
        return False

def main():
    """Run all tests"""
    print("TESTING EVALUATION SYSTEM FIXES")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_loading,
        test_qa_generation,
        test_rag_engine
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("All tests passed! Evaluation system is ready.")
    else:
        print("Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 