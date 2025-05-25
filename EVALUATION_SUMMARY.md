# Tóm tắt Hệ thống Đánh giá RAG

## Tổng quan
Hệ thống đánh giá toàn diện cho mô hình RAG dựa trên dữ liệu Wikipedia tiếng Việt. Hệ thống bao gồm:

## Các thành phần chính

### 1. **Tạo Dataset Đánh giá Tự động** (`eval.py`)
- **Tự động tạo câu hỏi-đáp** từ 30+ files wiki data
- **10 template câu hỏi đa dạng** cho mỗi chủ đề
- **Tìm kiếm đoạn văn liên quan** sử dụng sentence similarity
- **Trích xuất câu trả lời reference** từ nội dung gốc

### 2. **Đánh giá Retrieval Performance**
- **Precision@K**: Độ chính xác documents được retrieve
- **Recall@K**: Độ bao phủ documents liên quan  
- **MRR**: Mean Reciprocal Rank
- **Hit Rate**: Tỷ lệ thành công tìm kiếm

### 3. **Đánh giá Generation Quality**
- **BLEU Score**: N-gram overlap với reference
- **ROUGE-1/2/L**: Overlap từ vựng và chuỗi con
- **BERT Score**: Semantic similarity
- **Faithfulness**: Độ trung thực với context
- **Answer Relevancy**: Độ liên quan với câu hỏi

## Cấu trúc thư mục

```
src/
├── eval.py                    # Hệ thống đánh giá chính
├── run_evaluation.py          # Script wrapper với error handling
├── demo_eval.py              # Demo test functionality
├── eval_requirements.txt     # Dependencies cho evaluation
├── EVALUATION_README.md      # Hướng dẫn chi tiết
└── app.py                    # Đã thêm --evaluate option
```

## Cách sử dụng

### Cài đặt dependencies
```bash
pip install -r src/eval_requirements.txt
```

### Chạy đánh giá đầy đủ
```bash
cd src
python app.py --evaluate
```

### Hoặc chạy trực tiếp
```bash
cd src
python eval.py --wiki_data data/wiki_data --output results.json
```

### Test demo
```bash
cd src
python demo_eval.py
```

## Output

### 1. **Generated Dataset** (`generated_qa_dataset.json`)
- ~150 câu hỏi-đáp từ wiki data
- Mỗi entry có: question, reference_answer, source_file, title, relevant_context

### 2. **Evaluation Results** (`evaluation_results.json`)
```json
{
  "dataset_info": {...},
  "retrieval_evaluation": {
    "avg_precision_at_k": 0.75,
    "avg_recall_at_k": 0.68,
    "avg_mrr": 0.82,
    "hit_rate": 0.90
  },
  "generation_evaluation": {
    "avg_bleu": 0.45,
    "avg_rouge1": 0.52,
    "avg_bert_score": 0.71,
    "avg_faithfulness": 0.65,
    "avg_answer_relevancy": 0.73
  },
  "overall_score": 0.68
}
```

## Tính năng nổi bật

### **Tự động hóa hoàn toàn**
- Không cần dataset có sẵn
- Tự tạo Q&A từ wiki content
- Tự động đánh giá cả retrieval và generation

### **Metrics toàn diện**
- 7 metrics khác nhau
- Bao phủ cả lexical và semantic similarity
- Đánh giá faithfulness và relevancy

### **Dễ sử dụng và mở rộng**
- Command line interface đơn giản
- Modular design dễ customize
- Comprehensive documentation

### **Robust error handling**
- Logging chi tiết
- Graceful degradation
- Clear error messages

## Kết quả mong đợi

Với dữ liệu wiki về các trường đại học và nhân vật Việt Nam:
- **Retrieval**: Hit rate cao (~0.8-0.9) do domain specific
- **Generation**: ROUGE scores trung bình (~0.4-0.6) 
- **Semantic**: BERT scores tốt (~0.6-0.8) với tiếng Việt
- **Overall**: Điểm tổng thể ~0.6-0.7

## Workflow đánh giá

1. **Load wiki data** → Parse JSON files
2. **Generate Q&A** → Create question-answer pairs  
3. **Evaluate Retrieval** → Test search performance
4. **Evaluate Generation** → Test answer quality
5. **Calculate Overall Score** → Combine all metrics
6. **Save Results** → JSON output + console display

## Lưu ý quan trọng

- **Sample size**: Generation evaluation chỉ test 20 samples để tiết kiệm thời gian
- **Similarity threshold**: 0.7 cho retrieval evaluation (có thể điều chỉnh)
- **Dependencies**: Cần internet để download sentence transformer model lần đầu
- **Memory**: Cần ~2GB RAM cho sentence transformer

Hệ thống đánh giá này cung cấp cái nhìn toàn diện về performance của RAG model và có thể được sử dụng để so sánh các cấu hình khác nhau hoặc theo dõi cải tiến theo thời gian. 