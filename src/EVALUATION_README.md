# Hệ thống Đánh giá RAG Model

Hệ thống đánh giá toàn diện cho mô hình RAG (Retrieval-Augmented Generation) dựa trên dữ liệu Wikipedia tiếng Việt.

## Tính năng

### 1. Tạo Dataset Đánh giá Tự động
- Tự động tạo câu hỏi-đáp từ dữ liệu wiki
- Sử dụng template câu hỏi đa dạng
- Tìm kiếm đoạn văn liên quan nhất cho mỗi câu hỏi
- Trích xuất câu trả lời reference từ nội dung

### 2. Đánh giá Retrieval
- **Precision@K**: Độ chính xác của top-K documents được retrieve
- **Recall@K**: Độ bao phủ của documents liên quan
- **MRR (Mean Reciprocal Rank)**: Vị trí trung bình của document đầu tiên liên quan
- **Hit Rate**: Tỷ lệ query có ít nhất 1 document liên quan trong top-K

### 3. Đánh giá Generation
- **BLEU Score**: Đánh giá độ tương đồng n-gram với reference
- **ROUGE Score**: Đánh giá overlap từ vựng (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERT Score**: Đánh giá semantic similarity sử dụng sentence embeddings
- **Faithfulness**: Độ trung thực của câu trả lời với context được retrieve
- **Answer Relevancy**: Độ liên quan của câu trả lời với câu hỏi

## Cài đặt

### 1. Setup tự động (Khuyến nghị)
```bash
cd src
python setup_evaluation.py
```

### 2. Setup thủ công

#### Cài đặt dependencies
```bash
cd src
pip install -r eval_requirements.txt
```

#### Cài đặt NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

#### Download sentence transformer model
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Sẽ tự động download
```

### 3. Kiểm tra cài đặt
```bash
cd src
python demo_eval.py
```

## Sử dụng

### 1. Chạy đánh giá đầy đủ
```bash
cd src
python eval.py --wiki_data data/wiki_data --output evaluation_results.json --config config.yaml
```

### 2. Sử dụng script wrapper
```bash
cd src
python run_evaluation.py
```

### 3. Sử dụng trong code
```python
from eval import RAGEvaluator

# Khởi tạo evaluator
evaluator = RAGEvaluator("config.yaml")

# Chạy đánh giá
results = evaluator.run_full_evaluation(
    wiki_data_path="data/wiki_data",
    output_path="my_results.json"
)

# In kết quả
print(f"Overall Score: {results['overall_score']:.4f}")
```

## Cấu trúc Output

### 1. Dataset được tạo (`generated_qa_dataset.json`)
```json
[
  {
    "question": "Đại học Quốc gia Hà Nội là gì?",
    "reference_answer": "Đại học Quốc gia Hà Nội là một trong hai hệ thống...",
    "source_file": "path/to/source.json",
    "title": "Đại học Quốc gia Hà Nội",
    "relevant_context": "Đoạn văn liên quan..."
  }
]
```

### 2. Kết quả đánh giá (`evaluation_results.json`)
```json
{
  "dataset_info": {
    "total_qa_pairs": 150,
    "evaluation_sample_size": 20,
    "wiki_files_processed": 30
  },
  "retrieval_evaluation": {
    "avg_precision_at_k": 0.75,
    "avg_recall_at_k": 0.68,
    "avg_mrr": 0.82,
    "hit_rate": 0.90
  },
  "generation_evaluation": {
    "avg_bleu": 0.45,
    "avg_rouge1": 0.52,
    "avg_rouge2": 0.38,
    "avg_rougeL": 0.48,
    "avg_bert_score": 0.71,
    "avg_faithfulness": 0.65,
    "avg_answer_relevancy": 0.73
  },
  "overall_score": 0.68
}
```

## Tùy chỉnh

### 1. Thay đổi template câu hỏi
Chỉnh sửa `question_templates` trong method `_generate_questions_from_content()`:

```python
question_templates = [
    f"{title} là gì?",
    f"Khi nào {title} được thành lập?",
    # Thêm template mới...
]
```

### 2. Điều chỉnh threshold similarity
Thay đổi threshold trong `evaluate_retrieval()`:

```python
if similarity > 0.7:  # Thay đổi threshold này
    relevant_retrieved += 1
```

### 3. Thay đổi số lượng câu hỏi per file
```bash
python eval.py --wiki_data data/wiki_data --num_questions 10
```

## Metrics Giải thích

### Retrieval Metrics
- **Precision@K**: Tỷ lệ documents liên quan trong top-K results
- **Recall@K**: Tỷ lệ documents liên quan được tìm thấy trong top-K
- **MRR**: Trung bình nghịch đảo của rank của document liên quan đầu tiên
- **Hit Rate**: Tỷ lệ queries có ít nhất 1 document liên quan trong top-K

### Generation Metrics
- **BLEU**: Đo overlap n-gram giữa generated và reference text
- **ROUGE-1/2/L**: Đo overlap unigram/bigram/longest common subsequence
- **BERT Score**: Đo semantic similarity sử dụng contextualized embeddings
- **Faithfulness**: Đo mức độ generated answer faithful với retrieved context
- **Answer Relevancy**: Đo mức độ generated answer relevant với question

## Troubleshooting

### 1. Lỗi import
```bash
# Đảm bảo đang ở thư mục src
cd src
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 2. Lỗi Pinecone connection
- Kiểm tra API key và index name trong config.yaml
- Đảm bảo index đã được tạo và có dữ liệu

### 3. Lỗi memory
- Giảm sample size cho generation evaluation
- Xử lý từng batch nhỏ thay vì toàn bộ dataset

### 4. Lỗi encoding
- Đảm bảo tất cả files được save với encoding UTF-8
- Kiểm tra locale settings

## Mở rộng

### 1. Thêm metrics mới
Implement method mới trong class `RAGEvaluator`:

```python
def evaluate_custom_metric(self, qa_dataset):
    # Implementation
    pass
```

### 2. Thêm loại câu hỏi mới
Mở rộng `question_templates` hoặc implement logic tạo câu hỏi phức tạp hơn.

### 3. Đánh giá theo domain
Phân chia dataset theo domain và đánh giá riêng biệt.

## Liên hệ

Nếu có vấn đề hoặc đề xuất cải tiến, vui lòng tạo issue hoặc liên hệ trực tiếp. 