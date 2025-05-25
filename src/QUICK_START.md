# Quick Start - Hệ thống Đánh giá RAG

## 🚀 Bắt đầu nhanh

### Bước 1: Setup môi trường
```bash
cd src
python setup_evaluation.py
```

### Bước 2: Test hệ thống (Khuyến nghị)
```bash
python test_eval_fix.py
```

### Bước 3: Demo functionality
```bash
python demo_eval.py
```

### Bước 4: Chạy đánh giá đầy đủ
```bash
python app.py --evaluate
```

## 📋 Checklist trước khi chạy

- [ ] ✅ Python 3.7+
- [ ] ✅ File `config.yaml` tồn tại
- [ ] ✅ Thư mục `data/wiki_data` có dữ liệu
- [ ] ✅ Pinecone index đã được tạo và có dữ liệu
- [ ] ✅ API keys được cấu hình đúng

## 🔧 Troubleshooting nhanh

### Lỗi import
```bash
cd src
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Lỗi NLTK
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Lỗi sentence-transformers
```bash
pip install sentence-transformers==2.2.2
```

### Lỗi Pinecone
- Kiểm tra API key trong config.yaml
- Đảm bảo index đã được tạo: `python app.py --upsert`

## 📊 Kết quả mong đợi

Sau khi chạy thành công, bạn sẽ có:

1. **`generated_qa_dataset.json`** - Dataset câu hỏi-đáp (~150 entries)
2. **`evaluation_results.json`** - Kết quả đánh giá chi tiết
3. **Console output** - Hiển thị metrics và overall score

## 🎯 Metrics chính

- **Retrieval**: Precision@K, Recall@K, MRR, Hit Rate
- **Generation**: BLEU, ROUGE-1/2/L, BERT Score, Faithfulness, Answer Relevancy
- **Overall Score**: Điểm tổng hợp từ tất cả metrics

## 💡 Tips

- Lần đầu chạy sẽ mất thời gian download models (~500MB)
- Generation evaluation chỉ test 20 samples để tiết kiệm thời gian
- Có thể điều chỉnh threshold similarity trong code
- Kết quả được lưu tự động, không cần lo mất dữ liệu

## 📞 Hỗ trợ

Nếu gặp vấn đề, kiểm tra:
1. `evaluation.log` - Log chi tiết
2. `EVALUATION_README.md` - Hướng dẫn đầy đủ
3. `demo_eval.py` - Test từng phần riêng biệt 