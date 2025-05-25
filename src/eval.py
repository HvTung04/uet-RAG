import json
import yaml
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict

# Evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# RAG components
from indexer.pinecone import PineconeIndex
from generator.groq_model import GroqModel
from engine.rag_engine import RAGEngine

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class RAGEvaluator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Initialize RAG components
        pinecone_config = self.config.get("pinecone")
        generator_config = self.config.get("generator")
        
        self.indexer = PineconeIndex(
            index_name=pinecone_config["index_name"],
            model_name=pinecone_config["model_name"],
            dimension=pinecone_config["dimension"],
        )
        self.generator = GroqModel(
            model_name=generator_config["model_name"],
        )
        self.rag_engine = RAGEngine(indexer=self.indexer, generator=self.generator)
        
        # Initialize evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.smoothing = SmoothingFunction().method1
        
    def generate_qa_dataset(self, wiki_data_path: str, num_questions_per_file: int = 5) -> List[Dict]:
        """
        Tạo bộ dữ liệu câu hỏi-đáp từ wiki_data
        """
        print("Generating Q&A dataset from wiki data...")
        qa_dataset = []
        
        wiki_files = list(Path(wiki_data_path).glob("*.json"))
        
        for file_path in tqdm(wiki_files, desc="Processing wiki files"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            title = data["raw_content"]["title"]
            content_paragraphs = data["raw_content"]["content"]
            
            # Tạo câu hỏi từ nội dung
            questions = self._generate_questions_from_content(title, content_paragraphs, num_questions_per_file)
            
            for question_data in questions:
                qa_dataset.append({
                    "question": question_data["question"],
                    "reference_answer": question_data["answer"],
                    "source_file": str(file_path),
                    "title": title,
                    "relevant_context": question_data["context"]
                })
        
        return qa_dataset
    
    def _generate_questions_from_content(self, title: str, content_paragraphs: List[str], num_questions: int) -> List[Dict]:
        """
        Tạo câu hỏi từ nội dung wiki
        """
        questions = []
        
        # Các template câu hỏi
        question_templates = [
            f"{title} là gì?",
            f"Khi nào {title} được thành lập?",
            f"Ai là người sáng lập {title}?",
            f"Địa chỉ của {title} ở đâu?",
            f"Tại sao {title} quan trọng?",
            f"Lịch sử phát triển của {title} như thế nào?",
            f"Các đặc điểm chính của {title} là gì?",
            f"Vai trò của {title} trong giáo dục Việt Nam?",
            f"Cơ cấu tổ chức của {title} ra sao?",
            f"Thành tựu nổi bật của {title}?"
        ]
        
        # Lọc các đoạn văn có thông tin
        informative_paragraphs = [p for p in content_paragraphs if len(p.strip()) > 50]
        
        if not informative_paragraphs:
            return questions
        
        # Tạo câu hỏi và câu trả lời
        selected_questions = random.sample(question_templates, min(num_questions, len(question_templates)))
        
        for question in selected_questions:
            # Tìm đoạn văn phù hợp nhất cho câu hỏi
            relevant_paragraph = self._find_most_relevant_paragraph(question, informative_paragraphs)
            
            if relevant_paragraph:
                # Tạo câu trả lời từ đoạn văn
                answer = self._extract_answer_from_paragraph(question, relevant_paragraph, title)
                
                questions.append({
                    "question": question,
                    "answer": answer,
                    "context": relevant_paragraph
                })
        
        return questions
    
    def _find_most_relevant_paragraph(self, question: str, paragraphs: List[str]) -> str:
        """
        Tìm đoạn văn liên quan nhất đến câu hỏi
        """
        if not paragraphs:
            return ""
        
        # Sử dụng sentence similarity để tìm đoạn văn phù hợp
        question_embedding = self.sentence_model.encode([question])
        paragraph_embeddings = self.sentence_model.encode(paragraphs)
        
        similarities = cosine_similarity(question_embedding, paragraph_embeddings)[0]
        best_idx = np.argmax(similarities)
        
        return paragraphs[best_idx]
    
    def _extract_answer_from_paragraph(self, question: str, paragraph: str, title: str) -> str:
        """
        Trích xuất câu trả lời từ đoạn văn
        """
        # Đơn giản hóa: lấy 2-3 câu đầu của đoạn văn làm câu trả lời
        sentences = paragraph.split('. ')
        if len(sentences) >= 2:
            return '. '.join(sentences[:2]) + '.'
        else:
            return paragraph
    
    def evaluate_retrieval(self, qa_dataset: List[Dict], top_k: int = 5) -> Dict:
        """
        Đánh giá chất lượng retrieval
        """
        print("Evaluating retrieval performance...")
        
        retrieval_scores = {
            'precision_at_k': [],
            'recall_at_k': [],
            'mrr': [],  # Mean Reciprocal Rank
            'hit_rate': []
        }
        
        for qa_item in tqdm(qa_dataset, desc="Evaluating retrieval"):
            query = qa_item["question"]
            relevant_context = qa_item["relevant_context"]
            
            try:
                # Tìm kiếm documents
                search_results = self.indexer.search(query, top_k)
                retrieved_texts = []
                
                if hasattr(search_results, 'matches') and search_results.matches:
                    for match in search_results.matches:
                        if hasattr(match, 'metadata') and match.metadata:
                            text = match.metadata.get('text', '')
                            if text:
                                retrieved_texts.append(str(text))
                
                # Tính precision và recall
                relevant_retrieved = 0
                reciprocal_rank = 0
                
                for i, retrieved_text in enumerate(retrieved_texts):
                    # Kiểm tra xem text có liên quan không (sử dụng similarity)
                    similarity = self._calculate_text_similarity(relevant_context, retrieved_text)
                    if similarity > 0.7:  # threshold
                        relevant_retrieved += 1
                        if reciprocal_rank == 0:
                            reciprocal_rank = 1 / (i + 1)
                
                precision = relevant_retrieved / len(retrieved_texts) if retrieved_texts else 0
                recall = relevant_retrieved / 1  # Giả sử có 1 document liên quan
                hit_rate = 1 if relevant_retrieved > 0 else 0
                
                retrieval_scores['precision_at_k'].append(precision)
                retrieval_scores['recall_at_k'].append(recall)
                retrieval_scores['mrr'].append(reciprocal_rank)
                retrieval_scores['hit_rate'].append(hit_rate)
                
            except Exception as e:
                print(f"Error processing retrieval for query '{query}': {e}")
                # Thêm giá trị mặc định
                retrieval_scores['precision_at_k'].append(0.0)
                retrieval_scores['recall_at_k'].append(0.0)
                retrieval_scores['mrr'].append(0.0)
                retrieval_scores['hit_rate'].append(0.0)
        
        # Tính trung bình
        avg_scores = {
            'avg_precision_at_k': np.mean(retrieval_scores['precision_at_k']),
            'avg_recall_at_k': np.mean(retrieval_scores['recall_at_k']),
            'avg_mrr': np.mean(retrieval_scores['mrr']),
            'hit_rate': np.mean(retrieval_scores['hit_rate'])
        }
        
        return avg_scores
    
    def evaluate_generation(self, qa_dataset: List[Dict]) -> Dict:
        """
        Đánh giá chất lượng generation
        """
        print("Evaluating generation performance...")
        
        generation_scores = {
            'bleu_scores': [],
            'rouge_scores': {'rouge1': [], 'rouge2': [], 'rougeL': []},
            'bert_scores': [],
            'faithfulness_scores': [],
            'answer_relevancy_scores': []
        }
        
        for qa_item in tqdm(qa_dataset, desc="Evaluating generation"):
            question = qa_item["question"]
            reference_answer = qa_item["reference_answer"]
            
            try:
                # Tạo câu trả lời từ RAG
                generated_answer = self.rag_engine.generate_answer(question)
                
                # BLEU Score
                bleu_score = self._calculate_bleu_score(reference_answer, generated_answer)
                generation_scores['bleu_scores'].append(bleu_score)
                
                # ROUGE Score
                rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    generation_scores['rouge_scores'][metric].append(rouge_scores[metric].fmeasure)
                
                # BERTScore (sử dụng sentence similarity)
                bert_score = self._calculate_text_similarity(reference_answer, generated_answer)
                generation_scores['bert_scores'].append(bert_score)
                
                # Faithfulness (độ trung thực với context)
                search_results = self.indexer.search(question, 3)
                if hasattr(search_results, 'matches') and search_results.matches:
                    context_texts = []
                    for match in search_results.matches:
                        if hasattr(match, 'metadata') and match.metadata:
                            text = match.metadata.get('text', '')
                            if text:
                                context_texts.append(str(text))
                    context_text = " ".join(context_texts)
                else:
                    context_text = ""
                
                faithfulness = self._calculate_faithfulness(generated_answer, context_text)
                generation_scores['faithfulness_scores'].append(faithfulness)
                
                # Answer Relevancy (độ liên quan của câu trả lời với câu hỏi)
                relevancy = self._calculate_text_similarity(question, generated_answer)
                generation_scores['answer_relevancy_scores'].append(relevancy)
                
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                # Thêm giá trị mặc định để tránh lỗi
                generation_scores['bleu_scores'].append(0.0)
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    generation_scores['rouge_scores'][metric].append(0.0)
                generation_scores['bert_scores'].append(0.0)
                generation_scores['faithfulness_scores'].append(0.0)
                generation_scores['answer_relevancy_scores'].append(0.0)
        
        # Tính trung bình
        avg_scores = {
            'avg_bleu': np.mean(generation_scores['bleu_scores']),
            'avg_rouge1': np.mean(generation_scores['rouge_scores']['rouge1']),
            'avg_rouge2': np.mean(generation_scores['rouge_scores']['rouge2']),
            'avg_rougeL': np.mean(generation_scores['rouge_scores']['rougeL']),
            'avg_bert_score': np.mean(generation_scores['bert_scores']),
            'avg_faithfulness': np.mean(generation_scores['faithfulness_scores']),
            'avg_answer_relevancy': np.mean(generation_scores['answer_relevancy_scores'])
        }
        
        return avg_scores
    
    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Tính BLEU score
        """
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smoothing)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Tính độ tương đồng giữa hai văn bản
        """
        embeddings = self.sentence_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Tính độ trung thực của câu trả lời với context
        """
        if not context.strip():
            return 0.0
        
        # Sử dụng similarity để đánh giá faithfulness
        return self._calculate_text_similarity(answer, context)
    
    def run_full_evaluation(self, wiki_data_path: str, output_path: str = "evaluation_results.json"):
        """
        Chạy đánh giá toàn diện
        """
        print("Starting full RAG evaluation...")
        
        # 1. Tạo dataset
        qa_dataset = self.generate_qa_dataset(wiki_data_path)
        print(f"Generated {len(qa_dataset)} Q&A pairs")
        
        # Lưu dataset
        with open("generated_qa_dataset.json", "w", encoding="utf-8") as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        
        # 2. Đánh giá retrieval
        retrieval_results = self.evaluate_retrieval(qa_dataset)
        
        # 3. Đánh giá generation (chỉ lấy một phần để tiết kiệm thời gian)
        sample_size = min(20, len(qa_dataset))
        sample_dataset = random.sample(qa_dataset, sample_size)
        generation_results = self.evaluate_generation(sample_dataset)
        
        # 4. Tổng hợp kết quả
        final_results = {
            "dataset_info": {
                "total_qa_pairs": len(qa_dataset),
                "evaluation_sample_size": sample_size,
                "wiki_files_processed": len(list(Path(wiki_data_path).glob("*.json")))
            },
            "retrieval_evaluation": retrieval_results,
            "generation_evaluation": generation_results,
            "overall_score": self._calculate_overall_score(retrieval_results, generation_results)
        }
        
        # Lưu kết quả
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # In kết quả
        self._print_results(final_results)
        
        return final_results
    
    def _calculate_overall_score(self, retrieval_results: Dict, generation_results: Dict) -> float:
        """
        Tính điểm tổng thể
        """
        retrieval_score = (retrieval_results['avg_precision_at_k'] + 
                          retrieval_results['avg_recall_at_k'] + 
                          retrieval_results['hit_rate']) / 3
        
        generation_score = (generation_results['avg_bleu'] + 
                           generation_results['avg_rouge1'] + 
                           generation_results['avg_bert_score'] + 
                           generation_results['avg_faithfulness'] + 
                           generation_results['avg_answer_relevancy']) / 5
        
        overall_score = (retrieval_score + generation_score) / 2
        return overall_score
    
    def _print_results(self, results: Dict):
        """
        In kết quả đánh giá
        """
        print("\n" + "="*60)
        print("RAG EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nDataset Info:")
        print(f"  Total Q&A pairs: {results['dataset_info']['total_qa_pairs']}")
        print(f"  Evaluation sample size: {results['dataset_info']['evaluation_sample_size']}")
        print(f"  Wiki files processed: {results['dataset_info']['wiki_files_processed']}")
        
        print(f"\nRetrieval Evaluation:")
        retrieval = results['retrieval_evaluation']
        print(f"  Precision@K: {retrieval['avg_precision_at_k']:.4f}")
        print(f"  Recall@K: {retrieval['avg_recall_at_k']:.4f}")
        print(f"  MRR: {retrieval['avg_mrr']:.4f}")
        print(f"  Hit Rate: {retrieval['hit_rate']:.4f}")
        
        print(f"\nGeneration Evaluation:")
        generation = results['generation_evaluation']
        print(f"  BLEU Score: {generation['avg_bleu']:.4f}")
        print(f"  ROUGE-1: {generation['avg_rouge1']:.4f}")
        print(f"  ROUGE-2: {generation['avg_rouge2']:.4f}")
        print(f"  ROUGE-L: {generation['avg_rougeL']:.4f}")
        print(f"  BERT Score: {generation['avg_bert_score']:.4f}")
        print(f"  Faithfulness: {generation['avg_faithfulness']:.4f}")
        print(f"  Answer Relevancy: {generation['avg_answer_relevancy']:.4f}")
        
        print(f"\nOverall Score: {results['overall_score']:.4f}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG model")
    parser.add_argument("--wiki_data", default="src/data/wiki_data", help="Path to wiki data directory")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--config", default="src/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(args.config)
    results = evaluator.run_full_evaluation(args.wiki_data, args.output)

if __name__ == "__main__":
    main()
