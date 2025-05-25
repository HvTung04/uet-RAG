import yaml
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from indexer.pinecone import PineconeIndex
from indexer.utils import rechunking
from engine.rag_engine import RAGEngine
from generator.groq_model import GroqModel

parser = argparse.ArgumentParser()
parser.add_argument("--upsert", action="store_true", help="Upsert wiki pages to Pinecone")
parser.add_argument("--query", required=False, type=str, help="Query to search in Pinecone")
parser.add_argument("--evaluate", action="store_true", help="Run RAG evaluation")
args = parser.parse_args()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

pinecone_config = config.get("pinecone")
generator_config = config.get("generator")

def find_json_files(folder_path):
    folder = Path(folder_path)
    return list(folder.glob("*.json"))

if __name__ == "__main__":
    indexer = PineconeIndex(
        index_name=pinecone_config["index_name"],
        model_name=pinecone_config["model_name"],
        dimension=pinecone_config["dimension"],
    )
    generator = GroqModel(
        model_name=generator_config["model_name"],
    )
    engine = RAGEngine(indexer=indexer, generator=generator)

    # Upserting all wiki pages
    if args.upsert:
        print("Upserting all wiki pages...")
        json_files = find_json_files(config["wiki_data"])
        for file in tqdm(json_files, desc="Processing JSON files", unit="file"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = rechunking(data["raw_content"]["content"])
            indexer.upsert_texts(texts, file_source=file)

    if args.query:
        print("Searching for query...")
        response = engine.generate_answer(args.query)
        print(f"Query: {args.query}")
        print(f"Response: {response}")
    
    # Run evaluation
    if args.evaluate:
        print("Running RAG evaluation...")
        try:
            from eval import RAGEvaluator
            evaluator = RAGEvaluator("config.yaml")
            results = evaluator.run_full_evaluation(
                wiki_data_path=config["wiki_data"],
                output_path="evaluation_results.json"
            )
            print("Evaluation completed successfully!")
        except ImportError:
            print("Error: Evaluation dependencies not installed.")
            print("Please install: pip install -r eval_requirements.txt")
        except Exception as e:
            print(f"Error during evaluation: {e}")