import yaml
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from indexer.pinecone import PineconeIndex

parser = argparse.ArgumentParser()
parser.add_argument("--upsert", action="store_true", help="Upsert wiki pages to Pinecone")
parser.add_argument("--query", required=False, type=str, help="Query to search in Pinecone")
args = parser.parse_args()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

pinecone_config = config.get("pinecone")

def find_json_files(folder_path):
    folder = Path(folder_path)
    return list(folder.glob("*.json"))

if __name__ == "__main__":
    indexer = PineconeIndex(
        index_name=pinecone_config["index_name"],
        model_name=pinecone_config["model_name"],
        dimension=pinecone_config["dimension"]
    )

    # Upserting all wiki pages
    if args.upsert:
        print("Upserting all wiki pages...")
        json_files = find_json_files(config["wiki_data"])
        for file in tqdm(json_files, desc="Processing JSON files", unit="file"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = data["raw_content"]["content"]
            indexer.upsert_texts(texts, file_source=file)

    if args.query:
        print("Searching for query...")
        results = indexer.search(args.query, top_k=5)
        print(results)
        print(type(results))