from pinecone import Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
from embedder.huggingface import HuggingFaceEmbedder

load_dotenv()

class PineconeIndex:
    def __init__(self, index_name, model_name, dimension):
        self.api_key = os.getenv("PINECONE_API_KEY")

        if not all([self.api_key]):
            raise ValueError("Please set PINECONE_API_KEY in your .env file.")

        self.pinecone = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.embedding_model = HuggingFaceEmbedder(model_name)
        self.dimension = dimension
        self.create_index()

    def create_index(self):
        if not self.pinecone.has_index(self.index_name):
            self.pinecone.create_index(
                name=self.index_name,
                vector_type="dense",
                dimension=self.dimension,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="disabled",
                tags={
                    "environment": "development"
                }
            )
        else:
            print(f"Index {self.index_name} already exists.")
        self.index = self.pinecone.Index(self.index_name)

    def preprocess(self, texts):
        return [text for text in texts if len(text) > 5]

    def generate_embeddings(self, texts, batch_size=8):
        texts = self.preprocess(texts)
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(self.embedding_model.encode(batch))
        return embeddings
    
    def upsert_texts(self, texts, file_source):
        texts = self.preprocess(texts)
        embeddings = self.generate_embeddings(texts)
        ids = [f"_{i}" for i in range(len(texts))]
        for i in range(0, len(texts), 100):
            batch_ids = ids[i:i + 100]
            batch_embeddings = embeddings[i:i + 100]
            self.index.upsert(
                vectors=[
                    {
                        "id": id,
                        "values": embedding,
                        "metadata": {
                            "text": text,
                            "file_source": str(file_source),
                        }
                    }
                    for id, embedding, text in zip(batch_ids, batch_embeddings, texts[i:i + 100])
                ],
            )
        
    def search(self, query, top_k=10):
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )
        return results