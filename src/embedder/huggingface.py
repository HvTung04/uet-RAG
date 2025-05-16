from sentence_transformers import SentenceTransformer
import torch

class HuggingFaceEmbedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    
    def encode(self, texts):
        """
        Encode a list of texts into embeddings.

        :param texts: List of texts to encode.
        :return: List of embeddings.
        """
        return self.model.encode(texts)