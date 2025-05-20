from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def rechunking(texts):
    document = " ".join(texts)
    chunks = splitter.split_text(document)
    return chunks