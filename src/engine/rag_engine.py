from generator.prompt import PROMPT_TEMPLATE

class RAGEngine:
    def __init__(self, indexer, generator):
        self.indexer = indexer
        self.generator = generator

    def generate_answer(self, query, top_k=5):
        # Search for relevant documents
        context = self.indexer.search(query, top_k).matches
        
        # Generate answer using the generator
        prompt = PROMPT_TEMPLATE.format(question=query, context=context)
        answer = self.generator.generate(prompt)
        return answer