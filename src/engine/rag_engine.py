from generator.prompt import PROMPT_TEMPLATE

class RAGEngine:
    def __init__(self, indexer, generator):
        self.indexer = indexer
        self.generator = generator

    def generate_answer(self, query, top_k=5):
        # Search for relevant documents
        search_results = self.indexer.search(query, top_k)
        
        # Extract text from search results
        context_texts = []
        if hasattr(search_results, 'matches') and search_results.matches:
            for match in search_results.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    text = match.metadata.get('text', '')
                    if text:
                        context_texts.append(str(text))
        
        # Join context texts
        context = "\n".join(context_texts) if context_texts else "No relevant context found."
        
        # Generate answer using the generator
        prompt = PROMPT_TEMPLATE.format(question=query, context=context)
        answer = self.generator.generate(prompt)
        return answer