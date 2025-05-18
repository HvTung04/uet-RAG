from groq import Groq
from prompt import RAG_SYSTEM

class GroqModel:
    def __init__(self, model_name, system_prompt=RAG_SYSTEM):
        self.client = Groq()
        self.model_name = model_name
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.history = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.max_history_length = 5

    def generate(self, query):
        if len(self.history) > self.max_history_length:
            self.history = self.history[0] + self.history[-self.max_history_length:]
        self.history.append({"role": "user", "content": query})
        response = self.client.chat.completions.create(
            messages=self.history,
            model=self.model_name,
        ).choices[0].message.content

        self.history.append({"role": "assistant", "content": response})
        return response
    
    def reset(self):
        self.history = [
            {"role": "system", "content": self.system_prompt},
        ]