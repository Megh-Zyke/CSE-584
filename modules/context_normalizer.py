from ollama import AsyncClient
from modules.query import Message
from typing import List
import time 

class ContextNormalizer:
    def __init__(self, model_name = "llama3.2"):
        self.model_name= model_name
        self.client = AsyncClient(host = 'http://localhost:11434')
    
    async def normalize(self, query: str, history = List['Message']) -> str:
            if not history:
                return query

            context_lines = [f"{m.query}: {m.response}" for m in history]
            context_str = "\n".join(context_lines)
            prompt = (
                "You are a query rewriting system.\n\n"

                "Your task is to rewrite the current query into a clear, standalone question "
                "by resolving ambiguous references (e.g., 'it', 'they', 'this') using ONLY the provided conversation history.\n\n"

                "STRICT RULES:\n"
                "1. Preserve the original meaning exactly. Do NOT add, remove, or infer new information.\n"
                "2. Only rewrite if the history clearly resolves ambiguity.\n"
                "3. If the history is irrelevant, insufficient, or ambiguous, return the original query EXACTLY.\n"
                "4. Do NOT answer the question.\n"
                "5. Do NOT explain your reasoning.\n"
                "6. Output ONLY the final query text.\n\n"

                "History:\n"
                f"{context_str}\n\n"

                "Current Query:\n"
                f"{query}\n\n"

                "Final Query:"
            )

            try:
                start = time.perf_counter()
                response = await self.client.generate ( 
                    model = self.model_name, 
                    prompt = prompt, 
                    keep_alive = -1,                  
                )
                end = time.perf_counter()
                print(f"ContextNormalizer latency: {(end - start) * 1000} ms")
                return response.response.strip()
                
            except Exception as e:
                print(f"ContextNormalizer error: {e}")
                return query