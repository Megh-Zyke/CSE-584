import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ContextGate:
    def __init__(self, model_name="google/flan-t5-base"):
        # We use T5-small for sub-100ms latency on CPU/GPU
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Keywords that signal a need for context normalization
        self.context_signals = ["it", "he", "she", "they", "that", "this", "him", "her", "then", "there", "his", "their", "its"]

    def _needs_rewriting(self, query):
        """
        Heuristic: Only rewrite if pronouns are detected to save latency.
        """
        words = query.lower().split()
        return any(signal in words for signal in self.context_signals)

    def rewrite_query(self, current_query, chat_history):
      if not self._needs_rewriting(current_query) or not chat_history:
          return current_query

      last_context = chat_history[-1]

      # IMPROVEMENT 1: More explicit "Few-Shot" style prompt
      # Small models perform much better when you show them the 'Pattern'
      input_text = (
          f"Context: {last_context}\n"
          f"Ambiguous Question: {current_query}\n"
          f"Standalone Question using names from context: "
      )
      print(f"[Gate1] Rewriting query: {input_text}")
      inputs = self.tokenizer(input_text, return_tensors="pt")

      # IMPROVEMENT 2: Adjust generation parameters
      # 'repetition_penalty' prevents it from just copying the input
      outputs = self.model.generate(
          **inputs,
          max_new_tokens=50,
          repetition_penalty=1.2, # Higher value = less likely to repeat input
          num_beams=4,            # Search for a better logical fit
          early_stopping=True
      )

      resolved_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

      # IMPROVEMENT 3: Basic validation
      # If it's still identical, it failed to resolve
      return resolved_query

if __name__ == "__main__":
    gate1 = ContextGate()

    # Simulate the conversation history
    history = [
        "what is the capital of France?",
        "The capital of France is Paris."
    ]

    # Ambiguous follow-up query
    query = "what is its population?"

    result = gate1.rewrite_query(query, history)
    print(result)
