from gensim.downloader import load
import torch
from transformers import pipeline

# Load pre-trained GloVe embeddings and GPT-2
embedding_model = load("glove-wiki-gigaword-50")
text_generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
torch.manual_seed(42)

# Enrich prompt with similar words
def enrich_prompt(prompt):
    words = prompt.split()
    return " ".join(" ".join(w for w, _ in embedding_model.most_similar(word, topn=3)) for word in words)

# Generate text for original and enriched prompts
def generate_text(prompt):
    return text_generator(prompt, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7)[0]["generated_text"]

original_prompt = "lung cancer"
enriched_prompt = enrich_prompt(original_prompt)

print("Original:", generate_text(original_prompt))
print("\nEnriched:", generate_text(enriched_prompt))
