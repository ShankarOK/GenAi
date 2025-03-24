from gensim.downloader import load
import random
import nltk

nltk.download('punkt')


glove_model = load("glove-wiki-gigaword-50")

def create_paragraph(topic_word, similar_words):
    random.shuffle(similar_words)
    return f"{topic_word.capitalize()} relates to {', '.join(similar_words)}."

topic_word = "hacking"
similar_words = [word for word, _ in glove_model.most_similar(topic_word, topn=5)]
paragraph = create_paragraph(topic_word, similar_words)

print(paragraph)
