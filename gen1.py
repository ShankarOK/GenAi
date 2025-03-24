from gensim.models import KeyedVectors

def explore_word_relationships(model_path):
    try:
        print("Loading the model...")

        # Load the Word2Vec model
        word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")
        print("Model loaded successfully!")

        # Define words for analogy (king - man + woman)
        words = ['king', 'man', 'woman']

        # Check if all words exist in the model
        missing_words = [word for word in words if word.lower() not in word_vectors]
        if missing_words:
            print(f"Warning: Words not found in the model: {', '.join(missing_words)}")
            return
        
        # Perform vector arithmetic: (king - man + woman)
        result_vector = word_vectors['king'] - word_vectors['man'] + word_vectors['woman']

        # Fetch top 5 similar words
        similar_words = word_vectors.similar_by_vector(result_vector, topn=5)

        # Display results
        print("\nWords similar to (king - man + woman):")
        for word, score in similar_words:
            print(f"  {word}: {score:.4f}")

    except Exception as e:
        print(f"Error: {e}")

# Path to Word2Vec model (Modify this path accordingly)
model_path = "/home/pc/Downloads/GoogleNews-vectors-negative300.bin"
explore_word_relationships(model_path)
