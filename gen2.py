import matplotlib
matplotlib.use('TkAgg')  # Switch backend to TkAgg

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


words = ['football', 'basketball', 'cricket', 'technology', 'computer', 'robot', 'AI', 'cloud', 'python', 'data']
np.random.seed(42) 
word_vectors = {word: np.random.rand(100) for word in words} 

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(np.array(list(word_vectors.values())))

# Plot the results
plt.figure(figsize=(8, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1])

# Annotate points
for i, word in enumerate(words):
    plt.annotate(word, (pca_result[i, 0], pca_result[i, 1]))

plt.title('Word Embedding Visualization with PCA')
plt.show()
