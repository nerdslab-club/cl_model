import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# Sample word embeddings
word_vectors = {
    "cat": np.array([0.2, 0.8]),
    "dog": np.array([0.5, 0.7]),
    "apple": np.array([0.8, 0.3]),
    "banana": np.array([0.7, 0.4]),
    "elephant": np.array([0.1, 0.9]),
}

# Convert to array
words = list(word_vectors.keys())
vectors = np.array([word_vectors[word] for word in words])

# Visualize word embeddings using t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Decrease perplexity
word_embeddings_2d = tsne.fit_transform(vectors)

plt.figure(figsize=(8, 6))
plt.scatter(word_embeddings_2d[:, 0], word_embeddings_2d[:, 1], c='blue')
for i, word in enumerate(words):
    plt.annotate(word, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))
plt.title('Word Embeddings Visualization (t-SNE)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()

# Sample sentence embeddings
sentences = [
    "I like cats and dogs",
    "Apples are delicious fruits",
    "Bananas are yellow",
    "Elephants are large animals",
    "Cats and dogs are pets"
]

# Convert sentences to sentence embeddings (average of word vectors)
def sentence_embedding(sentence):
    words = sentence.split()
    embedding = np.mean([word_vectors[word] for word in words if word in word_vectors], axis=0)
    return embedding

sentence_embeddings = np.array([sentence_embedding(sentence) for sentence in sentences])

# Visualize sentence embeddings using PCA
pca = PCA(n_components=2)
sentence_embeddings_2d = pca.fit_transform(sentence_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(sentence_embeddings_2d[:, 0], sentence_embeddings_2d[:, 1], c='red')
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (sentence_embeddings_2d[i, 0], sentence_embeddings_2d[i, 1]))
plt.title('Sentence Embeddings Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
