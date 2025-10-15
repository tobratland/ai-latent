"""
Simple Word Embedding Heatmap
Just modify the WORDS list and run!

Requirements: pip install gensim matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api

# ============= MODIFY THIS LIST =============
WORDS = [
    "queen",
    "king", 
    "man",
    "woman",
    "spaghetti",
]
# ==========================================

def main():
    # Load embeddings
    print("Loading model...")
    model = api.load("glove-wiki-gigaword-50")
    
    # Get word vectors
    embeddings = []
    valid_words = []
    
    for word in WORDS:
        try:
            embeddings.append(model[word.lower()])
            valid_words.append(word)
        except KeyError:
            print(f"'{word}' not found - skipping")
    
    embeddings = np.array(embeddings)
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    
    # Plot with red/blue colormap
    plt.imshow(embeddings, cmap='RdBu_r', aspect='auto', 
               vmin=-np.max(np.abs(embeddings)), vmax=np.max(np.abs(embeddings)))
    
    # Labels
    plt.yticks(range(len(valid_words)), valid_words, fontsize=14)
    plt.xlabel('Embedding Dimensions', fontsize=12)
    
    # Clean up
    plt.tight_layout()
    plt.show()
    
    print(f"Showing embeddings for: {valid_words}")

if __name__ == "__main__":
    main()