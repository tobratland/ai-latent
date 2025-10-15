"""
Simple Word Analogy: base - remove + add ≈ compare_result_to
Shows vector arithmetic with clear visualization.

Requirements: pip install gensim matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api


base = "king" # king, good
remove = "queen" # queen, better
add = "woman" # woman, bad
compare_result_to = "man" # man, worse


def load_model():
    print("Loading word embeddings...")
    model = api.load("glove-wiki-gigaword-50")
    print("Model loaded!")
    return model

def solve_analogy(model, base_word, remove_word, add_word):
    """Compute: base - remove + add"""
    try:
        base_vec = model[base_word.lower()]
        remove_vec = model[remove_word.lower()]  
        add_vec = model[add_word.lower()]
        
        # Calculate: base - remove + add
        result_vector = base_vec - remove_vec + add_vec
        
        # Find most similar words to result
        exclude_words = {base_word.lower(), remove_word.lower(), add_word.lower()}
        
        similarities = []
        for word in model.key_to_index:
            if word not in exclude_words:
                similarity = np.dot(result_vector, model[word]) / (
                    np.linalg.norm(result_vector) * np.linalg.norm(model[word])
                )
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5], result_vector, [base_vec, remove_vec, add_vec]
        
    except KeyError as e:
        print(f"Word not found: {e}")
        return None, None, None

def main():
    model = load_model()
    
    print(f"\nComputing: {base} - {remove} + {add}")
    print(f"Expected result: {compare_result_to}")
    
    # Solve the analogy
    top_matches, result_vector, input_vectors = solve_analogy(model, base, remove, add)
    
    if not top_matches:
        print("Error: Could not compute analogy")
        return
    
    # Get comparison word embedding
    try:
        compare_vector = model[compare_result_to.lower()]
        has_comparison = True
    except KeyError:
        print(f"Warning: '{compare_result_to}' not found in vocabulary")
        has_comparison = False
    
    # Prepare vectors for visualization
    words = [
        base,
        remove, 
        add,
        "COMPUTED RESULT",
    ]
    vectors = input_vectors + [result_vector]
    
    # Add comparison if available
    if has_comparison:
        # Calculate similarity
        similarity = np.dot(result_vector, compare_vector) / (
            np.linalg.norm(result_vector) * np.linalg.norm(compare_vector)
        )
        words.append(f"ACTUAL: {compare_result_to} ({similarity:.3f})")
        vectors.append(compare_vector)
        
        print(f"Similarity to '{compare_result_to}': {similarity:.4f}")
    
    # Add top 5 matches
    for i, (word, sim) in enumerate(top_matches):
        try:
            match_vector = model[word]
            words.append(f"{i+1}. {word} ({sim:.3f})")
            vectors.append(match_vector)
        except KeyError:
            continue
    
    embeddings = np.array(vectors)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Create heatmap
    vmax = np.max(np.abs(embeddings))
    plt.imshow(embeddings, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Labels
    plt.yticks(range(len(words)), words, fontsize=12)
    plt.xlabel('Embedding Dimensions (50D)', fontsize=12)
    
    # Title
    title = f'Vector Arithmetic: "{base}" - "{remove}" + "{add}"'
    if has_comparison:
        title += f' ≈ "{compare_result_to}"'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add separator line before top matches
    separator_idx = 4 + (1 if has_comparison else 0)
    plt.axhline(y=separator_idx-0.5, color='black', linewidth=2, alpha=0.7)
    separator_idx = 3
    plt.axhline(y=separator_idx-0.5, color='black', linewidth=2, alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nTop 5 closest words to computed result:")
    for i, (word, sim) in enumerate(top_matches):
        marker = " ✓" if word.lower() == compare_result_to.lower() else ""
        print(f"  {i+1}. {word}: {sim:.4f}{marker}")
    
    if has_comparison:
        # Check where expected answer ranks
        expected_rank = None
        for rank, (word, sim) in enumerate(top_matches):
            if word.lower() == compare_result_to.lower():
                expected_rank = rank + 1
                break
        
        if expected_rank:
            print(f"\n✓ Expected answer '{compare_result_to}' ranked #{expected_rank}")
        else:
            print(f"\n✗ Expected answer '{compare_result_to}' not in top 5")

if __name__ == "__main__":
    main()