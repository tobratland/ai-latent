import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from matplotlib import font_manager
import io
from PIL import Image

def create_attention_plot(sentence, target_word_idx, attention_weights, title):
    """
    Create a beautiful attention visualization
    """
    words = sentence.split()
    n_words = len(words)
    
    # Create figure with clean style
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax.set_xlim(-0.5, n_words + 0.5)
    ax.set_ylim(-1, 2.5)
    ax.axis('off')
    
    # Title
    ax.text(n_words/2, 2.2, title, 
            fontsize=18, fontweight='bold', ha='center',
            color='#2c3e50')
    
    # Normalize attention weights
    attention_weights = np.array(attention_weights)
    attention_weights = attention_weights / attention_weights.max()
    
    # Draw words
    word_positions = []
    for i, word in enumerate(words):
        x = i
        y = 0.5 if i == target_word_idx else 0
        
        # Determine color
        if i == target_word_idx:
            color = '#e74c3c'  # Red for target word
            text_color = 'white'
        else:
            color = '#3498db'  # Blue for context words
            text_color = 'white'
        
        # Draw box
        box = FancyBboxPatch(
            (x - 0.3, y - 0.15), 0.6, 0.3,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='none',
            alpha=0.9,
            zorder=10
        )
        ax.add_patch(box)
        
        # Draw text
        ax.text(x, y, word, 
                fontsize=14, fontweight='bold',
                ha='center', va='center',
                color=text_color,
                zorder=11)
        
        word_positions.append((x, y))
    
    # Draw attention arrows
    target_x, target_y = word_positions[target_word_idx]
    
    for i, (x, y) in enumerate(word_positions):
        if i != target_word_idx:
            attention = attention_weights[i]
            
            # Draw arrow with varying thickness and opacity
            arrow = FancyArrowPatch(
                (target_x, target_y - 0.15),
                (x, y + 0.15),
                arrowstyle='->,head_width=0.4,head_length=0.4',
                color='#9b59b6',
                linewidth=1 + attention * 4,
                alpha=0.3 + attention * 0.6,
                zorder=5,
                connectionstyle="arc3,rad=0.2"
            )
            ax.add_patch(arrow)
            
            # Add attention score label
            mid_x = (target_x + x) / 2
            mid_y = (target_y + y) / 2 + 0.3
            ax.text(mid_x, mid_y, f'{attention:.2f}',
                   fontsize=10,
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor='#9b59b6',
                            alpha=0.8),
                   zorder=12)
    
    # Add legend
    ax.text(n_words/2, -0.7, 
           f'Attention from "{words[target_word_idx]}" to context words',
           fontsize=12, ha='center', style='italic',
           color='#7f8c8d')
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def visualize_bank_attention():
    """Visualize bank in different contexts"""
    
    # Sentence 1: River bank
    sentence1 = "I sat by the river bank"
    target_idx1 = 5  # "bank"
    # Simulated attention weights (in reality these come from the model)
    # High attention to "river" and "sat", low to others
    attention1 = [0.15, 0.25, 0.10, 0.15, 0.95, 0.0]  # river gets highest
    
    # Sentence 2: Financial bank
    sentence2 = "I deposited money at the bank"
    target_idx2 = 5  # "bank"
    # High attention to "deposited" and "money"
    attention2 = [0.10, 0.95, 0.85, 0.12, 0.15, 0.0]  # deposited and money get highest
    
    img1 = create_attention_plot(sentence1, target_idx1, attention1, 
                                  'Context 1: "bank" = edge of river')
    img2 = create_attention_plot(sentence2, target_idx2, attention2,
                                  'Context 2: "bank" = financial institution')
    
    return img1, img2

def visualize_plane_attention():
    """Visualize plane in different contexts"""
    
    # Sentence 1: Airplane
    sentence1 = "The plane flew across the sky"
    target_idx1 = 1  # "plane"
    # High attention to "flew" and "sky"
    attention1 = [0.12, 0.0, 0.95, 0.65, 0.15, 0.88]  # 6 words total
    
    # Sentence 2: Plane of existence
    sentence2 = "meditation opens a different plane of existence"
    target_idx2 = 4  # "plane" (corrected index!)
    # High attention to "different", "existence", "meditation"
    # meditation, opens, a, different, plane, of, existence = 7 words
    attention2 = [0.75, 0.25, 0.15, 0.82, 0.0, 0.45, 0.95]  # 7 values now!
    
    img1 = create_attention_plot(sentence1, target_idx1, attention1,
                                  'Context 1: "plane" = aircraft')
    img2 = create_attention_plot(sentence2, target_idx2, attention2,
                                  'Context 2: "plane" = dimension/realm')
    
    return img1, img2


# Create Gradio interface
with gr.Blocks(title="Attention Mechanism Visualization", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 Transformer Attention Mechanism")
    gr.Markdown("""
    ### How Transformers Understand Context
    
    Unlike Word2Vec (which gives the same embedding for a word regardless of context), 
    transformers use **attention** to compute different embeddings based on surrounding words.
    
    The visualization shows:
    - 🔴 **Red box**: The target word being embedded
    - 🔵 **Blue boxes**: Context words it can attend to
    - 🟣 **Purple arrows**: Attention weights (thicker = stronger attention)
    """)
    
    with gr.Tab("Example 1: Bank"):
        gr.Markdown("### The word 'bank' has different meanings based on context")
        
        bank_btn = gr.Button("🎨 Generate Visualization", variant="primary")
        
        with gr.Row():
            bank_img1 = gr.Image(label="River Bank", type="pil")
            bank_img2 = gr.Image(label="Financial Bank", type="pil")
        
        gr.Markdown("""
        **Key Insight**: 
        - In context 1, "bank" attends strongly to "**river**" → learns it means *riverbank*
        - In context 2, "bank" attends strongly to "**deposited**" and "**money**" → learns it means *financial institution*
        
        Same word, different embeddings! 🎯
        """)
        
        bank_btn.click(
            fn=visualize_bank_attention,
            inputs=[],
            outputs=[bank_img1, bank_img2]
        )
    
    with gr.Tab("Example 2: Plane"):
        gr.Markdown("### The word 'plane' can mean aircraft OR dimension")
        
        plane_btn = gr.Button("🎨 Generate Visualization", variant="primary")
        
        with gr.Row():
            plane_img1 = gr.Image(label="Airplane", type="pil")
            plane_img2 = gr.Image(label="Plane of Existence", type="pil")
        
        gr.Markdown("""
        **Key Insight**: 
        - In context 1, "plane" attends to "**flew**" and "**sky**" → learns it's an *aircraft*
        - In context 2, "plane" attends to "**different**" and "**existence**" → learns it's a *dimension/realm*
        
        The attention mechanism lets each word "look around" and adjust its meaning! 🧠
        """)
        
        plane_btn.click(
            fn=visualize_plane_attention,
            inputs=[],
            outputs=[plane_img1, plane_img2]
        )
    
    gr.Markdown("""
    ---
    ### 💡 The Breakthrough
    
    **Word2Vec**: `bank → [0.23, -0.45, 0.67, ...]` (always the same)
    
    **Transformers**: `bank → [0.23, -0.45, ...]` OR `[0.89, 0.12, ...]` depending on context!
    
    This is why transformers revolutionized NLP - they create **dynamic, context-aware embeddings**! 🚀
    """)

if __name__ == "__main__":
    demo.launch(share=False)