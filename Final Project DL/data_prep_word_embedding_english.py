# =============================================================================
#  WORD EMBEDDING IN NLP — English Only
#  TASKS 2.1 & 2.2: Data Preparation + Word Embedding (English)
#
#  Trains a CBOW word embedding model on the ENGLISH sentences only from
#  the English-Bengali parallel corpus.  All query words, nearest neighbours,
#  and PCA plots are English only.
#
#  DATA:    ben-eng/ben.txt   (column 0 = English sentences only)
#
#  HOW TO RUN:
#    python data_prep_word_embedding_english.py
#
#  OUTPUT (saved to same folder as this script):
#    eng_pca_all_words.png         — PCA scatter of entire English vocabulary
#    eng_pca_top_words.png         — PCA scatter of top 50 English words
#    eng_nearest_neighbours.png    — nearest-neighbour bar charts (English)
# =============================================================================

import os
import re
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models             import Sequential
from tensorflow.keras.layers             import Dense, Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils              import to_categorical

from sklearn.decomposition    import PCA
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
#  FILE PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(BASE_DIR, "ben-eng", "ben.txt")

OUT_PCA_ALL = os.path.join(BASE_DIR, "eng_pca_all_words.png")
OUT_PCA_TOP = os.path.join(BASE_DIR, "eng_pca_top_words.png")
OUT_NEAREST = os.path.join(BASE_DIR, "eng_nearest_neighbours.png")


# =============================================================================
#  STEPS 1 & 2 — Load English column only, preprocess
# =============================================================================

def load_english(tsv_path):
    """
    Read column 0 (English) from the TSV only.
    Column 1 (Bengali) and column 2 (CC-BY) are ignored.
    Returns a list of raw English sentence strings.
    """
    sentences = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 1:
                continue
            text = parts[0].strip()
            if text:
                sentences.append(text)
    return sentences


def preprocess_english(raw_sentences):
    """Lowercase, keep only a-z letters and whitespace."""
    cleaned = []
    for sentence in raw_sentences:
        sentence = re.sub(r'[^a-zA-Z\s]', ' ', sentence)
        sentence = sentence.lower()
        tokens   = sentence.split()
        if tokens:
            cleaned.append(' '.join(tokens))
    return cleaned


# =============================================================================
#  STEPS 3 & 4 — Tokenize, CBOW pairs, train
# =============================================================================

EMBEDDING_SIZE = 10
WINDOW_SIZE    = 2
EPOCHS         = 100
BATCH_SIZE     = 256


def build_and_train(sentences):
    tokenizer  = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences  = tokenizer.texts_to_sequences(sentences)
    vocab_size = len(tokenizer.word_index) + 1

    contexts, targets = [], []
    for seq in sequences:
        for i in range(WINDOW_SIZE, len(seq) - WINDOW_SIZE):
            context = seq[i - WINDOW_SIZE:i] + seq[i + 1:i + WINDOW_SIZE + 1]
            contexts.append(context)
            targets.append(seq[i])

    if not contexts:
        raise ValueError("No context-target pairs generated.")

    X = np.array(contexts)
    y = to_categorical(targets, num_classes=vocab_size)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE,
                  input_length=2 * WINDOW_SIZE),
        Lambda(lambda x: tf.reduce_mean(x, axis=1)),
        Dense(units=vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"  Training CBOW model on English ({EPOCHS} epochs) ...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    print("  Training complete.")
    return model, tokenizer


# =============================================================================
#  WORD EMBEDDING HELPERS
# =============================================================================

def get_embeddings(model, tokenizer):
    embeddings = model.layers[0].get_weights()[0]
    words      = list(tokenizer.word_index.keys())
    return embeddings, words


def cosine_nearest(query_word, embeddings, tokenizer, top_n=6):
    """
    Return the top_n nearest English words to query_word by cosine similarity.
    Returns list of (word, score) or None if word not in vocab.
    """
    if query_word not in tokenizer.word_index:
        return None

    query_idx = tokenizer.word_index[query_word]
    query_vec = embeddings[query_idx].reshape(1, -1)
    all_sims  = cosine_similarity(query_vec, embeddings)[0]

    word_sims = [
        (tokenizer.index_word[i], float(all_sims[i]))
        for i in range(1, len(tokenizer.word_index) + 1)
        if i != query_idx
    ]
    word_sims.sort(key=lambda x: -x[1])
    return word_sims[:top_n]


# =============================================================================
#  PLOT A — PCA scatter of ALL English vocabulary words
# =============================================================================

def plot_pca_all_words(embeddings, tokenizer, save_path):
    print("  Running PCA on all English vocabulary words...")

    pca    = PCA(n_components=2)
    result = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(18, 14))

    for word, idx in tokenizer.word_index.items():
        ax.scatter(result[idx, 0], result[idx, 1],
                   s=10, alpha=0.45, color='#4e79a7')
        ax.annotate(word,
                    xy=(result[idx, 0], result[idx, 1]),
                    xytext=(4, 2), textcoords='offset points',
                    ha='right', va='bottom',
                    fontsize=4.5, alpha=0.65, color='#1a1209')

    ax.set_title("Word Embeddings — PCA (All English Words)\n"
                 "English-Bengali Parallel Corpus  ·  English CBOW Model",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("PCA Component 1", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  PLOT B — PCA scatter of TOP 50 most frequent English words
# =============================================================================

def plot_pca_top_words(embeddings, tokenizer, save_path, top_n=50):
    print(f"  Running PCA on top {top_n} most frequent English words...")

    top_words   = [tokenizer.index_word[i] for i in range(1, top_n + 1)
                   if i in tokenizer.index_word]
    top_vectors = np.array([embeddings[tokenizer.word_index[w]] for w in top_words])

    pca    = PCA(n_components=2)
    result = pca.fit_transform(top_vectors)

    # Colour groups for English words
    PRONOUNS = {'i', 'you', 'he', 'she', 'we', 'they', 'it', 'me',
                'him', 'her', 'us', 'them', 'my', 'your', 'his'}
    VERBS    = {'is', 'are', 'was', 'do', 'did', 'have', 'has', 'go',
                'went', 'want', 'know', 'like', 'said', 'get', 'can'}

    def word_color(w):
        if w in PRONOUNS: return '#e15759'   # red   — pronouns
        if w in VERBS:    return '#59a14f'   # green — verbs
        return '#4e79a7'                      # blue  — other

    fig, ax = plt.subplots(figsize=(13, 10))

    for i, word in enumerate(top_words):
        color = word_color(word)
        ax.scatter(result[i, 0], result[i, 1], s=55, color=color,
                   zorder=3, edgecolors='white', linewidth=0.5)
        ax.annotate(word, xy=(result[i, 0], result[i, 1]),
                    xytext=(5, 2), textcoords='offset points',
                    ha='right', va='bottom',
                    fontsize=9, fontweight='bold', color='#1a1209')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#e15759', label='Pronouns'),
        Patch(facecolor='#59a14f', label='Verbs'),
        Patch(facecolor='#4e79a7', label='Other'),
    ], fontsize=9, loc='upper right')

    ax.set_title(f"Word Embeddings — PCA (Top {top_n} English Words)\n"
                 "English-Bengali Parallel Corpus  ·  English CBOW Model",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("PCA Component 1", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  PLOT C — Nearest-neighbour bar charts (English only)
# =============================================================================

def plot_nearest_neighbours(embeddings, tokenizer, query_words, save_path, top_n=6):
    print("  Computing English nearest neighbours...")

    valid  = [w for w in query_words if w in tokenizer.word_index]
    n      = len(valid)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()

    ENG_COLOR = '#4e79a7'

    for ax_idx, word in enumerate(valid):
        nbrs = cosine_nearest(word, embeddings, tokenizer, top_n=top_n)
        ax   = axes[ax_idx]

        if nbrs is None:
            ax.set_title(f'"{word}" — not in vocab')
            ax.axis('off')
            continue

        nbr_words  = [p[0] for p in nbrs]
        nbr_scores = [p[1] for p in nbrs]

        ax.barh(nbr_words[::-1], nbr_scores[::-1],
                color=ENG_COLOR, alpha=0.85, edgecolor='white')

        for i, sc in enumerate(nbr_scores[::-1]):
            ax.text(max(sc - 0.03, 0.02), i, f'{sc:.3f}',
                    va='center', ha='right', fontsize=7,
                    color='white', fontweight='bold')

        ax.set_title(f'"{word}"  →  Nearest English words',
                     fontsize=10, fontweight='bold', color=ENG_COLOR)
        ax.set_xlabel('Cosine Similarity', fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    for ax_idx in range(len(valid), len(axes)):
        axes[ax_idx].axis('off')

    fig.suptitle(
        "Nearest Neighbours by Cosine Similarity  —  English Model\n"
        "English-Bengali Parallel Corpus  ·  English CBOW",
        fontsize=13, fontweight='bold', color='#1a1209'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':

    QUERY_WORDS = [
        'you',   'can',   'the',   'is',
        'do',    'he',    'what',  'want',
        'have',  'know',  'go',    'like',
    ]

    print("=" * 65)
    print("TASKS 2.1 & 2.2 — ENGLISH DATA PREP + WORD EMBEDDING")
    print("  Training on English sentences only")
    print("=" * 65)

    # ── Load & preprocess English only ──────────────────────────────────────
    print(f"\n[1] Loading English sentences from: {TSV_PATH}")
    raw_sentences = load_english(TSV_PATH)
    sentences     = preprocess_english(raw_sentences)
    print(f"      {len(sentences):,} English sentences loaded")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[2] Building and training English CBOW model (~1-2 minutes)...")
    model, tokenizer = build_and_train(sentences)
    vocab_size = len(tokenizer.word_index)
    print(f"      English vocabulary size: {vocab_size:,} words")

    # ── Extract embeddings ───────────────────────────────────────────────────
    print("\n[3] Extracting embedding matrix...")
    embeddings, words = get_embeddings(model, tokenizer)
    print(f"      Shape: {embeddings.shape}  "
          f"({embeddings.shape[0]} words x {embeddings.shape[1]} dims)")

    # ── Console nearest-neighbour report ─────────────────────────────────────
    print("\n[4] Nearest English neighbours (cosine similarity):")
    print(f"      {'Query':<10}  Nearest words")
    print(f"      {'-'*10}  {'-'*55}")
    for word in QUERY_WORDS:
        nbrs = cosine_nearest(word, embeddings, tokenizer, top_n=5)
        if nbrs is None:
            print(f"      {word:<10}  (not in vocabulary)")
        else:
            result_str = ',  '.join(f"{w}({s:.3f})" for w, s in nbrs)
            print(f"      {word:<10}  {result_str}")

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\n[5] Generating English visualizations...")

    print("\n  Plot A — PCA all English words:")
    plot_pca_all_words(embeddings, tokenizer, OUT_PCA_ALL)

    print("\n  Plot B — PCA top 50 English words:")
    plot_pca_top_words(embeddings, tokenizer, OUT_PCA_TOP, top_n=50)

    print("\n  Plot C — English nearest neighbours:")
    plot_nearest_neighbours(embeddings, tokenizer, QUERY_WORDS, OUT_NEAREST, top_n=6)

    print("\n" + "=" * 65)
    print("English model complete. Files saved:")
    print(f"  1. eng_pca_all_words.png      — PCA of all {vocab_size:,} English words")
    print(f"  2. eng_pca_top_words.png      — PCA of top 50 English words")
    print(f"  3. eng_nearest_neighbours.png — nearest-neighbour charts")
    print("=" * 65)
