# =============================================================================
#  WORD EMBEDDING IN NLP — Bengali Only
#  TASKS 2.1 & 2.2: Data Preparation + Word Embedding (Bengali)
#
#  Trains a CBOW word embedding model on the BENGALI sentences only from
#  the English-Bengali parallel corpus.  All query words, nearest neighbours,
#  and PCA plots are Bengali only.
#
#  BENGALI FONT SETUP (required to display Bengali characters in plots):
#    macOS:   brew install font-noto-sans-bengali
#    Linux:   sudo apt install fonts-noto-cjk
#    Manual:  download NotoSansBengali-Regular.ttf from fonts.google.com
#             and place it in the same folder as this script.
#
#  DATA:    ben-eng/ben.txt   (column 1 = Bengali sentences only)
#
#  HOW TO RUN:
#    python data_prep_word_embedding_bengali.py
#
#  OUTPUT (saved to same folder as this script):
#    word_based_ben_pca_all_words.png         — PCA scatter of entire Bengali vocabulary
#    word_based_ben_pca_top_words.png         — PCA scatter of top 50 Bengali words
#    word_based_ben_nearest_neighbours.png    — nearest-neighbour bar charts (Bengali)
# =============================================================================

import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import tensorflow as tf
from tensorflow.keras.models             import Sequential
from tensorflow.keras.layers             import Dense, Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils              import to_categorical

from sklearn.decomposition    import PCA
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
#  BENGALI FONT SETUP
# =============================================================================

def find_bengali_font():
    search_names = [
        'NotoSansBengali', 'NotoSerifBengali',
        'Nirmala', 'Vrinda', 'Lohit-Bengali', 'MuktiBangla', 'SolaimanLipi',
        'Kalpurush', 'AdorshoLipi',
    ]
    search_dirs = [
        '/Library/Fonts',
        os.path.expanduser('~/Library/Fonts'),
        '/System/Library/Fonts',
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        os.path.expanduser('~/.fonts'),
        os.path.expanduser('~/.local/share/fonts'),
        'C:\\Windows\\Fonts',
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.endswith(('.ttf', '.ttc', '.otf')):
                    for name in search_names:
                        if name.lower() in fname.lower():
                            return os.path.join(root, fname)
    for name in search_names:
        try:
            path = fm.findfont(fm.FontProperties(family=name),
                               fallback_to_default=False)
            if path:
                return path
        except Exception:
            pass
    return None


def setup_bengali_font():
    font_path = find_bengali_font()
    if font_path:
        fm.fontManager.addfont(font_path)
        prop      = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        matplotlib.rcParams['font.family']     = font_name
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        print(f"  Bengali font loaded: {font_name}  ({font_path})")
        return font_name
    else:
        print("  WARNING: No Bengali font found — text will appear as squares.")
        print("  Fix: place NotoSansBengali-Regular.ttf in this folder and re-run.")
        return None


# =============================================================================
#  FILE PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(BASE_DIR, "ben-eng", "ben.txt")

OUT_PCA_ALL = os.path.join(BASE_DIR, "word_based_ben_pca_all_words.png")
OUT_PCA_TOP = os.path.join(BASE_DIR, "word_based_ben_pca_top_words.png")
OUT_NEAREST = os.path.join(BASE_DIR, "word_based_ben_nearest_neighbours.png")


# =============================================================================
#  STEPS 1 & 2 — Load Bengali column only, preprocess
# =============================================================================

def load_bengali(tsv_path):
    """
    Read column 1 (Bengali) from the TSV only.
    Column 0 (English) and column 2 (CC-BY) are ignored.
    Returns a list of raw Bengali sentence strings.
    """
    sentences = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            text = parts[1].strip()
            if text:
                sentences.append(text)
    return sentences


def preprocess_bengali(raw_sentences):
    """Keep only Bengali Unicode characters (\\u0980-\\u09FF) and whitespace."""
    cleaned = []
    for sentence in raw_sentences:
        sentence = re.sub(r'[^\u0980-\u09FF\s]', ' ', sentence)
        tokens   = sentence.split()
        if tokens:
            cleaned.append(' '.join(tokens))
    return cleaned


# =============================================================================
#  STEPS 3 & 4 — Tokenize, CBOW pairs, train
# =============================================================================

EMBEDDING_SIZE = 50
WINDOW_SIZE    = 2
EPOCHS         = 150
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

    print(f"  Training CBOW model on Bengali ({EPOCHS} epochs) ...")
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


def cosine_nearest(query_word, embeddings, tokenizer, top_n=6, min_count=3):
    """
    Return the top_n nearest Bengali words to query_word by cosine similarity.
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
        and tokenizer.word_counts.get(tokenizer.index_word[i], 0) >= min_count
    ]
    word_sims.sort(key=lambda x: -x[1])
    return word_sims[:top_n]


# =============================================================================
#  PLOT A — PCA scatter of ALL Bengali vocabulary words
# =============================================================================

def plot_pca_all_words(embeddings, tokenizer, save_path, font_prop=None):
    print("  Running PCA on all Bengali vocabulary words...")

    pca    = PCA(n_components=2)
    result = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(18, 14))

    for word, idx in tokenizer.word_index.items():
        ax.scatter(result[idx, 0], result[idx, 1],
                   s=10, alpha=0.45, color='#e15759')
        ann = ax.annotate(word,
                          xy=(result[idx, 0], result[idx, 1]),
                          xytext=(4, 2), textcoords='offset points',
                          ha='right', va='bottom',
                          fontsize=4.5, alpha=0.65, color='#1a1209')
        if font_prop is not None:
            ann.set_fontproperties(font_prop)

    ax.set_title("Word Embeddings — PCA (All Bengali Words)\n"
                 "English-Bengali Parallel Corpus  ·  Bengali CBOW Model",
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
#  PLOT B — PCA scatter of TOP 50 most frequent Bengali words
# =============================================================================

def plot_pca_top_words(embeddings, tokenizer, save_path, font_prop=None, top_n=50):
    print(f"  Running PCA on top {top_n} most frequent Bengali words...")

    top_words   = [tokenizer.index_word[i] for i in range(1, top_n + 1)
                   if i in tokenizer.index_word]
    top_vectors = np.array([embeddings[tokenizer.word_index[w]] for w in top_words])

    pca    = PCA(n_components=2)
    result = pca.fit_transform(top_vectors)

    # Colour groups — known common Bengali word categories
    PRONOUNS = {'আমি', 'তুমি', 'আপনি', 'সে', 'তারা', 'আমরা',
                'আমার', 'তোমার', 'আপনার', 'তার', 'এটা', 'ওটা'}
    VERBS    = {'করে', 'আছে', 'চাই', 'হয়', 'যায়', 'করতে',
                'হবে', 'ছিল', 'করেছে', 'বলে', 'দিতে', 'নেই'}

    def word_color(w):
        if w in PRONOUNS: return '#e15759'   # red   — pronouns
        if w in VERBS:    return '#59a14f'   # green — verbs
        return '#f28e2b'                      # orange — other Bengali

    fig, ax = plt.subplots(figsize=(13, 10))

    for i, word in enumerate(top_words):
        color = word_color(word)
        ax.scatter(result[i, 0], result[i, 1], s=55, color=color,
                   zorder=3, edgecolors='white', linewidth=0.5)
        ann = ax.annotate(word, xy=(result[i, 0], result[i, 1]),
                          xytext=(5, 2), textcoords='offset points',
                          ha='right', va='bottom',
                          fontsize=9, fontweight='bold', color='#1a1209')
        if font_prop is not None:
            ann.set_fontproperties(font_prop)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#e15759', label='Pronouns'),
        Patch(facecolor='#59a14f', label='Verbs'),
        Patch(facecolor='#f28e2b', label='Other'),
    ], fontsize=9, loc='upper right')

    ax.set_title(f"Word Embeddings — PCA (Top {top_n} Bengali Words)\n"
                 "English-Bengali Parallel Corpus  ·  Bengali CBOW Model",
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
#  PLOT C — Nearest-neighbour bar charts (Bengali only)
# =============================================================================

def plot_nearest_neighbours(embeddings, tokenizer, query_words, translations,
                             save_path, font_prop=None, top_n=6):
    """
    translations : dict mapping Bengali query word -> English meaning string.
                   Shown in each panel title so the chart is readable without
                   knowing Bengali.
    """
    print("  Computing Bengali nearest neighbours...")

    valid  = [w for w in query_words if w in tokenizer.word_index]
    n      = len(valid)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.8))
    axes = axes.flatten()

    BEN_COLOR = '#e15759'

    for ax_idx, word in enumerate(valid):
        nbrs = cosine_nearest(word, embeddings, tokenizer, top_n=top_n)
        ax   = axes[ax_idx]

        meaning = translations.get(word, '')
        title   = f'"{word}"  (= {meaning})\n→  Nearest Bengali words'

        if nbrs is None:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'not in vocabulary',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='grey')
            ax.axis('off')
            continue

        nbr_words  = [p[0] for p in nbrs]
        nbr_scores = [p[1] for p in nbrs]

        ax.barh(nbr_words[::-1], nbr_scores[::-1],
                color=BEN_COLOR, alpha=0.85, edgecolor='white')

        for i, sc in enumerate(nbr_scores[::-1]):
            ax.text(max(sc - 0.03, 0.02), i, f'{sc:.3f}',
                    va='center', ha='right', fontsize=7,
                    color='white', fontweight='bold')

        # Apply Bengali font to both the y-axis labels and the title
        if font_prop is not None:
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)

        ax.set_title(title, fontsize=9, fontweight='bold', color=BEN_COLOR)
        ax.set_xlabel('Cosine Similarity', fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    for ax_idx in range(len(valid), len(axes)):
        axes[ax_idx].axis('off')

    fig.suptitle(
        "Nearest Neighbours by Cosine Similarity  —  Bengali Model\n"
        "English-Bengali Parallel Corpus  ·  Bengali CBOW",
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

    # Curated report examples: mostly strong contextual neighbours, plus
    # one weaker/common word example for honest analysis.
    QUERY_WORDS = [
        'আপনি', 'তুমি', 'এটা', 'চাই', 'সে', 'আমি',
    ]

    TRANSLATIONS = {
        'আমি':  'I',
        'করে':  'does / doing',
        'না':   'no',
        'আমার': 'my',
        'কি':   'what',
        'আপনি': 'you (formal)',
        'এটা':  'this / it',
        'তুমি': 'you (informal)',
        'আছে':  'is / exists',
        'চাই':  'want',
        'আমরা': 'we',
        'সে':   'he / she',
    }

    print("=" * 65)
    print("TASKS 2.1 & 2.2 — BENGALI DATA PREP + WORD EMBEDDING")
    print("  Training on Bengali sentences only")
    print("=" * 65)

    # ── Bengali font setup ───────────────────────────────────────────────────
    print("\n[0] Setting up Bengali font...")
    font_name = setup_bengali_font()
    font_prop = fm.FontProperties(family=font_name) if font_name else None

    # ── Load & preprocess Bengali only ──────────────────────────────────────
    print(f"\n[1] Loading Bengali sentences from: {TSV_PATH}")
    raw_sentences = load_bengali(TSV_PATH)
    sentences     = preprocess_bengali(raw_sentences)
    print(f"      {len(sentences):,} Bengali sentences loaded")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[2] Building and training Bengali CBOW model (~1-2 minutes)...")
    model, tokenizer = build_and_train(sentences)
    vocab_size = len(tokenizer.word_index)
    print(f"      Bengali vocabulary size: {vocab_size:,} words")

    # ── Extract embeddings ───────────────────────────────────────────────────
    print("\n[3] Extracting embedding matrix...")
    embeddings, words = get_embeddings(model, tokenizer)
    print(f"      Shape: {embeddings.shape}  "
          f"({embeddings.shape[0]} words x {embeddings.shape[1]} dims)")

    # ── Console nearest-neighbour report ─────────────────────────────────────
    print("\n[4] Nearest Bengali neighbours (cosine similarity):")
    print(f"      {'Query':<12}  {'Meaning':<18}  Nearest words")
    print(f"      {'-'*12}  {'-'*18}  {'-'*45}")
    for word in QUERY_WORDS:
        nbrs    = cosine_nearest(word, embeddings, tokenizer, top_n=5)
        meaning = TRANSLATIONS.get(word, '?')
        if nbrs is None:
            print(f"      {word:<12}  {meaning:<18}  (not in vocabulary)")
        else:
            result_str = ',  '.join(f"{w}({s:.3f})" for w, s in nbrs)
            print(f"      {word:<12}  {meaning:<18}  {result_str}")

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\n[5] Generating Bengali visualizations...")

    print("\n  Plot A — PCA all Bengali words:")
    plot_pca_all_words(embeddings, tokenizer, OUT_PCA_ALL, font_prop)

    print("\n  Plot B — PCA top 50 Bengali words:")
    plot_pca_top_words(embeddings, tokenizer, OUT_PCA_TOP, font_prop, top_n=50)

    print("\n  Plot C — Bengali nearest neighbours:")
    plot_nearest_neighbours(embeddings, tokenizer, QUERY_WORDS, TRANSLATIONS,
                            OUT_NEAREST, font_prop, top_n=6)

    print("\n" + "=" * 65)
    print("Bengali model complete. Files saved:")
    print(f"  1. word_based_ben_pca_all_words.png      — PCA of all {vocab_size:,} Bengali words")
    print(f"  2. word_based_ben_pca_top_words.png      — PCA of top 50 Bengali words")
    print(f"  3. word_based_ben_nearest_neighbours.png — nearest-neighbour charts")
    print("=" * 65)
