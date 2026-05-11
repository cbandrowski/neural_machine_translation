# =============================================================================
#  WORD EMBEDDING IN NLP — English–Bengali Parallel Corpus 
#  TASKS 2.1 & 2.2: Data Preparation + Word Embedding Assessment
#
#  DATA:  ben-eng/ben.txt  (tab-separated: English \t Bengali \t [ignored])
#
#  BENGALI FONT SETUP (required to display Bengali characters in plots):
#  If Bengali words appear as squares, install the Noto Sans Bengali font:
#
#    macOS:
#      brew install font-noto-sans-bengali
#    OR manually download NotoSansBengali-Regular.ttf from:
#      https://fonts.google.com/noto/specimen/Noto+Sans+Bengali
#    and place it anywhere — the script will find it automatically.
#
#    Linux:
#      sudo apt install fonts-noto-cjk   OR   pip install matplotlib-fontja
#
#  OUTPUT:
#    word_embedding_pca_all_words.png       — PCA scatter of entire vocabulary
#    word_embedding_pca_top_words.png       — PCA scatter of top 50 words, labelled
#    word_embedding_nearest_neighbours.png  — nearest-neighbour bar charts, split by
#                                    language (English | Bengali side-by-side)
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
import matplotlib.font_manager as fm

import tensorflow as tf
from tensorflow.keras.models             import Sequential
from tensorflow.keras.layers             import Dense, Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils              import to_categorical

from sklearn.decomposition    import PCA
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
#  BENGALI FONT SETUP
#  Searches common locations for any Noto Bengali / Vrinda / Mukti font.
#  Falls back to default if none found (Bengali will show as squares).
# =============================================================================

def find_bengali_font():
    """
    Search common system and user font locations for a Bengali-capable font.
    Returns a font path string if found, otherwise None.
    """
    search_names = [
        'NotoSansBengali', 'NotoSerifBengali',
        'Vrinda', 'Lohit-Bengali', 'MuktiBangla', 'SolaimanLipi',
        'Kalpurush', 'AdorshoLipi',
    ]

    # Directories to search
    search_dirs = [
        # macOS system & user fonts
        '/Library/Fonts',
        os.path.expanduser('~/Library/Fonts'),
        '/System/Library/Fonts',
        # Linux
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        os.path.expanduser('~/.fonts'),
        os.path.expanduser('~/.local/share/fonts'),
        # Script's own folder (drop font here as fallback)
        os.path.dirname(os.path.abspath(__file__)),
    ]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.endswith(('.ttf', '.otf')):
                    for name in search_names:
                        if name.lower() in fname.lower():
                            return os.path.join(root, fname)

    # Last resort: ask matplotlib's font manager
    for name in search_names:
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if path:
                return path
        except Exception:
            pass

    return None


def setup_bengali_font():
    """
    Configure matplotlib to use a Bengali-capable font.
    Returns the font name that was registered (or None if not found).
    """
    font_path = find_bengali_font()
    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        matplotlib.rcParams['font.family'] = font_name
        # Also add it as a fallback for any font family
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        print(f"  Bengali font loaded: {font_name}  ({font_path})")
        return font_name
    else:
        print("  WARNING: No Bengali font found — Bengali text will appear as squares.")
        print("  To fix: download NotoSansBengali-Regular.ttf from fonts.google.com")
        print("  and place it in the same folder as this script, then re-run.")
        return None


# =============================================================================
#  FILE PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(BASE_DIR, "ben-eng", "ben.txt")

OUT_PCA_ALL = os.path.join(BASE_DIR, "word_embedding_pca_all_words.png")
OUT_PCA_TOP = os.path.join(BASE_DIR, "word_embedding_pca_top_words.png")
OUT_NEAREST = os.path.join(BASE_DIR, "word_embedding_nearest_neighbours.png")


# =============================================================================
#  LANGUAGE DETECTION
# =============================================================================

def is_bengali(word):
    return any('\u0980' <= ch <= '\u09FF' for ch in word)

def is_english(word):
    return all('a' <= ch <= 'z' for ch in word)


# =============================================================================
#  STEPS 1 & 2 — Load TSV and preprocess
# =============================================================================

def load_tsv(tsv_path):
    sentences = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            for col in (0, 1):
                text = parts[col].strip()
                if text:
                    sentences.append(text)
    return sentences


def preprocess(raw_sentences):
    cleaned = []
    for sentence in raw_sentences:
        sentence = sentence.replace("’", "'")
        sentence = re.sub(r"[^\u0980-\u09FFa-zA-Z\s']", ' ', sentence)
        sentence = re.sub(r"\s+'\s+", ' ', sentence)
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

    print(f"  Training CBOW model ({EPOCHS} epochs) ...")
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


def cosine_nearest_by_language(query_word, embeddings, tokenizer, top_n=6):
    """
    Return two ranked lists: closest English words, closest Bengali words.
    Returns (None, None) if query word not in vocab.
    """
    if query_word not in tokenizer.word_index:
        return None, None

    query_idx = tokenizer.word_index[query_word]
    query_vec = embeddings[query_idx].reshape(1, -1)
    all_sims  = cosine_similarity(query_vec, embeddings)[0]

    eng, ben = [], []
    for i in range(1, len(tokenizer.word_index) + 1):
        if i == query_idx:
            continue
        word  = tokenizer.index_word[i]
        score = float(all_sims[i])
        if is_bengali(word):
            ben.append((word, score))
        elif is_english(word):
            eng.append((word, score))

    eng.sort(key=lambda x: -x[1])
    ben.sort(key=lambda x: -x[1])
    return eng[:top_n], ben[:top_n]


# =============================================================================
#  Helper: draw a single bar chart panel
# =============================================================================

def _draw_bar_panel(ax, items, color, title, font_prop=None, use_bengali_labels=False):
    """Draw a horizontal bar chart for a list of (word, score) items."""
    if not items:
        ax.text(0.5, 0.5, 'No neighbours found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='grey')
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
        return

    words  = [p[0] for p in items]
    scores = [p[1] for p in items]

    ax.barh(words[::-1], scores[::-1],
            color=color, alpha=0.85, edgecolor='white')

    for i, sc in enumerate(scores[::-1]):
        ax.text(max(sc - 0.03, 0.02), i, f'{sc:.3f}',
                va='center', ha='right', fontsize=7,
                color='white', fontweight='bold')

    # Apply Bengali font to y-axis tick labels when showing Bengali words
    if use_bengali_labels and font_prop is not None:
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)

    ax.set_title(title, fontsize=9, fontweight='bold', color=color, pad=6)
    ax.set_xlabel('Cosine Similarity', fontsize=8)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)


# =============================================================================
#  PLOT A — PCA all words  (blue = English, red = Bengali)
# =============================================================================

def plot_pca_all_words(embeddings, tokenizer, save_path, font_prop=None):
    print("  Running PCA on all vocabulary words...")

    pca    = PCA(n_components=2)
    result = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(18, 14))

    for word, idx in tokenizer.word_index.items():
        color = '#e15759' if is_bengali(word) else '#4e79a7'
        ax.scatter(result[idx, 0], result[idx, 1], s=10, alpha=0.45, color=color)
        ann = ax.annotate(word, xy=(result[idx, 0], result[idx, 1]),
                          xytext=(4, 2), textcoords='offset points',
                          ha='right', va='bottom',
                          fontsize=4.5, alpha=0.65, color='#1a1209')
        if font_prop is not None and is_bengali(word):
            ann.set_fontproperties(font_prop)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#4e79a7', label='English'),
                        Patch(facecolor='#e15759', label='Bengali')],
              fontsize=10, loc='upper right')
    ax.set_title("Word Embeddings — PCA (All Words)\n"
                 "English-Bengali Parallel Corpus  ·  CBOW Model",
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
#  PLOT B — PCA top 50 words  (blue = English, red = Bengali)
# =============================================================================

def plot_pca_top_words(embeddings, tokenizer, save_path, font_prop=None, top_n=50):
    print(f"  Running PCA on top {top_n} most frequent words...")

    top_words   = [tokenizer.index_word[i] for i in range(1, top_n + 1)
                   if i in tokenizer.index_word]
    top_vectors = np.array([embeddings[tokenizer.word_index[w]] for w in top_words])

    pca    = PCA(n_components=2)
    result = pca.fit_transform(top_vectors)

    fig, ax = plt.subplots(figsize=(13, 10))

    for i, word in enumerate(top_words):
        color = '#e15759' if is_bengali(word) else '#4e79a7'
        ax.scatter(result[i, 0], result[i, 1], s=55, color=color,
                   zorder=3, edgecolors='white', linewidth=0.5)
        ann = ax.annotate(word, xy=(result[i, 0], result[i, 1]),
                          xytext=(5, 2), textcoords='offset points',
                          ha='right', va='bottom',
                          fontsize=9, fontweight='bold', color='#1a1209')
        if font_prop is not None and is_bengali(word):
            ann.set_fontproperties(font_prop)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#4e79a7', label='English words'),
                        Patch(facecolor='#e15759', label='Bengali words')],
              fontsize=9, loc='upper right')
    ax.set_title(f"Word Embeddings — PCA (Top {top_n} Words)\n"
                 "English-Bengali Parallel Corpus  ·  CBOW Model",
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
#  PLOT C — Nearest neighbours split by language
#
#  English plot:  each row = one English query word
#    Left  (blue) = nearest OTHER English words
#    Right (red)  = nearest Bengali words (translations / related)
#
#  Bengali plot:  each row = one Bengali query word + its English meaning
#    Left  (blue) = nearest English words
#    Right (red)  = nearest OTHER Bengali words
# =============================================================================

def plot_nearest_neighbours(embeddings, tokenizer, query_words, save_path,
                             font_prop=None, top_n=6, translations=None):
    """
    translations : optional dict mapping query_word -> English meaning string.
                   When provided the meaning is shown in the panel title.
    """
    print("  Computing nearest neighbours (split by language)...")

    valid = [w for w in query_words if w in tokenizer.word_index]
    n     = len(valid)

    fig, axes = plt.subplots(n, 2, figsize=(13, n * 3.2))
    if n == 1:
        axes = np.array([axes])

    ENG_COLOR = '#4e79a7'
    BEN_COLOR = '#e15759'

    for row, word in enumerate(valid):
        eng_nbrs, ben_nbrs = cosine_nearest_by_language(
            word, embeddings, tokenizer, top_n=top_n)

        # Build a meaning suffix for Bengali queries e.g.  " (= I)"
        meaning = ''
        if translations and word in translations:
            meaning = f'  (= {translations[word]})'

        # ── Left panel: nearest English words ────────────────────────────────
        eng_title = f'"{word}"{meaning}  →  Nearest English words'
        _draw_bar_panel(axes[row, 0], eng_nbrs, ENG_COLOR,
                        eng_title, font_prop=font_prop,
                        use_bengali_labels=False)

        # ── Right panel: nearest Bengali words ───────────────────────────────
        ben_title = f'"{word}"{meaning}  →  Nearest Bengali words'
        _draw_bar_panel(axes[row, 1], ben_nbrs, BEN_COLOR,
                        ben_title, font_prop=font_prop,
                        use_bengali_labels=True)   # Bengali font on y-axis

    fig.suptitle(
        "Nearest Neighbours by Cosine Similarity  —  Split by Language\n"
        "English-Bengali Parallel Corpus  ·  CBOW Model",
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

    # Most frequent English words in the corpus
    ENG_QUERY_WORDS = [
        'you',   'can',   'the',   'is',
        'do',    'he',    'what',  'want',
        'have',  'know',  'go',    'like',
    ]

    # Most frequent Bengali words in the corpus
    BEN_QUERY_WORDS = [
        'আমি',    'করে',    'না',     'আমার',
        'কি',     'আপনি',   'এটা',    'তুমি',
        'আছে',    'চাই',    'আমরা',   'সে',
    ]

    # English translations shown in the Bengali plot headings
    BEN_TRANSLATIONS = {
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

    # Combined — used for the single nearest-neighbour plot
    QUERY_WORDS = ENG_QUERY_WORDS + BEN_QUERY_WORDS

    print("=" * 65)
    print("TASKS 2.1 & 2.2 — WORD EMBEDDING ASSESSMENT")
    print("  Data: English-Bengali Parallel Corpus (ben.txt)")
    print("=" * 65)

    # ── Bengali font setup ───────────────────────────────────────────────────
    print("\n[0] Setting up Bengali font...")
    font_name = setup_bengali_font()
    font_prop = fm.FontProperties(family=font_name) if font_name else None

    # ── Load & preprocess ────────────────────────────────────────────────────
    print(f"\n[1] Loading TSV: {TSV_PATH}")
    raw_sentences = load_tsv(TSV_PATH)
    sentences     = preprocess(raw_sentences)
    print(f"      {len(raw_sentences):,} raw -> {len(sentences):,} cleaned sentences")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[2] Building and training CBOW model (~2-5 minutes)...")
    model, tokenizer = build_and_train(sentences)
    vocab_size = len(tokenizer.word_index)
    n_eng      = sum(1 for w in tokenizer.word_index if is_english(w))
    n_ben      = sum(1 for w in tokenizer.word_index if is_bengali(w))
    print(f"      Vocab: {vocab_size:,} total  ({n_eng:,} English | {n_ben:,} Bengali)")

    # ── Extract embeddings ───────────────────────────────────────────────────
    print("\n[3] Extracting embedding matrix...")
    embeddings, words = get_embeddings(model, tokenizer)
    print(f"      Shape: {embeddings.shape}")

    # ── Console report ───────────────────────────────────────────────────────
    print("\n[4] Nearest neighbours — split by language:")
    print("\n  English queries:")
    print(f"      {'Query':<10}  {'Top English':<45}  Top Bengali")
    print(f"      {'-'*10}  {'-'*45}  {'-'*40}")
    for word in ENG_QUERY_WORDS:
        eng, ben = cosine_nearest_by_language(word, embeddings, tokenizer, top_n=3)
        if eng is None:
            print(f"      {word:<10}  (not in vocabulary)")
            continue
        e_str = ', '.join(f"{w}({s:.2f})" for w, s in eng)
        b_str = ', '.join(f"{w}({s:.2f})" for w, s in ben)
        print(f"      {word:<10}  {e_str:<45}  {b_str}")
    print("\n  Bengali queries:")
    print(f"      {'Query':<12}  {'Top English':<45}  Top Bengali")
    print(f"      {'-'*12}  {'-'*45}  {'-'*40}")
    for word in BEN_QUERY_WORDS:
        eng, ben = cosine_nearest_by_language(word, embeddings, tokenizer, top_n=3)
        if eng is None:
            print(f"      {word:<12}  (not in vocabulary)")
            continue
        e_str = ', '.join(f"{w}({s:.2f})" for w, s in eng)
        b_str = ', '.join(f"{w}({s:.2f})" for w, s in ben)
        print(f"      {word:<12}  {e_str:<45}  {b_str}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n[5] Generating visualizations...")

    print("\n  Plot A — PCA all words:")
    plot_pca_all_words(embeddings, tokenizer, OUT_PCA_ALL, font_prop)

    print("\n  Plot B — PCA top 50 words:")
    plot_pca_top_words(embeddings, tokenizer, OUT_PCA_TOP, font_prop, top_n=50)

    OUT_NEAREST_ENG = os.path.join(BASE_DIR, "word_embedding_nearest_neighbours_english.png")
    OUT_NEAREST_BEN = os.path.join(BASE_DIR, "word_embedding_nearest_neighbours_bengali.png")

    print("\n  Plot C1 — English query words nearest neighbours:")
    plot_nearest_neighbours(embeddings, tokenizer, ENG_QUERY_WORDS, OUT_NEAREST_ENG,
                            font_prop, top_n=6)

    print("\n  Plot C2 — Bengali query words nearest neighbours:")
    plot_nearest_neighbours(embeddings, tokenizer, BEN_QUERY_WORDS, OUT_NEAREST_BEN,
                            font_prop, top_n=6, translations=BEN_TRANSLATIONS)

    print("\n" + "=" * 65)
    print("Step 5 complete. Files saved:")
    print(f"  1. word_embedding_pca_all_words.png")
    print(f"  2. word_embedding_pca_top_words.png")
    print(f"  3. word_embedding_nearest_neighbours_english.png  (English query words)")
    print(f"  4. word_embedding_nearest_neighbours_bengali.png  (Bengali query words)")
    print("=" * 65)
