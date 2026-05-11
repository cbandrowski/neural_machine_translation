# =============================================================================
#  WORD EMBEDDING WITH CBOW — New Dataset Sample (English + Bengali)
#
#  Task 2.2:
#    1. Train separate CBOW word embeddings for English and Bengali
#    2. Assess them with PCA visualizations and nearest-neighbour examples
#
#  DATA:
#    newDS/data/newds_subset.en
#    newDS/data/newds_subset.bn
#
#  HOW TO RUN:
#    python word_embedding_cbow_newds.py
#
#  OUTPUT:
#    output/eng_cbow_pca_sample.png
#    output/eng_cbow_pca_top_words.png
#    output/eng_cbow_nearest_neighbours.png
#    output/eng_cbow_nearest_neighbours.txt
#    output/ben_cbow_pca_sample.png
#    output/ben_cbow_pca_top_words.png
#    output/ben_cbow_nearest_neighbours.png
#    output/ben_cbow_nearest_neighbours.txt
# =============================================================================

import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"

ENG_PATH = DATA_DIR / "newds_subset.en"
BEN_PATH = DATA_DIR / "newds_subset.bn"

EMBEDDING_DIM = 64
WINDOW_SIZE = 2
EPOCHS = 12
BATCH_SIZE = 1024
MAX_WINDOWS = 300_000
MAX_VOCAB_ENG = 20_000
MAX_VOCAB_BEN = 30_000
TOP_WORDS_FOR_PCA = 80
PCA_SAMPLE_SIZE = 1200
TOP_NEIGHBOURS = 6
QUERY_WORDS_ENG = ["i", "you", "he", "she", "we", "go", "good", "thank", "help", "name"]
QUERY_WORDS_BEN = ["আমি", "তুমি", "সে", "আমরা", "যাও", "ভালো", "ধন্যবাদ", "সাহায্য", "নাম"]


def find_bengali_font():
    search_names = [
        'NotoSansBengali', 'NotoSerifBengali',
        'Vrinda', 'Lohit-Bengali', 'MuktiBangla', 'SolaimanLipi',
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
        str(BASE_DIR),
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

    for name in search_names:
        try:
            path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            if path:
                return path
        except Exception:
            pass
    return None


def setup_bengali_font():
    font_path = find_bengali_font()
    if not font_path:
        print("  WARNING: No Bengali font found — Bengali labels may render poorly.")
        return None

    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
    print(f"  Bengali font loaded: {font_name}  ({font_path})")
    return prop


def load_sentences(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def build_tokenizer(sentences, max_vocab):
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<unk>", filters='')
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def build_cbow_dataset(tokenizer, sentences, max_vocab, max_windows):
    sequences = tokenizer.texts_to_sequences(sentences)
    contexts, targets = [], []

    for seq in sequences:
        if len(seq) < (2 * WINDOW_SIZE + 1):
            continue
        for i in range(WINDOW_SIZE, len(seq) - WINDOW_SIZE):
            target = seq[i]
            if target == 0 or target >= max_vocab:
                continue
            context = seq[i - WINDOW_SIZE:i] + seq[i + 1:i + WINDOW_SIZE + 1]
            if any(tok == 0 or tok >= max_vocab for tok in context):
                continue
            contexts.append(context)
            targets.append(target)

    if not contexts:
        raise ValueError("No CBOW windows were generated.")

    if len(contexts) > max_windows:
        idx = np.random.default_rng(SEED).choice(len(contexts), size=max_windows, replace=False)
        contexts = np.array(contexts, dtype=np.int32)[idx]
        targets = np.array(targets, dtype=np.int32)[idx]
    else:
        contexts = np.array(contexts, dtype=np.int32)
        targets = np.array(targets, dtype=np.int32)

    return contexts, targets


def train_cbow_model(sentences, max_vocab, label):
    tokenizer = build_tokenizer(sentences, max_vocab)
    actual_vocab = min(len(tokenizer.word_index) + 1, max_vocab)
    X, y = build_cbow_dataset(tokenizer, sentences, actual_vocab, MAX_WINDOWS)

    model = Sequential([
        Embedding(input_dim=actual_vocab, output_dim=EMBEDDING_DIM,
                  input_length=2 * WINDOW_SIZE, name=f'{label}_embedding'),
        Lambda(lambda x: tf.reduce_mean(x, axis=1)),
        Dense(units=actual_vocab, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    print(f"  Training {label} CBOW model")
    print(f"    Vocabulary size used : {actual_vocab:,}")
    print(f"    Context windows used : {len(X):,}")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    embeddings = model.layers[0].get_weights()[0]
    return model, tokenizer, embeddings, actual_vocab


def get_trained_words(tokenizer, actual_vocab):
    return [tokenizer.index_word[i] for i in range(1, actual_vocab) if i in tokenizer.index_word]


def choose_query_words(words, candidates):
    chosen = [w for w in candidates if w in words]
    if len(chosen) >= 4:
        return chosen[:4]
    frequent_words = words[:4]
    for word in frequent_words:
        if word not in chosen:
            chosen.append(word)
        if len(chosen) == 4:
            break
    return chosen


def nearest_neighbours(query_word, embeddings, tokenizer, actual_vocab, top_n=TOP_NEIGHBOURS):
    if query_word not in tokenizer.word_index:
        return None

    query_idx = tokenizer.word_index[query_word]
    if query_idx >= actual_vocab:
        return None

    query_vec = embeddings[query_idx].reshape(1, -1)
    trained_vectors = embeddings[1:actual_vocab]
    sims = cosine_similarity(query_vec, trained_vectors)[0]

    items = []
    for idx in range(1, actual_vocab):
        if idx == query_idx or idx not in tokenizer.index_word:
            continue
        items.append((tokenizer.index_word[idx], float(sims[idx - 1])))
    items.sort(key=lambda x: -x[1])
    return items[:top_n]


def plot_pca_sample(embeddings, tokenizer, actual_vocab, save_path, title, font_prop=None):
    words = get_trained_words(tokenizer, actual_vocab)
    if len(words) > PCA_SAMPLE_SIZE:
        sampled = words[:PCA_SAMPLE_SIZE]
    else:
        sampled = words

    vectors = np.array([embeddings[tokenizer.word_index[w]] for w in sampled])
    result = PCA(n_components=2).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(result[:, 0], result[:, 1], s=12, alpha=0.5, color='#4e79a7')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_pca_top_words(embeddings, tokenizer, actual_vocab, save_path, title, font_prop=None):
    words = get_trained_words(tokenizer, actual_vocab)[:TOP_WORDS_FOR_PCA]
    vectors = np.array([embeddings[tokenizer.word_index[w]] for w in words])
    result = PCA(n_components=2).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, word in enumerate(words):
        ax.scatter(result[i, 0], result[i, 1], s=45, color='#e15759', alpha=0.85)
        ax.annotate(
            word,
            xy=(result[i, 0], result[i, 1]),
            xytext=(4, 2),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=8,
            fontproperties=font_prop,
        )

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_nearest_neighbours(query_words, embeddings, tokenizer, actual_vocab, save_path, title, font_prop=None):
    rows = len(query_words)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3.2 * rows))
    if rows == 1:
        axes = [axes]

    for ax, query_word in zip(axes, query_words):
        neighbours = nearest_neighbours(query_word, embeddings, tokenizer, actual_vocab)
        labels = [w for w, _ in neighbours] if neighbours else []
        scores = [s for _, s in neighbours] if neighbours else []

        ax.barh(range(len(labels)), scores, color='#59a14f', alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontproperties=font_prop)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Cosine similarity")
        ax.set_title(f"Nearest words to '{query_word}'", fontproperties=font_prop)
        ax.grid(True, axis='x', alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def write_nearest_neighbours_text(query_words, embeddings, tokenizer, actual_vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for query_word in query_words:
            f.write(f"{query_word}\n")
            neighbours = nearest_neighbours(query_word, embeddings, tokenizer, actual_vocab)
            if neighbours is None:
                f.write("  <not in trained vocabulary>\n\n")
                continue
            for word, score in neighbours:
                f.write(f"  {word}\t{score:.4f}\n")
            f.write("\n")
    print(f"  Saved -> {save_path}")


def run_language_pipeline(sentences, max_vocab, label, query_candidates, font_prop=None):
    model, tokenizer, embeddings, actual_vocab = train_cbow_model(sentences, max_vocab, label)
    trained_words = get_trained_words(tokenizer, actual_vocab)
    query_words = choose_query_words(trained_words, query_candidates)

    colour_title = "English" if label == "eng" else "Bengali"
    plot_pca_sample(
        embeddings, tokenizer, actual_vocab,
        OUT_DIR / f"{label}_cbow_pca_sample.png",
        f"{colour_title} CBOW Embeddings — PCA Sample",
        font_prop=font_prop,
    )
    plot_pca_top_words(
        embeddings, tokenizer, actual_vocab,
        OUT_DIR / f"{label}_cbow_pca_top_words.png",
        f"{colour_title} CBOW Embeddings — Top Frequent Words",
        font_prop=font_prop,
    )
    plot_nearest_neighbours(
        query_words, embeddings, tokenizer, actual_vocab,
        OUT_DIR / f"{label}_cbow_nearest_neighbours.png",
        f"{colour_title} CBOW Embeddings — Nearest Neighbours",
        font_prop=font_prop,
    )
    write_nearest_neighbours_text(
        query_words, embeddings, tokenizer, actual_vocab,
        OUT_DIR / f"{label}_cbow_nearest_neighbours.txt",
    )

    return {
        "vocab_used": actual_vocab,
        "query_words": query_words,
        "trained_words": len(trained_words),
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  NEW DATASET WORD EMBEDDING — CBOW")
    print("=" * 72)
    print(f"English data: {ENG_PATH}")
    print(f"Bengali data: {BEN_PATH}")
    print(f"Output dir  : {OUT_DIR}")

    font_prop = setup_bengali_font()
    eng_sentences = load_sentences(ENG_PATH)
    ben_sentences = load_sentences(BEN_PATH)

    print(f"\nLoaded {len(eng_sentences):,} English sentences")
    eng_info = run_language_pipeline(
        eng_sentences,
        MAX_VOCAB_ENG,
        "eng",
        QUERY_WORDS_ENG,
        font_prop=None,
    )

    print(f"\nLoaded {len(ben_sentences):,} Bengali sentences")
    ben_info = run_language_pipeline(
        ben_sentences,
        MAX_VOCAB_BEN,
        "ben",
        QUERY_WORDS_BEN,
        font_prop=font_prop,
    )

    print("\nSummary")
    print(f"  English vocab used : {eng_info['vocab_used']:,}")
    print(f"  Bengali vocab used : {ben_info['vocab_used']:,}")
    print(f"  English query words: {', '.join(eng_info['query_words'])}")
    print(f"  Bengali query words: {', '.join(ben_info['query_words'])}")
