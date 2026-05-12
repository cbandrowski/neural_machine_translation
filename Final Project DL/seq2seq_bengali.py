# =============================================================================
#  TRANSLATION WITH SEQ2SEQ LSTM — Bengali → English
#
#  Task 2.3: Sequence-to-Sequence encoder-decoder using LSTMs.
#
#  Architecture:
#    - Bidirectional LSTM encoder
#    - LSTM decoder with additive attention and teacher forcing
#    - Separate embedding layers for each language (dim=128)
#  Justification: The plain encoder-decoder was producing fluent but
#  semantically wrong sentences. Attention improves source-target alignment,
#  especially on multi-word Bengali inputs.
#
#  BENGALI FONT SETUP (required to display Bengali characters in plots):
#    macOS:   brew install font-noto-sans-bengali
#    Linux:   sudo apt install fonts-noto-cjk
#    Manual:  download NotoSansBengali-Regular.ttf from fonts.google.com
#             and place it in the same folder as this script.
#
#  DATA:    ben-eng/ben.txt   (col 0 = English, col 1 = Bengali)
#  SPLIT:   70 % train, 30 % test
#
#  HOW TO RUN:
#    python seq2seq_bengali.py
#
#  OUTPUT (saved to same folder as this script):
#    word_based_ben_seq2seq_attention_training_curve.png   — loss / accuracy over epochs
#    word_based_ben_seq2seq_attention_bleu_histogram.png   — per-sentence BLEU distribution
# =============================================================================

import os, re, sys, warnings
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
from tensorflow.keras.models              import Model
from tensorflow.keras.layers              import (
    Input, Embedding, LSTM, Dense, Bidirectional, AdditiveAttention, Concatenate
)
from tensorflow.keras.callbacks           import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection              import train_test_split
from nltk.translate.bleu_score            import corpus_bleu, sentence_bleu, SmoothingFunction

from tensorflow.keras.optimizers import Adam

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)
SEED = 42

# =============================================================================
#  PATHS
# =============================================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TSV_PATH        = os.path.join(BASE_DIR, "ben-eng", "ben.txt")
OUT_CURVE       = os.path.join(BASE_DIR, "word_based_ben_seq2seq_attention_training_curve.png")
OUT_BLEU_HIST   = os.path.join(BASE_DIR, "word_based_ben_seq2seq_attention_bleu_histogram.png")
OUT_ATTENTION   = os.path.join(BASE_DIR, "word_based_ben_seq2seq_attention_heatmap.png")
EXPERIMENT_LOG  = os.path.join(BASE_DIR, "word_based_SEQ2SEQ_EXPERIMENT_LOG.md")

# =============================================================================
#  HYPER-PARAMETERS  (minimal / justified above)
# =============================================================================

EMBEDDING_DIM = 128
LSTM_UNITS    = 256
BATCH_SIZE    = 64
EPOCHS        = 50
TEST_SIZE     = 0.30
VAL_SIZE      = 0.10
DROPOUT       = 0.25
RECURRENT_DROPOUT = 0.10
LEARNING_RATE = 1e-3
CLIPNORM      = 1.0
MAX_VOCAB_BEN = 8000   # Bengali morphology inflates vocabulary
MAX_VOCAB_ENG = 5000   # cap rare English words
BEAM_WIDTH    = 3
SPLIT_MODE    = 'length_stratified'

START_TOKEN = '<start>'
END_TOKEN   = '<end>'
UNK_TOKEN   = '<unk>'

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
        print("  WARNING: No Bengali font found — Bengali text may appear as squares.")
        print("  Fix: place NotoSansBengali-Regular.ttf in this folder and re-run.")
        return None


# =============================================================================
#  1. DATA LOADING
# =============================================================================

def load_pairs(tsv_path):
    """Read (Bengali, English) pairs; skip malformed / empty lines."""
    pairs = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng = parts[0].strip()
                ben = parts[1].strip()
                if eng and ben:
                    # Direction: Bengali (input) → English (output)
                    pairs.append((ben, eng))
    return pairs


# =============================================================================
#  2. PREPROCESSING
# =============================================================================

def clean_bengali(text):
    """Keep Bengali Unicode block (U+0980-U+09FF) + danda + spaces."""
    text = re.sub(r"[^ঀ-৿।॥\s]", ' ', text)
    return ' '.join(text.split())


def clean_english(text):
    """Lower-case; keep only a-z and spaces."""
    text = re.sub(r"[^a-zA-Z\s]", ' ', text)
    return ' '.join(text.lower().split())


def preprocess(pairs):
    ben_sentences, eng_sentences = [], []
    for ben, eng in pairs:
        b = clean_bengali(ben)
        e = clean_english(eng)
        if b and e:
            ben_sentences.append(b)
            # Decoder input/target uses explicit boundary tokens
            eng_sentences.append(f"{START_TOKEN} {e} {END_TOKEN}")
    return ben_sentences, eng_sentences


def shuffle_parallel(source_sentences, target_sentences, seed=SEED):
    """Apply the same random permutation to both languages."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(source_sentences))
    src = [source_sentences[i] for i in indices]
    tgt = [target_sentences[i] for i in indices]
    return src, tgt


def length_buckets(source_sentences, target_sentences):
    """
    Coarse sentence-length buckets used for more balanced train/val/test splits.
    This avoids a split that is dominated by only very short or very long pairs.
    """
    buckets = []
    for src, tgt in zip(source_sentences, target_sentences):
        total_len = len(src.split()) + len(tgt.split())
        if total_len <= 4:
            buckets.append('very_short')
        elif total_len <= 8:
            buckets.append('short')
        elif total_len <= 12:
            buckets.append('medium')
        else:
            buckets.append('long')
    return buckets


# =============================================================================
#  3. TOKENISATION & PADDING
# =============================================================================

def build_tokenizer(sentences, num_words):
    tok = Tokenizer(num_words=num_words, oov_token=UNK_TOKEN, filters='')
    tok.fit_on_texts(sentences)
    return tok


# =============================================================================
#  4. MODEL CONSTRUCTION
# =============================================================================

def build_model(enc_vocab, dec_vocab):
    """
    Returns the training model and the two inference sub-models.
    Encoder reads Bengali; decoder generates English.
    Architecture: bidirectional LSTM encoder + attention-based LSTM decoder.
    """
    encoder_half_units = LSTM_UNITS // 2

    # ── Shared layer objects (reused by inference models) ────────────────────
    enc_emb_layer  = Embedding(enc_vocab, EMBEDDING_DIM, name='enc_embedding',
                                mask_zero=True)
    enc_lstm_layer = Bidirectional(
        LSTM(
            encoder_half_units,
            return_sequences=True,
            return_state=True,
            dropout=DROPOUT,
            recurrent_dropout=RECURRENT_DROPOUT,
        ),
        name='enc_bi_lstm'
    )

    dec_emb_layer  = Embedding(dec_vocab, EMBEDDING_DIM, name='dec_embedding',
                                mask_zero=True)
    dec_lstm_layer = LSTM(
        LSTM_UNITS,
        return_sequences=True,
        return_state=True,
        dropout=DROPOUT,
        recurrent_dropout=RECURRENT_DROPOUT,
        name='dec_lstm'
    )
    attention_layer = AdditiveAttention(name='attention')
    concat_layer    = Concatenate(axis=-1, name='context_concat')
    proj_layer      = Dense(LSTM_UNITS, activation='tanh', name='context_projection')
    dec_dense      = Dense(dec_vocab, activation='softmax', name='dec_output')

    # ── Training model (teacher forcing) ─────────────────────────────────────
    enc_in                       = Input(shape=(None,), name='encoder_input')
    enc_emb                      = enc_emb_layer(enc_in)
    enc_out_seq, fh, fc, bh, bc  = enc_lstm_layer(enc_emb)
    state_h                      = Concatenate(name='enc_state_h')([fh, bh])
    state_c                      = Concatenate(name='enc_state_c')([fc, bc])
    enc_states                   = [state_h, state_c]

    dec_in                    = Input(shape=(None,), name='decoder_input')
    dec_emb                   = dec_emb_layer(dec_in)
    dec_out_seq, _, _         = dec_lstm_layer(dec_emb, initial_state=enc_states)
    context_seq               = attention_layer([dec_out_seq, enc_out_seq])
    dec_context               = concat_layer([dec_out_seq, context_seq])
    dec_context               = proj_layer(dec_context)
    dec_out                   = dec_dense(dec_context)

    training_model = Model([enc_in, dec_in], dec_out)
    training_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Inference encoder ─────────────────────────────────────────────────────
    encoder_model = Model(enc_in, [enc_out_seq] + enc_states)

    # ── Inference decoder (one step at a time) ────────────────────────────────
    dec_state_h_in = Input(shape=(LSTM_UNITS,), name='dec_state_h_in')
    dec_state_c_in = Input(shape=(LSTM_UNITS,), name='dec_state_c_in')
    enc_out_in     = Input(shape=(None, LSTM_UNITS), name='enc_out_in')
    dec_states_in  = [dec_state_h_in, dec_state_c_in]

    dec_single_emb       = dec_emb_layer(dec_in)
    dec_single_out, h, c = dec_lstm_layer(dec_single_emb,
                                          initial_state=dec_states_in)
    context_single       = attention_layer([dec_single_out, enc_out_in])
    dec_single_context   = concat_layer([dec_single_out, context_single])
    dec_single_context   = proj_layer(dec_single_context)
    dec_single_dense     = dec_dense(dec_single_context)

    decoder_model = Model(
        [dec_in, enc_out_in] + dec_states_in,
        [dec_single_dense, h, c, dec_single_out]
    )

    return training_model, encoder_model, decoder_model


# =============================================================================
#  5. TRAINING DATA PREPARATION
# =============================================================================

def make_decoder_targets(dec_sequences):
    """Shift decoder sequences by one step to create targets."""
    dec_in  = dec_sequences[:, :-1]   # <start> … last-word
    dec_tgt = dec_sequences[:, 1:]    # first-word … <end>
    # targets must be 3-D for sparse CE: (batch, time, 1)
    dec_tgt = np.expand_dims(dec_tgt, -1)
    return dec_in, dec_tgt


def make_dataset(enc_seq, dec_in_seq, dec_tgt_seq, batch_size,
                 shuffle=False, seed=SEED):
    """Build a tf.data pipeline with optional per-epoch shuffling."""
    dataset = tf.data.Dataset.from_tensor_slices(((enc_seq, dec_in_seq), dec_tgt_seq))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(enc_seq),
            seed=seed,
            reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =============================================================================
#  6. GREEDY INFERENCE
# =============================================================================

def translate(sentence, encoder_model, decoder_model,
              ben_tok, eng_tok, max_dec_len=20):
    """Greedy decode one Bengali sentence → English string.

    Uses direct model __call__ instead of model.predict() to avoid
    per-call Python-TF overhead, which would freeze on large test sets.
    """
    # encode
    seq = ben_tok.texts_to_sequences([clean_bengali(sentence)])
    seq = tf.constant(
        pad_sequences(seq, maxlen=encoder_model.input_shape[1], padding='post')
    )
    enc_outputs, state_h, state_c = encoder_model(seq, training=False)
    states = [state_h, state_c]

    # start token
    start_idx = eng_tok.word_index.get(START_TOKEN, 1)
    end_idx   = eng_tok.word_index.get(END_TOKEN, 2)

    target_seq     = tf.constant([[start_idx]])
    decoded_tokens = []

    for _ in range(max_dec_len):
        output, h, c, _ = decoder_model([target_seq, enc_outputs] + states, training=False)
        token_idx = int(tf.argmax(output[0, -1, :]))
        if token_idx == end_idx or token_idx == 0:
            break
        word = eng_tok.index_word.get(token_idx, '')
        if word and word not in (START_TOKEN, END_TOKEN, UNK_TOKEN):
            decoded_tokens.append(word)
        target_seq = tf.constant([[token_idx]])
        states     = [h, c]

    return ' '.join(decoded_tokens)


def translate_beam(sentence, encoder_model, decoder_model,
                   ben_tok, eng_tok, max_dec_len=20, beam_width=3):
    """Beam-search decode one Bengali sentence -> English string."""
    seq = ben_tok.texts_to_sequences([clean_bengali(sentence)])
    seq = tf.constant(
        pad_sequences(seq, maxlen=encoder_model.input_shape[1], padding='post')
    )
    enc_outputs, state_h, state_c = encoder_model(seq, training=False)

    start_idx = eng_tok.word_index.get(START_TOKEN, 1)
    end_idx = eng_tok.word_index.get(END_TOKEN, 2)
    blocked = {
        0,
        eng_tok.word_index.get(UNK_TOKEN, -1),
    }

    beams = [([start_idx], 0.0, [state_h, state_c], False)]

    for _ in range(max_dec_len):
        candidates = []
        for tokens, score, states, finished in beams:
            if finished:
                candidates.append((tokens, score, states, finished))
                continue

            target_seq = tf.constant([[tokens[-1]]])
            output, h, c, _ = decoder_model([target_seq, enc_outputs] + states, training=False)
            probs = output[0, -1, :].numpy()

            top_indices = np.argsort(probs)[-beam_width * 3:][::-1]
            added = 0
            for token_idx in top_indices:
                token_idx = int(token_idx)
                if token_idx in blocked:
                    continue
                prob = max(float(probs[token_idx]), 1e-12)
                next_tokens = tokens + [token_idx]
                next_finished = token_idx == end_idx
                candidates.append((next_tokens, score + np.log(prob), [h, c], next_finished))
                added += 1
                if added >= beam_width:
                    break

        def normalized(item):
            tokens, score, _, _ = item
            return score / max(len(tokens) - 1, 1)

        beams = sorted(candidates, key=normalized, reverse=True)[:beam_width]
        if all(item[3] for item in beams):
            break

    best_tokens = sorted(beams, key=lambda item: item[1] / max(len(item[0]) - 1, 1), reverse=True)[0][0]
    decoded_tokens = []
    for token_idx in best_tokens[1:]:
        if token_idx == end_idx:
            break
        word = eng_tok.index_word.get(token_idx, '')
        if word and word not in (START_TOKEN, END_TOKEN, UNK_TOKEN):
            decoded_tokens.append(word)

    return ' '.join(decoded_tokens)


def translate_with_attention(sentence, encoder_model, decoder_model,
                             ben_tok, eng_tok, max_dec_len=20):
    """Greedy decode one sentence and collect attention weights per output word."""
    cleaned = clean_bengali(sentence)
    source_tokens = cleaned.split()
    seq = ben_tok.texts_to_sequences([cleaned])
    seq = tf.constant(
        pad_sequences(seq, maxlen=encoder_model.input_shape[1], padding='post')
    )
    enc_outputs, state_h, state_c = encoder_model(seq, training=False)
    states = [state_h, state_c]

    start_idx = eng_tok.word_index.get(START_TOKEN, 1)
    end_idx = eng_tok.word_index.get(END_TOKEN, 2)
    target_seq = tf.constant([[start_idx]])
    decoded_tokens = []
    attention_rows = []

    for _ in range(max_dec_len):
        output, h, c, dec_step = decoder_model(
            [target_seq, enc_outputs] + states,
            training=False
        )
        token_idx = int(tf.argmax(output[0, -1, :]))
        if token_idx == end_idx or token_idx == 0:
            break
        word = eng_tok.index_word.get(token_idx, '')
        if word and word not in (START_TOKEN, END_TOKEN, UNK_TOKEN):
            decoded_tokens.append(word)
            query = dec_step[0, -1, :].numpy()
            keys = enc_outputs[0, :len(source_tokens), :].numpy()
            scores = np.sum(np.tanh(keys + query), axis=-1)
            scores = scores - np.max(scores)
            weights = np.exp(scores) / np.sum(np.exp(scores))
            attention_rows.append(weights)
        target_seq = tf.constant([[token_idx]])
        states = [h, c]

    if attention_rows:
        attention_matrix = np.vstack(attention_rows)
    else:
        attention_matrix = np.zeros((1, max(len(source_tokens), 1)))

    return source_tokens, decoded_tokens, attention_matrix


# =============================================================================
#  7. BLEU SCORE
# =============================================================================

def compute_bleu(pairs_test, encoder_model, decoder_model,
                 ben_tok, eng_tok, max_dec_len=20,
                 decoder_name='greedy', beam_width=3):
    """Corpus-level BLEU-4 on the test set."""
    references, hypotheses = [], []
    total = len(pairs_test)

    for i, (ben, eng_ref_full) in enumerate(pairs_test):
        if i % 100 == 0:
            print(f"    Translating {i}/{total} ...", flush=True)

        # strip boundary tokens from reference
        ref_clean = re.sub(
            rf'\s*{re.escape(START_TOKEN)}\s*|\s*{re.escape(END_TOKEN)}\s*',
            ' ', eng_ref_full
        ).strip().split()

        if decoder_name == 'beam':
            hyp_text = translate_beam(
                ben, encoder_model, decoder_model,
                ben_tok, eng_tok, max_dec_len=max_dec_len,
                beam_width=beam_width
            )
        else:
            hyp_text = translate(
                ben, encoder_model, decoder_model,
                ben_tok, eng_tok, max_dec_len=max_dec_len
            )

        hyp = hyp_text.split()

        references.append([ref_clean])
        hypotheses.append(hyp)

    bleu = corpus_bleu(references, hypotheses,
                       smoothing_function=SmoothingFunction().method1)
    return bleu, references, hypotheses


def append_experiment_log(
    train_count, val_count, test_count, enc_vocab, dec_vocab,
    greedy_bleu, greedy_mean, greedy_median,
    beam_bleu, beam_mean, beam_median,
):
    entry = f"""

## Bengali to English, attention LSTM with beam-search comparison

- Script: `seq2seq_bengali.py`
- Tokenization: word-based
- Direction: Bengali -> English
- Split: {SPLIT_MODE}; 70% train, 30% test; 10% of training portion used for validation
- Model: bidirectional LSTM encoder + additive attention + LSTM decoder
- Embedding dimension: {EMBEDDING_DIM}
- Training pairs: {train_count:,}
- Validation pairs: {val_count:,}
- Test pairs: {test_count:,}
- Bengali vocabulary (encoder): {enc_vocab:,}
- English vocabulary (decoder): {dec_vocab:,}
- Greedy corpus BLEU-4: {greedy_bleu:.4f}
- Greedy mean sentence BLEU: {greedy_mean:.4f}
- Greedy median sentence BLEU: {greedy_median:.4f}
- Beam width: {BEAM_WIDTH}
- Beam corpus BLEU-4: {beam_bleu:.4f}
- Beam mean sentence BLEU: {beam_mean:.4f}
- Beam median sentence BLEU: {beam_median:.4f}
"""
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(entry)


# =============================================================================
#  8. PLOTS
# =============================================================================

def plot_training_curve(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'],     label='Train loss',    color='#e15759')
    ax1.plot(history.history['val_loss'], label='Val loss',      color='#f28e2b',
             linestyle='--')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(history.history['accuracy'],     label='Train accuracy', color='#e15759')
    ax2.plot(history.history['val_accuracy'], label='Val accuracy',   color='#f28e2b',
             linestyle='--')
    ax2.set_title('Training & Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Seq2Seq LSTM  —  Bengali → English', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_bleu_histogram(per_sentence_bleus, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_sentence_bleus, bins=30, color='#e15759', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(per_sentence_bleus), color='#4e79a7', linestyle='--',
               linewidth=1.8, label=f'Mean = {np.mean(per_sentence_bleus):.4f}')
    ax.set_title('Per-Sentence BLEU Distribution (Test Set)', fontweight='bold')
    ax.set_xlabel('Sentence BLEU'); ax.set_ylabel('Count')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_attention_heatmap(source_tokens, target_tokens, attention_matrix,
                           save_path, font_prop=None):
    if not target_tokens:
        target_tokens = ['(empty)']

    fig_width = max(7, 0.75 * len(source_tokens) + 2)
    fig_height = max(4, 0.45 * len(target_tokens) + 2)
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(attention_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention weight')
    plt.xticks(
        range(len(source_tokens)),
        source_tokens,
        rotation=35,
        ha='right',
        fontproperties=font_prop
    )
    plt.yticks(range(len(target_tokens)), target_tokens)
    plt.xlabel('Bengali source tokens', fontproperties=font_prop)
    plt.ylabel('English generated tokens')
    plt.title('Word-Based Attention Heatmap: Bengali -> English',
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("  SEQ2SEQ LSTM  —  Bengali → English Translation")
    print("=" * 70)

    # ── 0. Bengali font ───────────────────────────────────────────────────────
    print("\n[0] Setting up Bengali font for console display ...")
    font_name = setup_bengali_font()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1] Loading data from {TSV_PATH}")
    raw_pairs = load_pairs(TSV_PATH)
    print(f"    {len(raw_pairs):,} sentence pairs loaded")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2] Preprocessing ...")
    ben_sents, eng_sents = preprocess(raw_pairs)
    print(f"    {len(ben_sents):,} valid pairs after cleaning")

    # ── 3. Shuffle + train / val / test split ────────────────────────────────
    print(f"\n[3] Splitting data with {SPLIT_MODE} mode (test = {int(TEST_SIZE*100)}%, val = {int(VAL_SIZE*100)}% of train) ...")
    all_buckets = length_buckets(ben_sents, eng_sents)
    (ben_train_full, ben_test,
     eng_train_full, eng_test,
     train_buckets, _) = train_test_split(
        ben_sents, eng_sents, all_buckets,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=all_buckets,
        shuffle=True
    )

    (ben_train, ben_val,
     eng_train, eng_val) = train_test_split(
        ben_train_full, eng_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=train_buckets,
        shuffle=True
    )
    print(f"    Train: {len(ben_train):,}  |  Val: {len(ben_val):,}  |  Test: {len(ben_test):,}")

    # ── 4. Tokenise ───────────────────────────────────────────────────────────
    print("\n[4] Building tokenisers ...")
    ben_tok = build_tokenizer(ben_train, MAX_VOCAB_BEN)
    eng_tok = build_tokenizer(eng_train, MAX_VOCAB_ENG)

    enc_vocab = min(len(ben_tok.word_index) + 1, MAX_VOCAB_BEN)
    dec_vocab = min(len(eng_tok.word_index) + 1, MAX_VOCAB_ENG)
    print(f"    Bengali vocab (encoder) : {enc_vocab:,} tokens")
    print(f"    English vocab (decoder) : {dec_vocab:,} tokens")

    # ── 5. Encode & pad ───────────────────────────────────────────────────────
    print("\n[5] Encoding and padding sequences ...")
    max_enc = max(len(s.split()) for s in ben_train)
    max_dec = max(len(s.split()) for s in eng_train)
    print(f"    Max encoder length (Bengali) : {max_enc}")
    print(f"    Max decoder length (English) : {max_dec}")

    enc_train_seq = pad_sequences(
        ben_tok.texts_to_sequences(ben_train), maxlen=max_enc, padding='post')
    dec_train_seq = pad_sequences(
        eng_tok.texts_to_sequences(eng_train), maxlen=max_dec, padding='post')
    enc_val_seq = pad_sequences(
        ben_tok.texts_to_sequences(ben_val), maxlen=max_enc, padding='post')
    dec_val_seq = pad_sequences(
        eng_tok.texts_to_sequences(eng_val), maxlen=max_dec, padding='post')

    dec_in_train, dec_tgt_train = make_decoder_targets(dec_train_seq)
    dec_in_val, dec_tgt_val = make_decoder_targets(dec_val_seq)

    # ── 6. Build model ────────────────────────────────────────────────────────
    print("\n[6] Building seq2seq model ...")
    training_model, encoder_model, decoder_model = build_model(enc_vocab, dec_vocab)
    training_model.summary()

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n[7] Training for {EPOCHS} epochs (batch={BATCH_SIZE}) ...")
    train_ds = make_dataset(
        enc_train_seq, dec_in_train, dec_tgt_train,
        batch_size=BATCH_SIZE, shuffle=True, seed=SEED
    )
    val_ds = make_dataset(
        enc_val_seq, dec_in_val, dec_tgt_val,
        batch_size=BATCH_SIZE, shuffle=False, seed=SEED
    )
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
    ]
    history = training_model.fit(train_ds, epochs=EPOCHS,
                                 validation_data=val_ds,
                                 callbacks=callbacks,
                                 verbose=2)

    # ── 8. Example translations ───────────────────────────────────────────────
    # Bengali input sentences with their known English meanings
    EXAMPLE_SENTENCES = [
        ('যাও।',                 'Go.'),
        ('বাঁচাও!',              'Help!'),
        ('আমি খুশি।',            'I am happy.'),
        ('তোমার নাম কি?',        'What is your name?'),
        ('সে একজন শিক্ষক।',     'She is a teacher.'),
        ('সে জানে না।',          'He does not know.'),
        ('আমরা বন্ধু।',          'We are friends.'),
        ('আমি বাড়ি যেতে চাই।', 'I want to go home.'),
        ('তুমি কি আমাকে সাহায্য করতে পারবে?', 'Can you help me?'),
        ('অনেক ধন্যবাদ।',        'Thank you very much.'),
    ]

    print("\n[8] Example translations:")
    print(f"    {'Bengali Input':<40}  {'Reference English':<28}  {'Greedy':<30}  Beam")
    print(f"    {'-'*40}  {'-'*28}  {'-'*30}  {'-'*30}")
    for ben_sent, ref_eng in EXAMPLE_SENTENCES:
        greedy_prediction = translate(ben_sent, encoder_model, decoder_model,
                                      ben_tok, eng_tok, max_dec_len=max_dec)
        beam_prediction = translate_beam(ben_sent, encoder_model, decoder_model,
                                         ben_tok, eng_tok, max_dec_len=max_dec,
                                         beam_width=BEAM_WIDTH)
        print(f"    {ben_sent:<40}  {ref_eng:<28}  {greedy_prediction if greedy_prediction else '(empty)':<30}  {beam_prediction if beam_prediction else '(empty)'}")

    # ── 9. BLEU score ─────────────────────────────────────────────────────────
    print("\n[9] Computing greedy BLEU score on test set ...")
    test_pairs = list(zip(ben_test, eng_test))
    greedy_bleu_score, greedy_references, greedy_hypotheses = compute_bleu(
        test_pairs, encoder_model, decoder_model,
        ben_tok, eng_tok, max_dec_len=max_dec,
        decoder_name='greedy'
    )

    print(f"\n[9b] Computing beam-search BLEU score on test set (beam width = {BEAM_WIDTH}) ...")
    beam_bleu_score, beam_references, beam_hypotheses = compute_bleu(
        test_pairs, encoder_model, decoder_model,
        ben_tok, eng_tok, max_dec_len=max_dec,
        decoder_name='beam',
        beam_width=BEAM_WIDTH
    )

    # per-sentence BLEU for histograms/summary
    smooth = SmoothingFunction().method1
    greedy_per_sent_bleus = []
    for ref, hyp in zip(greedy_references, greedy_hypotheses):
        s = sentence_bleu(ref, hyp, smoothing_function=smooth)
        greedy_per_sent_bleus.append(s)
    beam_per_sent_bleus = []
    for ref, hyp in zip(beam_references, beam_hypotheses):
        s = sentence_bleu(ref, hyp, smoothing_function=smooth)
        beam_per_sent_bleus.append(s)

    print(f"\n    Greedy corpus BLEU-4       : {greedy_bleu_score:.4f}")
    print(f"    Greedy mean sentence BLEU  : {np.mean(greedy_per_sent_bleus):.4f}")
    print(f"    Greedy median sentence BLEU: {np.median(greedy_per_sent_bleus):.4f}")
    print(f"\n    Beam corpus BLEU-4         : {beam_bleu_score:.4f}")
    print(f"    Beam mean sentence BLEU    : {np.mean(beam_per_sent_bleus):.4f}")
    print(f"    Beam median sentence BLEU  : {np.median(beam_per_sent_bleus):.4f}")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    print("\n[10] Generating plots ...")
    plot_training_curve(history, OUT_CURVE)
    plot_bleu_histogram(beam_per_sent_bleus, OUT_BLEU_HIST)
    heatmap_source, heatmap_target, heatmap_weights = translate_with_attention(
        'তুমি কি আমাকে সাহায্য করতে পারবে?',
        encoder_model,
        decoder_model,
        ben_tok,
        eng_tok,
        max_dec_len=max_dec
    )
    plot_attention_heatmap(
        heatmap_source,
        heatmap_target,
        heatmap_weights,
        OUT_ATTENTION,
        fm.FontProperties(family=font_name) if font_name else None
    )

    append_experiment_log(
        train_count=len(ben_train),
        val_count=len(ben_val),
        test_count=len(ben_test),
        enc_vocab=enc_vocab,
        dec_vocab=dec_vocab,
        greedy_bleu=greedy_bleu_score,
        greedy_mean=float(np.mean(greedy_per_sent_bleus)),
        greedy_median=float(np.median(greedy_per_sent_bleus)),
        beam_bleu=beam_bleu_score,
        beam_mean=float(np.mean(beam_per_sent_bleus)),
        beam_median=float(np.median(beam_per_sent_bleus)),
    )
    print(f"  Appended experiment details -> {EXPERIMENT_LOG}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print(f"  Training pairs              : {len(ben_train):,}")
    print(f"  Test pairs                  : {len(ben_test):,}")
    print(f"  Bengali vocabulary (enc)    : {enc_vocab:,}")
    print(f"  English vocabulary (dec)    : {dec_vocab:,}")
    print(f"  Greedy Corpus BLEU-4 (test) : {greedy_bleu_score:.4f}")
    print(f"  Beam Corpus BLEU-4 (test)   : {beam_bleu_score:.4f}")
    print(f"  Split mode                  : {SPLIT_MODE}")
    print(f"  Encoder LSTM units          : {LSTM_UNITS}")
    print(f"  Decoder LSTM units          : {LSTM_UNITS}")
    print(f"  Embedding dimension         : {EMBEDDING_DIM}")
    print(f"  Epochs                      : {EPOCHS}")
    print(f"  Beam width                  : {BEAM_WIDTH}")
    print("=" * 70)
    print("\nArchitecture justification:")
    print("  Bidirectional encoder + additive attention: the plain seq2seq")
    print("  model was generating fluent but semantically wrong English.")
    print("  Attention lets the decoder focus on relevant Bengali words")
    print("  during each output step, which is more suitable for translation.")
    print("  Dropout, gradient clipping, learning-rate reduction, and early")
    print("  stopping are used to control overfitting on this small corpus.")
    print("  256 decoder units and 128-dim embeddings remain compact enough")
    print("  for the dataset while still modeling short multi-word inputs.")
    print("=" * 70)
