# =============================================================================
#  TRANSLATION WITH SEQ2SEQ LSTM — Bengali → English
#
#  Task 2.3: Sequence-to-Sequence encoder-decoder using LSTMs.
#
#  Architecture (simplest viable design):
#    - 1-layer LSTM encoder (256 units)
#    - 1-layer LSTM decoder (256 units) with teacher forcing
#    - Separate embedding layers for each language (dim=64)
#  Justification: Only ~5 000 training pairs with short sentences
#  (avg < 8 tokens).  A single layer per side avoids overfitting while
#  still learning useful alignments.  256 units is the smallest power-of-two
#  that handles Bengali morphological complexity; 64-dim embeddings match
#  the small vocabulary sizes.
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
#    ben_seq2seq_training_curve.png   — loss / accuracy over epochs
#    ben_seq2seq_bleu_histogram.png   — per-sentence BLEU distribution
# =============================================================================

import os, re, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import tensorflow as tf
from tensorflow.keras.models              import Model
from tensorflow.keras.layers              import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection              import train_test_split
from nltk.translate.bleu_score            import corpus_bleu, sentence_bleu, SmoothingFunction

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
#  PATHS
# =============================================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TSV_PATH        = os.path.join(BASE_DIR, "ben-eng", "ben.txt")
OUT_CURVE       = os.path.join(BASE_DIR, "ben_seq2seq_training_curve.png")
OUT_BLEU_HIST   = os.path.join(BASE_DIR, "ben_seq2seq_bleu_histogram.png")

# =============================================================================
#  HYPER-PARAMETERS  (minimal / justified above)
# =============================================================================

EMBEDDING_DIM = 64
LSTM_UNITS    = 256
BATCH_SIZE    = 64
EPOCHS        = 50
TEST_SIZE     = 0.30
MAX_VOCAB_BEN = 8000   # Bengali morphology inflates vocabulary
MAX_VOCAB_ENG = 5000   # cap rare English words

START_TOKEN = '<start>'
END_TOKEN   = '<end>'
UNK_TOKEN   = '<unk>'

# =============================================================================
#  BENGALI FONT SETUP
# =============================================================================

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
    Architecture: single-layer LSTM encoder + single-layer LSTM decoder.
    """
    # ── Shared layer objects (reused by inference models) ────────────────────
    enc_emb_layer  = Embedding(enc_vocab, EMBEDDING_DIM, name='enc_embedding',
                                mask_zero=True)
    enc_lstm_layer = LSTM(LSTM_UNITS, return_state=True, name='enc_lstm')

    dec_emb_layer  = Embedding(dec_vocab, EMBEDDING_DIM, name='dec_embedding',
                                mask_zero=True)
    dec_lstm_layer = LSTM(LSTM_UNITS, return_sequences=True, return_state=True,
                          name='dec_lstm')
    dec_dense      = Dense(dec_vocab, activation='softmax', name='dec_output')

    # ── Training model (teacher forcing) ─────────────────────────────────────
    enc_in               = Input(shape=(None,), name='encoder_input')
    enc_emb              = enc_emb_layer(enc_in)
    _, state_h, state_c  = enc_lstm_layer(enc_emb)
    enc_states           = [state_h, state_c]

    dec_in                    = Input(shape=(None,), name='decoder_input')
    dec_emb                   = dec_emb_layer(dec_in)
    dec_out_seq, _, _         = dec_lstm_layer(dec_emb, initial_state=enc_states)
    dec_out                   = dec_dense(dec_out_seq)

    training_model = Model([enc_in, dec_in], dec_out)
    training_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ── Inference encoder ─────────────────────────────────────────────────────
    encoder_model = Model(enc_in, enc_states)

    # ── Inference decoder (one step at a time) ────────────────────────────────
    dec_state_h_in = Input(shape=(LSTM_UNITS,), name='dec_state_h_in')
    dec_state_c_in = Input(shape=(LSTM_UNITS,), name='dec_state_c_in')
    dec_states_in  = [dec_state_h_in, dec_state_c_in]

    dec_single_emb       = dec_emb_layer(dec_in)
    dec_single_out, h, c = dec_lstm_layer(dec_single_emb,
                                          initial_state=dec_states_in)
    dec_single_dense     = dec_dense(dec_single_out)

    decoder_model = Model(
        [dec_in] + dec_states_in,
        [dec_single_dense, h, c]
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
    states = encoder_model(seq, training=False)
    # encoder returns [state_h, state_c]
    states = list(states) if not isinstance(states, list) else states

    # start token
    start_idx = eng_tok.word_index.get(START_TOKEN, 1)
    end_idx   = eng_tok.word_index.get(END_TOKEN, 2)

    target_seq     = tf.constant([[start_idx]])
    decoded_tokens = []

    for _ in range(max_dec_len):
        output, h, c = decoder_model([target_seq] + states, training=False)
        token_idx = int(tf.argmax(output[0, -1, :]))
        if token_idx == end_idx or token_idx == 0:
            break
        word = eng_tok.index_word.get(token_idx, '')
        if word and word not in (START_TOKEN, END_TOKEN, UNK_TOKEN):
            decoded_tokens.append(word)
        target_seq = tf.constant([[token_idx]])
        states     = [h, c]

    return ' '.join(decoded_tokens)


# =============================================================================
#  7. BLEU SCORE
# =============================================================================

def compute_bleu(pairs_test, encoder_model, decoder_model,
                 ben_tok, eng_tok, max_dec_len=20):
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

        hyp = translate(ben, encoder_model, decoder_model,
                        ben_tok, eng_tok, max_dec_len).split()

        references.append([ref_clean])
        hypotheses.append(hyp)

    bleu = corpus_bleu(references, hypotheses,
                       smoothing_function=SmoothingFunction().method1)
    return bleu, references, hypotheses


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

    # ── 3. Train / test split (70 / 30) ──────────────────────────────────────
    print(f"\n[3] Splitting data (test = {int(TEST_SIZE*100)}%) ...")
    (ben_train, ben_test,
     eng_train, eng_test) = train_test_split(
        ben_sents, eng_sents,
        test_size=TEST_SIZE, random_state=42
    )
    print(f"    Train: {len(ben_train):,}  |  Test: {len(ben_test):,}")

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

    dec_in_train, dec_tgt_train = make_decoder_targets(dec_train_seq)

    # ── 6. Build model ────────────────────────────────────────────────────────
    print("\n[6] Building seq2seq model ...")
    training_model, encoder_model, decoder_model = build_model(enc_vocab, dec_vocab)
    training_model.summary()

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n[7] Training for {EPOCHS} epochs (batch={BATCH_SIZE}) ...")
    history = training_model.fit(
        [enc_train_seq, dec_in_train],
        dec_tgt_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

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

    print("\n[8] Example translations (greedy decode):")
    print(f"    {'Bengali Input':<40}  {'Reference English':<28}  Predicted English")
    print(f"    {'-'*40}  {'-'*28}  {'-'*30}")
    for ben_sent, ref_eng in EXAMPLE_SENTENCES:
        prediction = translate(ben_sent, encoder_model, decoder_model,
                               ben_tok, eng_tok, max_dec_len=max_dec)
        print(f"    {ben_sent:<40}  {ref_eng:<28}  {prediction if prediction else '(empty)'}")

    # ── 9. BLEU score ─────────────────────────────────────────────────────────
    print("\n[9] Computing BLEU score on test set ...")
    test_pairs = list(zip(ben_test, eng_test))
    corpus_bleu_score, references, hypotheses = compute_bleu(
        test_pairs, encoder_model, decoder_model,
        ben_tok, eng_tok, max_dec_len=max_dec
    )

    # per-sentence BLEU for histogram
    smooth = SmoothingFunction().method1
    per_sent_bleus = []
    for ref, hyp in zip(references, hypotheses):
        s = sentence_bleu(ref, hyp, smoothing_function=smooth)
        per_sent_bleus.append(s)

    print(f"\n    Corpus BLEU-4 on test set : {corpus_bleu_score:.4f}")
    print(f"    Mean sentence BLEU        : {np.mean(per_sent_bleus):.4f}")
    print(f"    Median sentence BLEU      : {np.median(per_sent_bleus):.4f}")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    print("\n[10] Generating plots ...")
    plot_training_curve(history, OUT_CURVE)
    plot_bleu_histogram(per_sent_bleus, OUT_BLEU_HIST)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print(f"  Training pairs              : {len(ben_train):,}")
    print(f"  Test pairs                  : {len(ben_test):,}")
    print(f"  Bengali vocabulary (enc)    : {enc_vocab:,}")
    print(f"  English vocabulary (dec)    : {dec_vocab:,}")
    print(f"  Corpus BLEU-4 (test)        : {corpus_bleu_score:.4f}")
    print(f"  Encoder LSTM units          : {LSTM_UNITS}")
    print(f"  Decoder LSTM units          : {LSTM_UNITS}")
    print(f"  Embedding dimension         : {EMBEDDING_DIM}")
    print(f"  Epochs                      : {EPOCHS}")
    print("=" * 70)
    print("\nArchitecture justification:")
    print("  Single-layer LSTM encoder/decoder: the corpus has only ~5 000")
    print("  training pairs with short sentences (avg < 8 tokens). Stacking")
    print("  LSTM layers would add parameters without enough data to train")
    print("  them, risking overfitting. One layer per side is sufficient to")
    print("  capture the sentence-level context needed for this dataset.")
    print("  256 units: smallest power-of-two that handles Bengali morphology")
    print("  (rich suffixing); 64-dim embeddings match vocabulary sizes < 5K.")
    print("  Bengali → English is slightly easier than the reverse because")
    print("  English output has a smaller, more regular vocabulary.")
    print("=" * 70)
