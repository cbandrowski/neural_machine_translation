# =============================================================================
#  TRANSLATION WITH SEQ2SEQ LSTM — Bengali → English
#
#  Task 2.3: Sequence-to-Sequence encoder-decoder using LSTMs.
#
#  Architecture:
#    - Bidirectional LSTM encoder
#    - LSTM decoder with additive attention and teacher forcing
#    - Separate embedding layers for each language (dim=64)
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
#    ben_seq2seq_training_curve.png   — loss / accuracy over epochs
#    ben_seq2seq_bleu_histogram.png   — per-sentence BLEU distribution
# =============================================================================

import os, re, warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
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
VAL_SIZE      = 0.10
DROPOUT       = 0.25
RECURRENT_DROPOUT = 0.10
LEARNING_RATE = 1e-3
CLIPNORM      = 1.0
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
    """Lower-case English while preserving contractions like don't / i'm."""
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    text = re.sub(r"\s+'\s+", ' ', text)
    return ' '.join(text.lower().split())


def strip_boundary_tokens(text):
    return re.sub(
        rf'\s*{re.escape(START_TOKEN)}\s*|\s*{re.escape(END_TOKEN)}\s*',
        ' ',
        text,
    ).strip()


def choose_canonical_target(targets):
    counts = Counter(targets)
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], len(item[0].split()), len(item[0]), item[0]),
    )
    return ranked[0][0]


def preprocess(pairs):
    grouped = defaultdict(list)
    for ben, eng in pairs:
        b = clean_bengali(ben)
        e = clean_english(eng)
        if b and e:
            grouped[b].append(e)

    ben_sentences, eng_sentences, eng_references = [], [], []
    for ben, eng_variants in grouped.items():
        canonical_eng = choose_canonical_target(eng_variants)
        unique_refs = sorted(
            set(eng_variants),
            key=lambda text: (len(text.split()), len(text), text),
        )
        ben_sentences.append(ben)
        eng_sentences.append(f"{START_TOKEN} {canonical_eng} {END_TOKEN}")
        eng_references.append([f"{START_TOKEN} {ref} {END_TOKEN}" for ref in unique_refs])
    return ben_sentences, eng_sentences, eng_references


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
        total_len = len(src.split()) + len(strip_boundary_tokens(tgt).split())
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


def masked_sequence_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


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
        metrics=[masked_sequence_accuracy]
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
        output, h, c = decoder_model([target_seq, enc_outputs] + states, training=False)
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

def compute_bleu(test_examples, encoder_model, decoder_model,
                 ben_tok, eng_tok, max_dec_len=20):
    """Corpus-level multi-reference BLEU-4 on the test set."""
    references, hypotheses = [], []
    total = len(test_examples)

    for i, (ben, _, eng_ref_full_list) in enumerate(test_examples):
        if i % 100 == 0:
            print(f"    Translating {i}/{total} ...", flush=True)

        ref_clean = [strip_boundary_tokens(ref).split() for ref in eng_ref_full_list]

        hyp = translate(ben, encoder_model, decoder_model,
                        ben_tok, eng_tok, max_dec_len).split()

        references.append(ref_clean)
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

    ax2.plot(history.history['masked_sequence_accuracy'],
             label='Train masked accuracy', color='#e15759')
    ax2.plot(history.history['val_masked_sequence_accuracy'],
             label='Val masked accuracy',   color='#f28e2b',
             linestyle='--')
    ax2.set_title('Training & Validation Masked Accuracy', fontweight='bold')
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
    print("\n[2] Preprocessing and grouping by Bengali source ...")
    ben_sents, eng_sents, eng_refs = preprocess(raw_pairs)
    print(f"    {len(ben_sents):,} unique Bengali source sentences")
    multi_ref_count = sum(1 for refs in eng_refs if len(refs) > 1)
    print(f"    {multi_ref_count:,} sources have multiple English references")

    # ── 3. Train / val / test split ──────────────────────────────────────────

    print(f"    Splitting data (test = {int(TEST_SIZE*100)}%, val = {int(VAL_SIZE*100)}% of train) ...")
    all_buckets = length_buckets(ben_sents, eng_sents)
    (ben_train_full, ben_test,
     eng_train_full, eng_test,
     refs_train_full, refs_test,
     train_buckets, _) = train_test_split(
        ben_sents, eng_sents, eng_refs, all_buckets,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=all_buckets,
        shuffle=True
    )

    (ben_train, ben_val,
     eng_train, eng_val,
     refs_train, refs_val) = train_test_split(
        ben_train_full, eng_train_full, refs_train_full,
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
                                 verbose=1)

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
    test_examples = list(zip(ben_test, eng_test, refs_test))
    corpus_bleu_score, references, hypotheses = compute_bleu(
        test_examples, encoder_model, decoder_model,
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
    print(f"  Unique Bengali sources      : {len(ben_sents):,}")
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
    print("  Bidirectional encoder + additive attention: the plain seq2seq")
    print("  model was generating fluent but semantically wrong English.")
    print("  Attention lets the decoder focus on relevant Bengali words")
    print("  during each output step, which is more suitable for translation.")
    print("  Dropout, gradient clipping, learning-rate reduction, and early")
    print("  stopping are used to control overfitting on this small corpus.")
    print("  256 decoder units and 64-dim embeddings remain compact enough")
    print("  for the dataset while still modeling short multi-word inputs.")
    print("=" * 70)
