# =============================================================================
#  TRANSLATION WITH SEQ2SEQ LSTM — English → Bengali
#
#  Task 2.3: Sequence-to-Sequence encoder-decoder using LSTMs.
#
#  Architecture:
#    - Bidirectional LSTM encoder
#    - LSTM decoder with additive attention and teacher forcing
#    - Separate embedding layers for each language (dim=64)
#
#  Notes on the dataset:
#    - Many English sentences map to multiple valid Bengali translations.
#    - We group examples by English source, train on one canonical target per
#      source sentence, and evaluate against all available Bengali references.
#
#  DATA:    ben-eng/ben.txt   (col 0 = English, col 1 = Bengali)
#  SPLIT:   Source-level 70 % train, 7 % val, 30 % test
#
#  HOW TO RUN:
#    python seq2seq_english.py
#
#  OUTPUT (saved to same folder as this script):
#    seq2seq_training_curve.png   — loss / masked accuracy over epochs
#    seq2seq_bleu_histogram.png   — per-sentence BLEU distribution
# =============================================================================

import os
import re
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Bidirectional, AdditiveAttention, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(BASE_DIR, "ben-eng", "ben.txt")
OUT_CURVE = os.path.join(BASE_DIR, "seq2seq_training_curve.png")
OUT_BLEU_HIST = os.path.join(BASE_DIR, "seq2seq_bleu_histogram.png")

# =============================================================================
#  HYPER-PARAMETERS
# =============================================================================

EMBEDDING_DIM = 64
LSTM_UNITS = 256
BATCH_SIZE = 64
EPOCHS = 50
TEST_SIZE = 0.30
VAL_SIZE = 0.10
DROPOUT = 0.25
RECURRENT_DROPOUT = 0.10
LEARNING_RATE = 1e-3
CLIPNORM = 1.0
MAX_VOCAB_ENG = 5000
MAX_VOCAB_BEN = 8000

START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'

# =============================================================================
#  1. DATA LOADING
# =============================================================================

def load_pairs(tsv_path):
    """Read (English, Bengali) pairs; skip malformed / empty lines."""
    pairs = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng = parts[0].strip()
                ben = parts[1].strip()
                if eng and ben:
                    pairs.append((eng, ben))
    return pairs


# =============================================================================
#  2. PREPROCESSING
# =============================================================================

def clean_english(text):
    """Lower-case English while preserving contractions like don't / i'm."""
    text = text.replace("’", "'")
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    text = re.sub(r"\s+'\s+", ' ', text)
    return ' '.join(text.lower().split())


def clean_bengali(text):
    """Keep Bengali Unicode block (U+0980-U+09FF) + danda + spaces."""
    text = re.sub(r"[^ঀ-৿।॥\s]", ' ', text)
    return ' '.join(text.split())


def strip_boundary_tokens(text):
    return re.sub(
        rf'\s*{re.escape(START_TOKEN)}\s*|\s*{re.escape(END_TOKEN)}\s*',
        ' ',
        text,
    ).strip()


def choose_canonical_target(targets):
    """
    Pick one stable training target per source sentence.
    Frequency wins first; then prefer the shorter variant to reduce noise.
    """
    counts = Counter(targets)
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], len(item[0].split()), len(item[0]), item[0]),
    )
    return ranked[0][0]


def prepare_examples(pairs):
    """
    Group by cleaned English source.

    Returns a list of:
      (source_english, canonical_target_with_tags, all_target_refs_with_tags)
    """
    grouped = defaultdict(list)
    for eng, ben in pairs:
        e = clean_english(eng)
        b = clean_bengali(ben)
        if e and b:
            grouped[e].append(b)

    examples = []
    for eng, ben_variants in grouped.items():
        canonical_ben = choose_canonical_target(ben_variants)
        unique_refs = sorted(
            set(ben_variants),
            key=lambda text: (len(text.split()), len(text), text),
        )
        canonical_target = f"{START_TOKEN} {canonical_ben} {END_TOKEN}"
        all_refs = [f"{START_TOKEN} {ref} {END_TOKEN}" for ref in unique_refs]
        examples.append((eng, canonical_target, all_refs))
    return examples


def length_buckets(source_sentences, target_sentences):
    """Coarse buckets to keep train/val/test splits balanced by length."""
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


# =============================================================================
#  4. TRAINING HELPERS
# =============================================================================

def masked_sequence_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def make_decoder_targets(dec_sequences):
    """Shift decoder sequences by one step to create targets."""
    dec_in = dec_sequences[:, :-1]
    dec_tgt = dec_sequences[:, 1:]
    dec_tgt = np.expand_dims(dec_tgt, -1)
    return dec_in, dec_tgt


def make_dataset(enc_seq, dec_in_seq, dec_tgt_seq, batch_size, shuffle=False, seed=SEED):
    dataset = tf.data.Dataset.from_tensor_slices(((enc_seq, dec_in_seq), dec_tgt_seq))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(enc_seq),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =============================================================================
#  5. MODEL CONSTRUCTION
# =============================================================================

def build_model(enc_vocab, dec_vocab):
    """
    Returns the training model and the two inference sub-models.
    Encoder reads English; decoder generates Bengali.
    """
    encoder_half_units = LSTM_UNITS // 2

    enc_emb_layer = Embedding(enc_vocab, EMBEDDING_DIM, name='enc_embedding', mask_zero=True)
    enc_lstm_layer = Bidirectional(
        LSTM(
            encoder_half_units,
            return_sequences=True,
            return_state=True,
            dropout=DROPOUT,
            recurrent_dropout=RECURRENT_DROPOUT,
        ),
        name='enc_bi_lstm',
    )

    dec_emb_layer = Embedding(dec_vocab, EMBEDDING_DIM, name='dec_embedding', mask_zero=True)
    dec_lstm_layer = LSTM(
        LSTM_UNITS,
        return_sequences=True,
        return_state=True,
        dropout=DROPOUT,
        recurrent_dropout=RECURRENT_DROPOUT,
        name='dec_lstm',
    )
    attention_layer = AdditiveAttention(name='attention')
    concat_layer = Concatenate(axis=-1, name='context_concat')
    proj_layer = Dense(LSTM_UNITS, activation='tanh', name='context_projection')
    dec_dense = Dense(dec_vocab, activation='softmax', name='dec_output')

    enc_in = Input(shape=(None,), name='encoder_input')
    enc_emb = enc_emb_layer(enc_in)
    enc_out_seq, fh, fc, bh, bc = enc_lstm_layer(enc_emb)
    state_h = Concatenate(name='enc_state_h')([fh, bh])
    state_c = Concatenate(name='enc_state_c')([fc, bc])
    enc_states = [state_h, state_c]

    dec_in = Input(shape=(None,), name='decoder_input')
    dec_emb = dec_emb_layer(dec_in)
    dec_out_seq, _, _ = dec_lstm_layer(dec_emb, initial_state=enc_states)
    context_seq = attention_layer([dec_out_seq, enc_out_seq])
    dec_context = concat_layer([dec_out_seq, context_seq])
    dec_context = proj_layer(dec_context)
    dec_out = dec_dense(dec_context)

    training_model = Model([enc_in, dec_in], dec_out)
    training_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss='sparse_categorical_crossentropy',
        metrics=[masked_sequence_accuracy],
    )

    encoder_model = Model(enc_in, [enc_out_seq] + enc_states)

    dec_state_h_in = Input(shape=(LSTM_UNITS,), name='dec_state_h_in')
    dec_state_c_in = Input(shape=(LSTM_UNITS,), name='dec_state_c_in')
    enc_out_in = Input(shape=(None, LSTM_UNITS), name='enc_out_in')
    dec_states_in = [dec_state_h_in, dec_state_c_in]

    dec_single_emb = dec_emb_layer(dec_in)
    dec_single_out, h, c = dec_lstm_layer(dec_single_emb, initial_state=dec_states_in)
    context_single = attention_layer([dec_single_out, enc_out_in])
    dec_single_context = concat_layer([dec_single_out, context_single])
    dec_single_context = proj_layer(dec_single_context)
    dec_single_dense = dec_dense(dec_single_context)

    decoder_model = Model(
        [dec_in, enc_out_in] + dec_states_in,
        [dec_single_dense, h, c],
    )

    return training_model, encoder_model, decoder_model


# =============================================================================
#  6. GREEDY INFERENCE
# =============================================================================

def translate(sentence, encoder_model, decoder_model, eng_tok, ben_tok, max_dec_len=20):
    """Greedy decode one English sentence → Bengali string."""
    seq = eng_tok.texts_to_sequences([clean_english(sentence)])
    seq = tf.constant(
        pad_sequences(seq, maxlen=encoder_model.input_shape[1], padding='post')
    )
    enc_outputs, state_h, state_c = encoder_model(seq, training=False)
    states = [state_h, state_c]

    start_idx = ben_tok.word_index.get(START_TOKEN, 1)
    end_idx = ben_tok.word_index.get(END_TOKEN, 2)

    target_seq = tf.constant([[start_idx]])
    decoded_tokens = []

    for _ in range(max_dec_len):
        output, h, c = decoder_model([target_seq, enc_outputs] + states, training=False)
        token_idx = int(tf.argmax(output[0, -1, :]))
        if token_idx == end_idx or token_idx == 0:
            break
        word = ben_tok.index_word.get(token_idx, '')
        if word and word not in (START_TOKEN, END_TOKEN, UNK_TOKEN):
            decoded_tokens.append(word)
        target_seq = tf.constant([[token_idx]])
        states = [h, c]

    return ' '.join(decoded_tokens)


# =============================================================================
#  7. BLEU SCORE
# =============================================================================

def compute_bleu(test_examples, encoder_model, decoder_model, eng_tok, ben_tok, max_dec_len=20):
    """Corpus-level multi-reference BLEU-4 on the test set."""
    references, hypotheses = [], []
    total = len(test_examples)

    for i, (eng, _, ben_ref_full_list) in enumerate(test_examples):
        if i % 100 == 0:
            print(f"    Translating {i}/{total} ...", flush=True)

        ref_clean = [strip_boundary_tokens(ref).split() for ref in ben_ref_full_list]
        hyp = translate(eng, encoder_model, decoder_model, eng_tok, ben_tok, max_dec_len).split()

        references.append(ref_clean)
        hypotheses.append(hyp)

    bleu = corpus_bleu(
        references,
        hypotheses,
        smoothing_function=SmoothingFunction().method1,
    )
    return bleu, references, hypotheses


# =============================================================================
#  8. PLOTS
# =============================================================================

def plot_training_curve(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train loss', color='#4e79a7')
    ax1.plot(history.history['val_loss'], label='Val loss', color='#f28e2b', linestyle='--')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(
        history.history['masked_sequence_accuracy'],
        label='Train masked accuracy',
        color='#4e79a7',
    )
    ax2.plot(
        history.history['val_masked_sequence_accuracy'],
        label='Val masked accuracy',
        color='#f28e2b',
        linestyle='--',
    )
    ax2.set_title('Training & Validation Masked Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Seq2Seq LSTM  —  English → Bengali', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_bleu_histogram(per_sentence_bleus, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_sentence_bleus, bins=30, color='#4e79a7', edgecolor='white', alpha=0.85)
    ax.axvline(
        np.mean(per_sentence_bleus),
        color='#e15759',
        linestyle='--',
        linewidth=1.8,
        label=f'Mean = {np.mean(per_sentence_bleus):.4f}',
    )
    ax.set_title('Per-Sentence BLEU Distribution (Test Set)', fontweight='bold')
    ax.set_xlabel('Sentence BLEU')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
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
    print("  SEQ2SEQ LSTM  —  English → Bengali Translation")
    print("=" * 70)

    print(f"\n[1] Loading data from {TSV_PATH}")
    raw_pairs = load_pairs(TSV_PATH)
    print(f"    {len(raw_pairs):,} sentence pairs loaded")

    print("\n[2] Grouping source sentences and preprocessing ...")
    examples = prepare_examples(raw_pairs)
    eng_sents = [src for src, _, _ in examples]
    ben_targets = [tgt for _, tgt, _ in examples]
    print(f"    {len(examples):,} unique English source sentences")

    multi_ref_count = sum(1 for _, _, refs in examples if len(refs) > 1)
    print(f"    {multi_ref_count:,} sources have multiple Bengali references")

    print(f"\n[3] Splitting data (test = {int(TEST_SIZE*100)}%, val = {int(VAL_SIZE*100)}% of train) ...")
    all_buckets = length_buckets(eng_sents, ben_targets)
    (
        eng_train_full,
        eng_test,
        ben_train_full,
        ben_test,
        refs_train_full,
        refs_test,
        train_buckets,
        _,
    ) = train_test_split(
        eng_sents,
        ben_targets,
        [refs for _, _, refs in examples],
        all_buckets,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=all_buckets,
        shuffle=True,
    )

    (
        eng_train,
        eng_val,
        ben_train,
        ben_val,
        refs_train,
        refs_val,
    ) = train_test_split(
        eng_train_full,
        ben_train_full,
        refs_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=train_buckets,
        shuffle=True,
    )
    print(f"    Train: {len(eng_train):,}  |  Val: {len(eng_val):,}  |  Test: {len(eng_test):,}")

    print("\n[4] Building tokenisers ...")
    eng_tok = build_tokenizer(eng_train, MAX_VOCAB_ENG)
    ben_tok = build_tokenizer(ben_train, MAX_VOCAB_BEN)

    enc_vocab = min(len(eng_tok.word_index) + 1, MAX_VOCAB_ENG)
    dec_vocab = min(len(ben_tok.word_index) + 1, MAX_VOCAB_BEN)
    print(f"    English vocab (encoder) : {enc_vocab:,} tokens")
    print(f"    Bengali vocab (decoder) : {dec_vocab:,} tokens")

    print("\n[5] Encoding and padding sequences ...")
    max_enc = max(len(s.split()) for s in eng_train)
    max_dec = max(len(s.split()) for s in ben_train)
    print(f"    Max encoder length (English) : {max_enc}")
    print(f"    Max decoder length (Bengali) : {max_dec}")

    enc_train_seq = pad_sequences(
        eng_tok.texts_to_sequences(eng_train), maxlen=max_enc, padding='post'
    )
    dec_train_seq = pad_sequences(
        ben_tok.texts_to_sequences(ben_train), maxlen=max_dec, padding='post'
    )
    enc_val_seq = pad_sequences(
        eng_tok.texts_to_sequences(eng_val), maxlen=max_enc, padding='post'
    )
    dec_val_seq = pad_sequences(
        ben_tok.texts_to_sequences(ben_val), maxlen=max_dec, padding='post'
    )

    dec_in_train, dec_tgt_train = make_decoder_targets(dec_train_seq)
    dec_in_val, dec_tgt_val = make_decoder_targets(dec_val_seq)

    print("\n[6] Building seq2seq model ...")
    training_model, encoder_model, decoder_model = build_model(enc_vocab, dec_vocab)
    training_model.summary()

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
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    history = training_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    EXAMPLE_SENTENCES = [
        "Go.",
        "Help!",
        "I am happy.",
        "What is your name?",
        "She is a teacher.",
        "He does not know.",
        "We are friends.",
        "I want to go home.",
        "Can you help me?",
        "Thank you very much.",
        "Don't talk.",
    ]

    print("\n[8] Example translations (greedy decode):")
    print(f"    {'English Input':<35}  Predicted Bengali")
    print(f"    {'-'*35}  {'-'*40}")
    for sent in EXAMPLE_SENTENCES:
        prediction = translate(sent, encoder_model, decoder_model, eng_tok, ben_tok, max_dec_len=max_dec)
        print(f"    {sent:<35}  {prediction if prediction else '(empty)'}")

    print("\n[9] Computing BLEU score on test set ...")
    test_examples = list(zip(eng_test, ben_test, refs_test))
    corpus_bleu_score, references, hypotheses = compute_bleu(
        test_examples,
        encoder_model,
        decoder_model,
        eng_tok,
        ben_tok,
        max_dec_len=max_dec,
    )

    smooth = SmoothingFunction().method1
    per_sent_bleus = []
    for refs, hyp in zip(references, hypotheses):
        per_sent_bleus.append(sentence_bleu(refs, hyp, smoothing_function=smooth))

    print(f"\n    Corpus BLEU-4 on test set : {corpus_bleu_score:.4f}")
    print(f"    Mean sentence BLEU        : {np.mean(per_sent_bleus):.4f}")
    print(f"    Median sentence BLEU      : {np.median(per_sent_bleus):.4f}")

    print("\n[10] Generating plots ...")
    plot_training_curve(history, OUT_CURVE)
    plot_bleu_histogram(per_sent_bleus, OUT_BLEU_HIST)

    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print(f"  Unique English sources      : {len(examples):,}")
    print(f"  Train sources               : {len(eng_train):,}")
    print(f"  Test sources                : {len(eng_test):,}")
    print(f"  English vocabulary (enc)    : {enc_vocab:,}")
    print(f"  Bengali vocabulary (dec)    : {dec_vocab:,}")
    print(f"  Corpus BLEU-4 (test)        : {corpus_bleu_score:.4f}")
    print(f"  Encoder LSTM units          : {LSTM_UNITS}")
    print(f"  Decoder LSTM units          : {LSTM_UNITS}")
    print(f"  Embedding dimension         : {EMBEDDING_DIM}")
    print(f"  Epochs                      : {EPOCHS}")
    print("=" * 70)
    print("\nArchitecture justification:")
    print("  Bidirectional encoder + additive attention: the English→Bengali")
    print("  corpus contains short sentences with many overlapping lexical")
    print("  patterns and multiple valid Bengali renderings. Attention helps")
    print("  the decoder stay aligned to the relevant English tokens.")
    print("  The larger gain here is dataset handling rather than embeddings:")
    print("  contractions are preserved, training is done on unique English")
    print("  sources, and BLEU is computed against every known Bengali")
    print("  reference instead of one arbitrary held-out variant.")
    print("=" * 70)
