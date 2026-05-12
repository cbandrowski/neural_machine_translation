# =============================================================================
#  TRANSLATION WITH SEQ2SEQ LSTM — English → Bengali
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
#  that reliably captures Bengali morphological variety on this corpus;
#  64-dim embeddings match the small vocabulary sizes.
#
#  DATA:    ben-eng/ben.txt   (col 0 = English, col 1 = Bengali)
#  SPLIT:   70 % train, 30 % test
#
#  HOW TO RUN:
#    python seq2seq_english.py
#
#  OUTPUT (saved to same folder as this script):
#    word_based_seq2seq_eng_to_ben_training_curve.png   — loss / accuracy over epochs
#    word_based_seq2seq_eng_to_ben_bleu_histogram.png   — per-sentence BLEU distribution
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

import tensorflow as tf
from tensorflow.keras.models              import Model
from tensorflow.keras.layers              import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection              import train_test_split
from nltk.translate.bleu_score            import corpus_bleu, SmoothingFunction

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
#  PATHS
# =============================================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TSV_PATH        = os.path.join(BASE_DIR, "ben-eng", "ben.txt")
OUT_DIR         = os.path.join(BASE_DIR, "output", "word_based")
OUT_CURVE       = os.path.join(OUT_DIR, "word_based_seq2seq_eng_to_ben_training_curve.png")
OUT_BLEU_HIST   = os.path.join(OUT_DIR, "word_based_seq2seq_eng_to_ben_bleu_histogram.png")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
#  HYPER-PARAMETERS  (minimal / justified above)
# =============================================================================

EMBEDDING_DIM = 64
LSTM_UNITS    = 256
BATCH_SIZE    = 64
EPOCHS        = 50
TEST_SIZE     = 0.30
MAX_VOCAB_ENG = 5000   # cap rare English words
MAX_VOCAB_BEN = 8000   # Bengali morphology inflates vocabulary

START_TOKEN = '<start>'
END_TOKEN   = '<end>'
UNK_TOKEN   = '<unk>'

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
    """Lower-case; keep only a-z and spaces."""
    text = re.sub(r"[^a-zA-Z\s]", ' ', text)
    return ' '.join(text.lower().split())


def clean_bengali(text):
    """Keep Bengali Unicode block (U+0980-U+09FF) + danda + spaces."""
    text = re.sub(r"[^ঀ-৿।॥\s]", ' ', text)
    return ' '.join(text.split())


def preprocess(pairs):
    eng_sentences, ben_sentences = [], []
    for eng, ben in pairs:
        e = clean_english(eng)
        b = clean_bengali(ben)
        if e and b:
            eng_sentences.append(e)
            # Decoder input/target uses explicit boundary tokens
            ben_sentences.append(f"{START_TOKEN} {b} {END_TOKEN}")
    return eng_sentences, ben_sentences


# =============================================================================
#  3. TOKENISATION & PADDING
# =============================================================================

def build_tokenizer(sentences, num_words):
    tok = Tokenizer(num_words=num_words, oov_token=UNK_TOKEN, filters='')
    tok.fit_on_texts(sentences)
    return tok


def encode_and_pad(tokenizer, sentences, maxlen):
    seqs = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(seqs, maxlen=maxlen, padding='post'), maxlen


# =============================================================================
#  4. MODEL CONSTRUCTION
# =============================================================================

def build_model(enc_vocab, dec_vocab):
    """
    Returns the training model and the two inference sub-models.
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
    enc_in         = Input(shape=(None,), name='encoder_input')
    enc_emb        = enc_emb_layer(enc_in)
    _, state_h, state_c = enc_lstm_layer(enc_emb)
    enc_states     = [state_h, state_c]

    dec_in         = Input(shape=(None,), name='decoder_input')
    dec_emb        = dec_emb_layer(dec_in)
    dec_out_seq, _, _ = dec_lstm_layer(dec_emb, initial_state=enc_states)
    dec_out        = dec_dense(dec_out_seq)

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

    dec_single_emb          = dec_emb_layer(dec_in)
    dec_single_out, h, c    = dec_lstm_layer(dec_single_emb,
                                             initial_state=dec_states_in)
    dec_single_dense        = dec_dense(dec_single_out)

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
              eng_tok, ben_tok, max_dec_len=20):
    """Greedy decode one English sentence → Bengali string.

    Uses direct model __call__ instead of model.predict() to avoid
    per-call Python-TF overhead, which would freeze on large test sets.
    """
    # encode
    seq = eng_tok.texts_to_sequences([clean_english(sentence)])
    seq = tf.constant(
        pad_sequences(seq, maxlen=encoder_model.input_shape[1], padding='post')
    )
    states = encoder_model(seq, training=False)
    # encoder returns [state_h, state_c]
    states = list(states) if not isinstance(states, list) else states

    # start token
    start_idx = ben_tok.word_index.get(START_TOKEN, 1)
    end_idx   = ben_tok.word_index.get(END_TOKEN, 2)

    target_seq     = tf.constant([[start_idx]])
    decoded_tokens = []

    for _ in range(max_dec_len):
        output, h, c = decoder_model([target_seq] + states, training=False)
        token_idx = int(tf.argmax(output[0, -1, :]))
        if token_idx == end_idx or token_idx == 0:
            break
        word = ben_tok.index_word.get(token_idx, '')
        if word and word not in (START_TOKEN, END_TOKEN, UNK_TOKEN):
            decoded_tokens.append(word)
        target_seq = tf.constant([[token_idx]])
        states     = [h, c]

    return ' '.join(decoded_tokens)


# =============================================================================
#  7. BLEU SCORE
# =============================================================================

def compute_bleu(pairs_test, encoder_model, decoder_model,
                 eng_tok, ben_tok, max_dec_len=20):
    """Corpus-level BLEU-4 on the test set."""
    references, hypotheses = [], []
    total = len(pairs_test)

    for i, (eng, ben_ref_full) in enumerate(pairs_test):
        if i % 100 == 0:
            print(f"    Translating {i}/{total} ...", flush=True)

        # strip boundary tokens from reference
        ref_clean = re.sub(
            rf'\s*{re.escape(START_TOKEN)}\s*|\s*{re.escape(END_TOKEN)}\s*',
            ' ', ben_ref_full
        ).strip().split()

        hyp = translate(eng, encoder_model, decoder_model,
                        eng_tok, ben_tok, max_dec_len).split()

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

    ax1.plot(history.history['loss'],     label='Train loss')
    ax1.plot(history.history['val_loss'], label='Val loss')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['accuracy'],     label='Train accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val accuracy')
    ax2.set_title('Training & Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle('Word-Based Plain Seq2Seq LSTM - English -> Bengali', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_bleu_histogram(per_sentence_bleus, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_sentence_bleus, bins=30, color='#4e79a7', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(per_sentence_bleus), color='#e15759', linestyle='--',
               linewidth=1.8, label=f'Mean = {np.mean(per_sentence_bleus):.4f}')
    ax.set_title('Word-Based Plain BLEU Distribution (Test Set)', fontweight='bold')
    ax.set_xlabel('Sentence BLEU'); ax.set_ylabel('Count')
    ax.legend(); ax.grid(True, alpha=0.3)
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

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1] Loading data from {TSV_PATH}")
    raw_pairs = load_pairs(TSV_PATH)
    print(f"    {len(raw_pairs):,} sentence pairs loaded")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2] Preprocessing ...")
    eng_sents, ben_sents = preprocess(raw_pairs)
    print(f"    {len(eng_sents):,} valid pairs after cleaning")

    # ── 3. Train / test split (70 / 30) ──────────────────────────────────────
    print(f"\n[3] Splitting data (test = {int(TEST_SIZE*100)}%) ...")
    (eng_train, eng_test,
     ben_train, ben_test) = train_test_split(
        eng_sents, ben_sents,
        test_size=TEST_SIZE, random_state=42
    )
    print(f"    Train: {len(eng_train):,}  |  Test: {len(eng_test):,}")

    # ── 4. Tokenise ───────────────────────────────────────────────────────────
    print("\n[4] Building tokenisers ...")
    eng_tok = build_tokenizer(eng_train, MAX_VOCAB_ENG)
    ben_tok = build_tokenizer(ben_train, MAX_VOCAB_BEN)

    enc_vocab = min(len(eng_tok.word_index) + 1, MAX_VOCAB_ENG)
    dec_vocab = min(len(ben_tok.word_index) + 1, MAX_VOCAB_BEN)
    print(f"    English vocab : {enc_vocab:,} tokens")
    print(f"    Bengali vocab : {dec_vocab:,} tokens")

    # ── 5. Encode & pad ───────────────────────────────────────────────────────
    print("\n[5] Encoding and padding sequences ...")
    max_enc = max(len(s.split()) for s in eng_train)
    max_dec = max(len(s.split()) for s in ben_train)
    print(f"    Max encoder length : {max_enc}")
    print(f"    Max decoder length : {max_dec}")

    enc_train_seq = pad_sequences(
        eng_tok.texts_to_sequences(eng_train), maxlen=max_enc, padding='post')
    dec_train_seq = pad_sequences(
        ben_tok.texts_to_sequences(ben_train), maxlen=max_dec, padding='post')

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
    ]

    print("\n[8] Example translations (greedy decode):")
    print(f"    {'English Input':<35}  {'Predicted Bengali'}")
    print(f"    {'-'*35}  {'-'*40}")
    for sent in EXAMPLE_SENTENCES:
        prediction = translate(sent, encoder_model, decoder_model,
                               eng_tok, ben_tok, max_dec_len=max_dec)
        print(f"    {sent:<35}  {prediction if prediction else '(empty)'}")

    # ── 9. BLEU score ─────────────────────────────────────────────────────────
    print("\n[9] Computing BLEU score on test set ...")
    test_pairs = list(zip(eng_test, ben_test))
    corpus_bleu_score, references, hypotheses = compute_bleu(
        test_pairs, encoder_model, decoder_model,
        eng_tok, ben_tok, max_dec_len=max_dec
    )

    # per-sentence BLEU for histogram
    smooth = SmoothingFunction().method1
    per_sent_bleus = []
    for ref, hyp in zip(references, hypotheses):
        from nltk.translate.bleu_score import sentence_bleu
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
    print(f"  Training pairs          : {len(eng_train):,}")
    print(f"  Test pairs              : {len(eng_test):,}")
    print(f"  English vocabulary      : {enc_vocab:,}")
    print(f"  Bengali vocabulary      : {dec_vocab:,}")
    print(f"  Corpus BLEU-4 (test)    : {corpus_bleu_score:.4f}")
    print(f"  Encoder LSTM units      : {LSTM_UNITS}")
    print(f"  Decoder LSTM units      : {LSTM_UNITS}")
    print(f"  Embedding dimension     : {EMBEDDING_DIM}")
    print(f"  Epochs                  : {EPOCHS}")
    print("=" * 70)
    print("\nArchitecture justification:")
    print("  Single-layer LSTM encoder/decoder: the corpus has only ~5 000")
    print("  training pairs with short sentences (avg < 8 tokens). Stacking")
    print("  LSTM layers would add parameters without enough data to train")
    print("  them, risking overfitting. One layer per side is sufficient to")
    print("  capture the sentence-level context needed for this dataset.")
    print("  256 units: smallest power-of-two that handles Bengali morphology")
    print("  (rich suffixing); 64-dim embeddings match vocabulary sizes < 5K.")
    print("=" * 70)
