# =============================================================================
#  TRANSLATION WITH SEQ2SEQ LSTM — Bengali → English (New Dataset Sample)
#
#  Task 2.3:
#    1. Build a seq2seq encoder-decoder model using LSTMs
#    2. Use a train-test split with about 30% for testing
#    3. Train using tokenized sequences and embedding layers
#    4. Assess with example translations and BLEU score
#    5. Keep the architecture as simple as possible and justify it
#
#  DATA:
#    Final Project DL/newDS/data/newds_subset.tsv
#
#  HOW TO RUN:
#    python seq2seq_bengali_newds.py
#
#  OUTPUT:
#    output/newds_seq2seq_training_curve.png
#    output/newds_seq2seq_bleu_histogram.png
# =============================================================================

import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


np.random.seed(42)
tf.random.set_seed(42)
SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "newds_subset.tsv"
OUT_DIR = BASE_DIR / "output"
OUT_CURVE = OUT_DIR / "newds_seq2seq_training_curve.png"
OUT_BLEU_HIST = OUT_DIR / "newds_seq2seq_bleu_histogram.png"

EMBEDDING_DIM = 64
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 35
TEST_SIZE = 0.30
VAL_SIZE = 0.10
LEARNING_RATE = 1e-3
CLIPNORM = 1.0
MAX_VOCAB_BEN = 30000
MAX_VOCAB_ENG = 20000

START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

EXAMPLE_SENTENCES = [
    ("যাও।", "Go."),
    ("বাঁচাও!", "Help!"),
    ("আমি খুশি।", "I am happy."),
    ("তোমার নাম কি?", "What is your name?"),
    ("সে একজন শিক্ষক।", "She is a teacher."),
    ("সে জানে না।", "He does not know."),
    ("আমরা বন্ধু।", "We are friends."),
    ("আমি বাড়ি যেতে চাই।", "I want to go home."),
    ("তুমি কি আমাকে সাহায্য করতে পারবে?", "Can you help me?"),
    ("অনেক ধন্যবাদ।", "Thank you very much."),
]


def find_bengali_font():
    search_names = [
        "NotoSansBengali",
        "NotoSerifBengali",
        "Vrinda",
        "Lohit-Bengali",
        "MuktiBangla",
        "SolaimanLipi",
        "Kalpurush",
        "AdorshoLipi",
    ]
    search_dirs = [
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "/System/Library/Fonts",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        str(BASE_DIR),
    ]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.endswith((".ttf", ".otf")):
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
    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        matplotlib.rcParams["font.family"] = font_name
        matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        print(f"  Bengali font loaded: {font_name}  ({font_path})")
        return font_name
    print("  WARNING: No Bengali font found — Bengali text may appear as squares.")
    print("  Fix: place NotoSansBengali-Regular.ttf in this folder and re-run.")
    return None


def clean_bengali(text):
    text = re.sub(r"[^ঀ-৿\s]", " ", text)
    return " ".join(text.split())


def clean_english(text):
    text = text.replace("’", "'").lower()
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


def load_pairs(tsv_path):
    pairs = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eng = clean_english(parts[0])
            ben = clean_bengali(parts[1])
            if eng and ben:
                pairs.append((ben, eng))
    return pairs


def preprocess_pairs(pairs):
    ben_sentences, eng_sentences = [], []
    for ben, eng in pairs:
        ben_sentences.append(ben)
        eng_sentences.append(f"{START_TOKEN} {eng} {END_TOKEN}")
    return ben_sentences, eng_sentences


def build_tokenizer(sentences, max_vocab):
    tok = Tokenizer(num_words=max_vocab, oov_token=UNK_TOKEN, filters="")
    tok.fit_on_texts(sentences)
    return tok


def masked_sequence_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def make_decoder_targets(dec_sequences):
    dec_in = dec_sequences[:, :-1]
    dec_tgt = dec_sequences[:, 1:]
    dec_tgt = np.expand_dims(dec_tgt, -1)
    return dec_in, dec_tgt


def make_dataset(enc_seq, dec_in_seq, dec_tgt_seq, batch_size, shuffle=False, seed=SEED):
    sample_weights = (np.squeeze(dec_tgt_seq, axis=-1) != 0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(
        ((enc_seq, dec_in_seq), dec_tgt_seq, sample_weights)
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(enc_seq),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(enc_vocab, dec_vocab):
    # Simplest viable seq2seq: one encoder LSTM and one decoder LSTM.
    enc_in = Input(shape=(None,), name="encoder_input")
    enc_emb = Embedding(
        enc_vocab,
        EMBEDDING_DIM,
        mask_zero=True,
        name="enc_embedding",
    )(enc_in)
    _, state_h, state_c = LSTM(
        LSTM_UNITS,
        return_state=True,
        name="encoder_lstm",
    )(enc_emb)
    enc_states = [state_h, state_c]

    dec_in = Input(shape=(None,), name="decoder_input")
    dec_emb_layer = Embedding(
        dec_vocab,
        EMBEDDING_DIM,
        mask_zero=True,
        name="dec_embedding",
    )
    dec_emb = dec_emb_layer(dec_in)
    dec_lstm = LSTM(
        LSTM_UNITS,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm",
    )
    dec_out_seq, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
    dec_dense = Dense(dec_vocab, activation="softmax", name="decoder_output")
    dec_out = dec_dense(dec_out_seq)

    training_model = Model([enc_in, dec_in], dec_out)
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss="sparse_categorical_crossentropy",
        metrics=[masked_sequence_accuracy],
        weighted_metrics=[],
    )

    encoder_model = Model(enc_in, enc_states)

    state_h_in = Input(shape=(LSTM_UNITS,), name="decoder_state_h")
    state_c_in = Input(shape=(LSTM_UNITS,), name="decoder_state_c")
    dec_states_in = [state_h_in, state_c_in]
    dec_single_emb = dec_emb_layer(dec_in)
    dec_single_out, h, c = dec_lstm(dec_single_emb, initial_state=dec_states_in)
    dec_single_probs = dec_dense(dec_single_out)
    decoder_model = Model(
        [dec_in] + dec_states_in,
        [dec_single_probs, h, c],
    )

    return training_model, encoder_model, decoder_model


def translate(sentence, encoder_model, decoder_model, ben_tok, eng_tok, max_dec_len):
    seq = ben_tok.texts_to_sequences([clean_bengali(sentence)])
    seq = tf.constant(pad_sequences(seq, maxlen=encoder_model.input_shape[1], padding="post"))
    states = list(encoder_model(seq, training=False))

    start_idx = eng_tok.word_index.get(START_TOKEN, 1)
    end_idx = eng_tok.word_index.get(END_TOKEN, 2)
    target_seq = tf.constant([[start_idx]])
    decoded_tokens = []

    for _ in range(max_dec_len):
        output, h, c = decoder_model([target_seq] + states, training=False)
        token_idx = int(tf.argmax(output[0, -1, :]))
        if token_idx in (0, end_idx):
            break
        word = eng_tok.index_word.get(token_idx, "")
        if word and word not in (START_TOKEN, END_TOKEN, UNK_TOKEN):
            decoded_tokens.append(word)
        target_seq = tf.constant([[token_idx]])
        states = [h, c]

    return " ".join(decoded_tokens)


def compute_bleu(test_pairs, encoder_model, decoder_model, ben_tok, eng_tok, max_dec_len):
    references, hypotheses = [], []
    total = len(test_pairs)
    for i, (ben, eng_ref_full) in enumerate(test_pairs):
        if i % 100 == 0:
            print(f"    Translating {i}/{total} ...", flush=True)

        ref = eng_ref_full.replace(START_TOKEN, "").replace(END_TOKEN, "").split()
        hyp = translate(ben, encoder_model, decoder_model, ben_tok, eng_tok, max_dec_len).split()
        references.append([ref])
        hypotheses.append(hyp)

    bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    return bleu, references, hypotheses


def plot_training_curve(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Train loss", color="#4e79a7")
    ax1.plot(history.history["val_loss"], label="Val loss", color="#f28e2b", linestyle="--")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(history.history["masked_sequence_accuracy"], label="Train masked accuracy", color="#4e79a7")
    ax2.plot(history.history["val_masked_sequence_accuracy"], label="Val masked accuracy", color="#f28e2b", linestyle="--")
    ax2.set_title("Training & Validation Masked Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Seq2Seq LSTM — Bengali → English (New Dataset)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_bleu_histogram(per_sentence_bleus, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_sentence_bleus, bins=30, color="#59a14f", edgecolor="white", alpha=0.85)
    ax.axvline(
        np.mean(per_sentence_bleus),
        color="#e15759",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {np.mean(per_sentence_bleus):.4f}",
    )
    ax.set_title("Per-Sentence BLEU Distribution (Test Set)", fontweight="bold")
    ax.set_xlabel("Sentence BLEU")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {save_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  SEQ2SEQ LSTM — Bengali → English Translation (New Dataset)")
    print("=" * 72)

    print("\n[0] Setting up Bengali font for console display ...")
    setup_bengali_font()

    print(f"\n[1] Loading data from {DATA_PATH}")
    raw_pairs = load_pairs(DATA_PATH)
    print(f"    {len(raw_pairs):,} sentence pairs loaded")

    print("\n[2] Preprocessing translation pairs ...")
    ben_sents, eng_sents = preprocess_pairs(raw_pairs)
    print(f"    {len(ben_sents):,} valid pairs prepared")

    print(f"\n[3] Splitting data (test = {int(TEST_SIZE * 100)}%, val = {int(VAL_SIZE * 100)}% of train) ...")
    ben_train_full, ben_test, eng_train_full, eng_test = train_test_split(
        ben_sents,
        eng_sents,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )
    ben_train, ben_val, eng_train, eng_val = train_test_split(
        ben_train_full,
        eng_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=True,
    )
    print(f"    Train: {len(ben_train):,}  |  Val: {len(ben_val):,}  |  Test: {len(ben_test):,}")

    print("\n[4] Building tokenisers ...")
    ben_tok = build_tokenizer(ben_train, MAX_VOCAB_BEN)
    eng_tok = build_tokenizer(eng_train, MAX_VOCAB_ENG)
    enc_vocab = min(len(ben_tok.word_index) + 1, MAX_VOCAB_BEN)
    dec_vocab = min(len(eng_tok.word_index) + 1, MAX_VOCAB_ENG)
    print(f"    Bengali vocab (encoder) : {enc_vocab:,} tokens")
    print(f"    English vocab (decoder) : {dec_vocab:,} tokens")

    print("\n[5] Encoding and padding sequences ...")
    max_enc = max(len(s.split()) for s in ben_train)
    max_dec = max(len(s.split()) for s in eng_train)
    print(f"    Max encoder length (Bengali) : {max_enc}")
    print(f"    Max decoder length (English) : {max_dec}")

    enc_train_seq = pad_sequences(ben_tok.texts_to_sequences(ben_train), maxlen=max_enc, padding="post")
    dec_train_seq = pad_sequences(eng_tok.texts_to_sequences(eng_train), maxlen=max_dec, padding="post")
    enc_val_seq = pad_sequences(ben_tok.texts_to_sequences(ben_val), maxlen=max_enc, padding="post")
    dec_val_seq = pad_sequences(eng_tok.texts_to_sequences(eng_val), maxlen=max_dec, padding="post")

    dec_in_train, dec_tgt_train = make_decoder_targets(dec_train_seq)
    dec_in_val, dec_tgt_val = make_decoder_targets(dec_val_seq)

    print("\n[6] Building seq2seq model ...")
    training_model, encoder_model, decoder_model = build_model(enc_vocab, dec_vocab)
    training_model.summary()

    print(f"\n[7] Training for {EPOCHS} epochs (batch={BATCH_SIZE}) ...")
    train_ds = make_dataset(enc_train_seq, dec_in_train, dec_tgt_train, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(enc_val_seq, dec_in_val, dec_tgt_val, batch_size=BATCH_SIZE, shuffle=False)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]
    history = training_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 8. Example translations ───────────────────────────────────────────────
    # Bengali input sentences with their known English meanings
    print("\n[8] Example translations (greedy decode):")
    print(f"    {'Bengali Input':<40}  {'Reference English':<28}  Predicted English")
    print(f"    {'-'*40}  {'-'*28}  {'-'*30}")
    for ben_sent, ref_eng in EXAMPLE_SENTENCES:
        prediction = translate(ben_sent, encoder_model, decoder_model, ben_tok, eng_tok, max_dec_len=max_dec)
        print(f"    {ben_sent:<40}  {ref_eng:<28}  {prediction if prediction else '(empty)'}")

    # ── 9. BLEU score ─────────────────────────────────────────────────────────
    print("\n[9] Computing BLEU score on test set ...")
    test_examples = list(zip(ben_test, eng_test))
    corpus_bleu_score, references, hypotheses = compute_bleu(
        test_examples,
        encoder_model,
        decoder_model,
        ben_tok,
        eng_tok,
        max_dec_len=max_dec,
    )
    smooth = SmoothingFunction().method1
    per_sent_bleus = [sentence_bleu(ref, hyp, smoothing_function=smooth) for ref, hyp in zip(references, hypotheses)]
    print(f"\n    Corpus BLEU-4 on test set : {corpus_bleu_score:.4f}")
    print(f"    Mean sentence BLEU        : {np.mean(per_sent_bleus):.4f}")
    print(f"    Median sentence BLEU      : {np.median(per_sent_bleus):.4f}")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    print("\n[10] Generating plots ...")
    plot_training_curve(history, OUT_CURVE)
    plot_bleu_histogram(per_sent_bleus, OUT_BLEU_HIST)

    print("\n" + "=" * 72)
    print("  Results Summary")
    print("=" * 72)
    print(f"  Total pairs used            : {len(ben_sents):,}")
    print(f"  Training pairs              : {len(ben_train):,}")
    print(f"  Test pairs                  : {len(ben_test):,}")
    print(f"  Bengali vocabulary (enc)    : {enc_vocab:,}")
    print(f"  English vocabulary (dec)    : {dec_vocab:,}")
    print(f"  Corpus BLEU-4 (test)        : {corpus_bleu_score:.4f}")
    print(f"  Encoder LSTM units          : {LSTM_UNITS}")
    print(f"  Decoder LSTM units          : {LSTM_UNITS}")
    print(f"  Embedding dimension         : {EMBEDDING_DIM}")
    print(f"  Epochs                      : {EPOCHS}")
    print("=" * 72)
    print("\nArchitecture justification:")
    print("  Plain encoder-decoder LSTM: this is the simplest viable baseline")
    print("  for the new dataset. One encoder LSTM, one decoder LSTM, and one")
    print("  embedding layer per language keep the parameter count manageable")
    print("  while still letting the model learn sentence-level structure.")
    print("  128 LSTM units and 64-dim embeddings are compact enough for the")
    print("  100k-pair subset and provide a clean baseline before attention.")
    print("=" * 72)
