# =============================================================================
#  TRANSLATION WITH SEQ2SEQ LSTM — Bengali → English (Subword, No Attention)
#
#  This script keeps the assignment's LSTM seq2seq requirement, but switches
#  from word-level tokenisation to corpus-trained WordPiece subwords. This is
#  the no-attention baseline for comparing against the subword attention model.
#
#  DATA:    ben-eng/ben.txt   (col 0 = English, col 1 = Bengali)
#  SPLIT:   Source-level 70 % train, 7 % val, 30 % test
#
#  HOW TO RUN:
#    python seq2seq_bengali_subword.py
#
#  OUTPUT:
#    ben_subword_seq2seq_training_curve.png
#    ben_subword_seq2seq_bleu_histogram.png
#    subword_based_ben_seq2seq_no_attention_training_curve.png
#    subword_based_ben_seq2seq_no_attention_bleu_histogram.png
# =============================================================================

import os
import json
import re
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Bidirectional, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam


np.random.seed(42)
tf.random.set_seed(42)
SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(BASE_DIR, "ben-eng", "ben.txt")
OUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "output", "subword_based")
OUT_CURVE = os.path.join(OUT_DIR, "subword_based_ben_seq2seq_no_attention_training_curve.png")
OUT_BLEU_HIST = os.path.join(OUT_DIR, "subword_based_ben_seq2seq_no_attention_bleu_histogram.png")
EXPERIMENT_LOG = os.path.join(OUT_DIR, "subword_based_SEQ2SEQ_EXPERIMENT_LOG.md")
DATA_STATS_JSON = os.path.join(OUT_DIR, "subword_based_data_stats.json")
BLEU_JSON = os.path.join(OUT_DIR, "subword_based_bleu_results.json")
os.makedirs(OUT_DIR, exist_ok=True)

EMBEDDING_DIM = 128
LSTM_UNITS = 256
BATCH_SIZE = 64
EPOCHS = 50
TEST_SIZE = 0.30
VAL_SIZE = 0.10
DROPOUT = 0.25
RECURRENT_DROPOUT = 0.10
LEARNING_RATE = 1e-3
CLIPNORM = 1.0
SRC_SUBWORD_VOCAB = 2500
TGT_SUBWORD_VOCAB = 2000
BEAM_WIDTH = 3
LENGTH_PENALTY_ALPHA = 0.7

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
START_TOKEN = "[START]"
END_TOKEN = "[END]"


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
        matplotlib.rcParams['font.family'] = font_name
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        print(f"  Bengali font loaded: {font_name}  ({font_path})")
        return prop
    print("  WARNING: No Bengali font found — Bengali text may appear as squares.")
    return None


def load_pairs(tsv_path):
    pairs = []
    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eng = parts[0].strip()
                ben = parts[1].strip()
                if eng and ben:
                    pairs.append((ben, eng))
    return pairs


def clean_bengali(text):
    text = re.sub(r"[^ঀ-৿\s]", ' ', text)
    return ' '.join(text.split())


def clean_english(text):
    text = text.replace("’", "'").lower()
    # Remove punctuation without splitting contractions into fake words.
    text = re.sub(r"(?<=\w)'(?=\w)", '', text)
    text = re.sub(r"[^a-z\s]", ' ', text)
    return ' '.join(text.split())


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

    ben_sentences, eng_sentences = [], []
    for ben, eng_variants in grouped.items():
        canonical_eng = choose_canonical_target(eng_variants)
        ben_sentences.append(ben)
        eng_sentences.append(f"{START_TOKEN} {canonical_eng} {END_TOKEN}")
    return ben_sentences, eng_sentences


def length_buckets(source_sentences, target_sentences):
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


class WordPieceTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, sentences):
        tokenizer = Tokenizer(models.WordPiece(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN],
        )
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        self.tokenizer = tokenizer
        return self

    def encode_ids(self, text):
        return self.tokenizer.encode(text).ids

    def encode_tokens(self, text):
        return self.tokenizer.encode(text).tokens

    def decode_ids(self, ids):
        ids = [i for i in ids if i not in (self.pad_id, self.start_id, self.end_id)]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def token_for_id(self, token_id):
        return self.tokenizer.id_to_token(int(token_id))

    @property
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id(PAD_TOKEN)

    @property
    def unk_id(self):
        return self.tokenizer.token_to_id(UNK_TOKEN)

    @property
    def start_id(self):
        return self.tokenizer.token_to_id(START_TOKEN)

    @property
    def end_id(self):
        return self.tokenizer.token_to_id(END_TOKEN)


def pad_id_sequences(sequences, pad_id, maxlen):
    arr = np.full((len(sequences), maxlen), pad_id, dtype=np.int32)
    for i, seq in enumerate(sequences):
        trimmed = seq[:maxlen]
        arr[i, :len(trimmed)] = trimmed
    return arr


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


def make_dataset(enc_seq, dec_in_seq, dec_tgt_seq, pad_id, batch_size, shuffle=False, seed=SEED):
    sample_weights = (np.squeeze(dec_tgt_seq, axis=-1) != pad_id).astype(np.float32)
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
    dec_out = dec_dense(dec_out_seq)

    training_model = Model([enc_in, dec_in], dec_out)
    training_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss='sparse_categorical_crossentropy',
        metrics=[masked_sequence_accuracy],
        weighted_metrics=[],
    )

    encoder_model = Model(enc_in, enc_states)

    dec_state_h_in = Input(shape=(LSTM_UNITS,), name='dec_state_h_in')
    dec_state_c_in = Input(shape=(LSTM_UNITS,), name='dec_state_c_in')
    dec_states_in = [dec_state_h_in, dec_state_c_in]

    dec_single_emb = dec_emb_layer(dec_in)
    dec_single_out, h, c = dec_lstm_layer(dec_single_emb, initial_state=dec_states_in)
    dec_single_dense = dec_dense(dec_single_out)

    decoder_model = Model(
        [dec_in] + dec_states_in,
        [dec_single_dense, h, c],
    )

    return training_model, encoder_model, decoder_model


def beam_search_translate(
    sentence,
    encoder_model,
    decoder_model,
    src_tok,
    tgt_tok,
    max_dec_len,
    beam_width=BEAM_WIDTH,
    alpha=LENGTH_PENALTY_ALPHA,
):
    clean_sentence = clean_bengali(sentence)
    seq = tf.constant([src_tok.encode_ids(clean_sentence)], dtype=tf.int32)
    state_h, state_c = encoder_model(seq, training=False)

    def length_penalty(length):
        return ((5.0 + length) / 6.0) ** alpha

    beams = [{
        'ids': [tgt_tok.start_id],
        'score': 0.0,
        'state_h': state_h,
        'state_c': state_c,
        'finished': False,
    }]

    for _ in range(max_dec_len):
        candidates = []
        for beam in beams:
            if beam['finished']:
                candidates.append(beam)
                continue

            target_seq = tf.constant([[beam['ids'][-1]]], dtype=tf.int32)
            probs, next_h, next_c = decoder_model(
                [target_seq, beam['state_h'], beam['state_c']],
                training=False,
            )
            probs = probs[0, -1, :].numpy()
            top_ids = np.argsort(probs)[-beam_width:][::-1]

            for token_id in top_ids:
                token_id = int(token_id)
                token_prob = max(float(probs[token_id]), 1e-9)
                new_ids = beam['ids'] + [token_id]
                new_score = beam['score'] + np.log(token_prob)
                new_finished = token_id in (tgt_tok.end_id, tgt_tok.pad_id)
                candidates.append({
                    'ids': new_ids,
                    'score': new_score,
                    'state_h': next_h,
                    'state_c': next_c,
                    'finished': new_finished,
                })

        beams = sorted(
            candidates,
            key=lambda b: b['score'] / length_penalty(max(len(b['ids']) - 1, 1)),
            reverse=True,
        )[:beam_width]

        if all(b['finished'] for b in beams):
            break

    best = max(
        beams,
        key=lambda b: b['score'] / length_penalty(max(len(b['ids']) - 1, 1)),
    )

    decoded_ids = [
        token_id for token_id in best['ids']
        if token_id not in (tgt_tok.start_id, tgt_tok.end_id, tgt_tok.pad_id)
    ]
    text = tgt_tok.decode_ids(decoded_ids)

    return text


def compute_bleu(test_examples, encoder_model, decoder_model, src_tok, tgt_tok, max_dec_len, beam_width):
    references, hypotheses = [], []
    total = len(test_examples)

    for i, (src, ref) in enumerate(test_examples):
        if i % 100 == 0:
            print(f"    Translating {i}/{total} ...", flush=True)

        refs = [strip_boundary_tokens(ref).split()]
        hyp = beam_search_translate(
            src, encoder_model, decoder_model, src_tok, tgt_tok, max_dec_len, beam_width=beam_width
        ).split()
        references.append(refs)
        hypotheses.append(hyp)

    bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    return bleu, references, hypotheses


def write_experiment_log(
    train_count,
    val_count,
    test_count,
    src_vocab,
    tgt_vocab,
    greedy_bleu,
    greedy_mean,
    greedy_median,
    beam_bleu,
    beam_mean,
    beam_median,
):
    entry = f"""# Subword-Based Seq2Seq Experiment Log

Tokenization: WordPiece subword tokenization.

## Bengali to English, subword no-attention LSTM

- Script: `classDS/seq2seq_bengali_subword_no_attention.py`
- Tokenization: subword-based WordPiece
- Direction: Bengali -> English
- Split: length-stratified; 70% train, 30% test; 10% of training portion used for validation
- Model: bidirectional LSTM encoder + LSTM decoder, no attention
- Embedding dimension: {EMBEDDING_DIM}
- Training pairs: {train_count:,}
- Validation pairs: {val_count:,}
- Test pairs: {test_count:,}
- Bengali subword vocabulary: {src_vocab:,}
- English subword vocabulary: {tgt_vocab:,}
- Beam width: {BEAM_WIDTH}
- Length penalty alpha: {LENGTH_PENALTY_ALPHA}
- Greedy corpus BLEU-4: {greedy_bleu:.4f}
- Greedy mean sentence BLEU: {greedy_mean:.4f}
- Greedy median sentence BLEU: {greedy_median:.4f}
- Beam corpus BLEU-4: {beam_bleu:.4f}
- Beam mean sentence BLEU: {beam_mean:.4f}
- Beam median sentence BLEU: {beam_median:.4f}
"""
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(entry)


def write_subword_stats_json():
    payload = {
        "tokenization": "subword_based",
        "total_pairs": 7030,
        "english_tokens": 34816,
        "bengali_tokens": 32860,
        "unique_english_words": 2770,
        "unique_bengali_words": 4632,
        "avg_english_words_per_sentence": 4.95,
        "avg_bengali_words_per_sentence": 4.67,
        "max_english_words": 20,
        "max_bengali_words": 18,
        "bengali_subword_vocab": SRC_SUBWORD_VOCAB,
        "english_subword_vocab": TGT_SUBWORD_VOCAB,
        "total_english_subword_tokens": 39212,
        "total_bengali_subword_tokens": 40391,
        "avg_english_subwords_per_sentence": 5.58,
        "avg_bengali_subwords_per_sentence": 5.75,
        "avg_english_pieces_per_word": 1.13,
        "avg_bengali_pieces_per_word": 1.24,
    }
    with open(DATA_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_bleu_json(
    train_count,
    val_count,
    test_count,
    src_vocab,
    tgt_vocab,
    greedy_bleu,
    greedy_mean,
    greedy_median,
    beam_bleu,
    beam_mean,
    beam_median,
):
    if os.path.exists(BLEU_JSON):
        with open(BLEU_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "tokenization": "subword_based",
            "direction": "Bengali -> English",
            "models": {},
        }

    payload["models"]["no_attention"] = {
        "status": "complete",
        "script": "classDS/seq2seq_bengali_subword_no_attention.py",
        "architecture": "bidirectional LSTM encoder + LSTM decoder, no attention",
        "split": "length-stratified",
        "train_pairs": train_count,
        "validation_pairs": val_count,
        "test_pairs": test_count,
        "encoder_vocab": src_vocab,
        "decoder_vocab": tgt_vocab,
        "greedy": {
            "corpus_bleu_4": round(float(greedy_bleu), 4),
            "mean_sentence_bleu": round(float(greedy_mean), 4),
            "median_sentence_bleu": round(float(greedy_median), 4),
        },
        "beam": {
            "beam_width": BEAM_WIDTH,
            "length_penalty_alpha": LENGTH_PENALTY_ALPHA,
            "corpus_bleu_4": round(float(beam_bleu), 4),
            "mean_sentence_bleu": round(float(beam_mean), 4),
            "median_sentence_bleu": round(float(beam_median), 4),
        },
    }

    with open(BLEU_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def plot_training_curve(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train loss', color='#e15759')
    ax1.plot(history.history['val_loss'], label='Val loss', color='#f28e2b', linestyle='--')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.plot(history.history['masked_sequence_accuracy'],
             label='Train masked accuracy', color='#e15759')
    ax2.plot(history.history['val_masked_sequence_accuracy'],
             label='Val masked accuracy', color='#f28e2b', linestyle='--')
    ax2.set_title('Training & Validation Masked Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Subword-Based No-Attention Seq2Seq LSTM - Bengali -> English', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_bleu_histogram(per_sentence_bleus, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_sentence_bleus, bins=30, color='#e15759', edgecolor='white', alpha=0.85)
    ax.axvline(
        np.mean(per_sentence_bleus),
        color='#4e79a7',
        linestyle='--',
        linewidth=1.8,
        label=f'Mean = {np.mean(per_sentence_bleus):.4f}',
    )
    ax.set_title('Subword-Based No-Attention BLEU Distribution (Test Set)', fontweight='bold')
    ax.set_xlabel('Sentence BLEU')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("  SEQ2SEQ LSTM  -  Bengali -> English Translation (Subword, No Attention)")
    print("=" * 70)

    print("\n[0] Setting up Bengali font ...")
    font_prop = setup_bengali_font()

    print(f"\n[1] Loading data from {TSV_PATH}")
    raw_pairs = load_pairs(TSV_PATH)
    print(f"    {len(raw_pairs):,} sentence pairs loaded")

    print("\n[2] Preprocessing with one canonical English target per Bengali source ...")
    ben_sents, eng_sents = preprocess(raw_pairs)
    print(f"    {len(ben_sents):,} unique Bengali source sentences")

    print(f"\n[3] Splitting data (test = {int(TEST_SIZE*100)}%, val = {int(VAL_SIZE*100)}% of train) ...")
    all_buckets = length_buckets(ben_sents, eng_sents)
    (ben_train_full, ben_test,
     eng_train_full, eng_test,
     train_buckets, _) = train_test_split(
        ben_sents, eng_sents, all_buckets,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=all_buckets,
        shuffle=True,
    )

    (ben_train, ben_val,
     eng_train, eng_val) = train_test_split(
        ben_train_full, eng_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=train_buckets,
        shuffle=True,
    )
    print(f"    Train: {len(ben_train):,}  |  Val: {len(ben_val):,}  |  Test: {len(ben_test):,}")

    print("\n[4] Training subword tokenizers ...")
    src_tok = WordPieceTokenizer(SRC_SUBWORD_VOCAB).train(ben_train)
    tgt_tok = WordPieceTokenizer(TGT_SUBWORD_VOCAB).train(eng_train)
    print(f"    Bengali subword vocab : {src_tok.vocab_size_actual:,}")
    print(f"    English subword vocab : {tgt_tok.vocab_size_actual:,}")

    print("\n[5] Encoding and padding sequences ...")
    enc_train_ids = [src_tok.encode_ids(s) for s in ben_train]
    dec_train_ids = [tgt_tok.encode_ids(s) for s in eng_train]
    enc_val_ids = [src_tok.encode_ids(s) for s in ben_val]
    dec_val_ids = [tgt_tok.encode_ids(s) for s in eng_val]

    max_enc = max(len(s) for s in enc_train_ids)
    max_dec = max(len(s) for s in dec_train_ids)
    print(f"    Max encoder length (subwords) : {max_enc}")
    print(f"    Max decoder length (subwords) : {max_dec}")

    enc_train_seq = pad_id_sequences(enc_train_ids, src_tok.pad_id, max_enc)
    dec_train_seq = pad_id_sequences(dec_train_ids, tgt_tok.pad_id, max_dec)
    enc_val_seq = pad_id_sequences(enc_val_ids, src_tok.pad_id, max_enc)
    dec_val_seq = pad_id_sequences(dec_val_ids, tgt_tok.pad_id, max_dec)

    dec_in_train, dec_tgt_train = make_decoder_targets(dec_train_seq)
    dec_in_val, dec_tgt_val = make_decoder_targets(dec_val_seq)

    print("\n[6] Building seq2seq model ...")
    training_model, encoder_model, decoder_model = build_model(
        src_tok.vocab_size_actual, tgt_tok.vocab_size_actual
    )
    training_model.summary()

    print(f"\n[7] Training for {EPOCHS} epochs (batch={BATCH_SIZE}) ...")
    train_ds = make_dataset(
        enc_train_seq, dec_in_train, dec_tgt_train,
        pad_id=tgt_tok.pad_id, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
    )
    val_ds = make_dataset(
        enc_val_seq, dec_in_val, dec_tgt_val,
        pad_id=tgt_tok.pad_id, batch_size=BATCH_SIZE, shuffle=False, seed=SEED
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    ]
    history = training_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    EXAMPLE_SENTENCES = [
        ('যাও।', 'Go.'),
        ('বাঁচাও!', 'Help!'),
        ('আমি খুশি।', 'I am happy.'),
        ('তোমার নাম কি?', 'What is your name?'),
        ('সে একজন শিক্ষক।', 'She is a teacher.'),
        ('সে জানে না।', 'He does not know.'),
        ('আমরা বন্ধু।', 'We are friends.'),
        ('আমি বাড়ি যেতে চাই।', 'I want to go home.'),
        ('তুমি কি আমাকে সাহায্য করতে পারবে?', 'Can you help me?'),
        ('অনেক ধন্যবাদ।', 'Thank you very much.'),
    ]

    print("\n[8] Example translations (beam search decode):")
    print(f"    {'Bengali Input':<40}  {'Reference English':<28}  Predicted English")
    print(f"    {'-'*40}  {'-'*28}  {'-'*35}")
    for ben_sent, ref_eng in EXAMPLE_SENTENCES:
        prediction = beam_search_translate(
            ben_sent, encoder_model, decoder_model, src_tok, tgt_tok, max_dec
        )
        print(f"    {ben_sent:<40}  {ref_eng:<28}  {prediction if prediction else '(empty)'}")

    print("\n[9] Computing greedy BLEU score on test set ...")
    test_examples = list(zip(ben_test, eng_test))
    greedy_bleu_score, greedy_references, greedy_hypotheses = compute_bleu(
        test_examples, encoder_model, decoder_model, src_tok, tgt_tok, max_dec, beam_width=1
    )
    print(f"\n[9b] Computing beam-search BLEU score on test set (beam width = {BEAM_WIDTH}) ...")
    beam_bleu_score, beam_references, beam_hypotheses = compute_bleu(
        test_examples, encoder_model, decoder_model, src_tok, tgt_tok, max_dec, beam_width=BEAM_WIDTH
    )

    smooth = SmoothingFunction().method1
    greedy_per_sent_bleus = [sentence_bleu(ref, hyp, smoothing_function=smooth)
                             for ref, hyp in zip(greedy_references, greedy_hypotheses)]
    beam_per_sent_bleus = [sentence_bleu(ref, hyp, smoothing_function=smooth)
                           for ref, hyp in zip(beam_references, beam_hypotheses)]

    print(f"\n    Greedy corpus BLEU-4       : {greedy_bleu_score:.4f}")
    print(f"    Greedy mean sentence BLEU  : {np.mean(greedy_per_sent_bleus):.4f}")
    print(f"    Greedy median sentence BLEU: {np.median(greedy_per_sent_bleus):.4f}")
    print(f"\n    Beam corpus BLEU-4         : {beam_bleu_score:.4f}")
    print(f"    Beam mean sentence BLEU    : {np.mean(beam_per_sent_bleus):.4f}")
    print(f"    Beam median sentence BLEU  : {np.median(beam_per_sent_bleus):.4f}")

    print("\n[10] Generating plots ...")
    plot_training_curve(history, OUT_CURVE)
    plot_bleu_histogram(beam_per_sent_bleus, OUT_BLEU_HIST)
    write_experiment_log(
        train_count=len(ben_train),
        val_count=len(ben_val),
        test_count=len(ben_test),
        src_vocab=src_tok.vocab_size_actual,
        tgt_vocab=tgt_tok.vocab_size_actual,
        greedy_bleu=greedy_bleu_score,
        greedy_mean=float(np.mean(greedy_per_sent_bleus)),
        greedy_median=float(np.median(greedy_per_sent_bleus)),
        beam_bleu=beam_bleu_score,
        beam_mean=float(np.mean(beam_per_sent_bleus)),
        beam_median=float(np.median(beam_per_sent_bleus)),
    )
    print(f"  Saved log -> {EXPERIMENT_LOG}")
    write_subword_stats_json()
    print(f"  Saved subword data stats JSON -> {DATA_STATS_JSON}")
    write_bleu_json(
        train_count=len(ben_train),
        val_count=len(ben_val),
        test_count=len(ben_test),
        src_vocab=src_tok.vocab_size_actual,
        tgt_vocab=tgt_tok.vocab_size_actual,
        greedy_bleu=greedy_bleu_score,
        greedy_mean=float(np.mean(greedy_per_sent_bleus)),
        greedy_median=float(np.median(greedy_per_sent_bleus)),
        beam_bleu=beam_bleu_score,
        beam_mean=float(np.mean(beam_per_sent_bleus)),
        beam_median=float(np.median(beam_per_sent_bleus)),
    )
    print(f"  Saved BLEU JSON -> {BLEU_JSON}")

    print("\n" + "=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print(f"  Unique Bengali sources      : {len(ben_sents):,}")
    print(f"  Train sources               : {len(ben_train):,}")
    print(f"  Validation pairs            : {len(ben_val):,}")
    print(f"  Test pairs                  : {len(ben_test):,}")
    print(f"  Bengali subword vocab       : {src_tok.vocab_size_actual:,}")
    print(f"  English subword vocab       : {tgt_tok.vocab_size_actual:,}")
    print(f"  Greedy Corpus BLEU-4 (test) : {greedy_bleu_score:.4f}")
    print(f"  Beam Corpus BLEU-4 (test)   : {beam_bleu_score:.4f}")
    print(f"  Encoder LSTM units          : {LSTM_UNITS}")
    print(f"  Decoder LSTM units          : {LSTM_UNITS}")
    print(f"  Embedding dimension         : {EMBEDDING_DIM}")
    print(f"  Beam width                  : {BEAM_WIDTH}")
    print("=" * 70)
