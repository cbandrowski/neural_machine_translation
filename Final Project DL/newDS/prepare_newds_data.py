import json
import os
import random
import re
from collections import Counter
from pathlib import Path


SEED = 42
TARGET_SAMPLE_SIZE = 100_000
MIN_TOKENS = 1
MAX_TOKENS = 25
MAX_LENGTH_RATIO = 2.5


BASE_DIR = Path(__file__).resolve().parent
RAW_EN_PATH = BASE_DIR / "2.75M" / "original_corpus.en"
RAW_BN_PATH = BASE_DIR / "2.75M" / "original_corpus.bn"
OUT_DIR = BASE_DIR / "data"

OUT_TSV = OUT_DIR / "newds_subset.tsv"
OUT_EN = OUT_DIR / "newds_subset.en"
OUT_BN = OUT_DIR / "newds_subset.bn"
OUT_VOCAB_EN = OUT_DIR / "vocab_english.txt"
OUT_VOCAB_BN = OUT_DIR / "vocab_bengali.txt"
OUT_STATS = OUT_DIR / "data_prep_stats.json"
OUT_PREVIEW = OUT_DIR / "preview_samples.txt"


def clean_english(text):
    text = text.replace("’", "'").lower()
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


def clean_bengali(text):
    text = re.sub(r"[^ঀ-৿\s]", " ", text)
    return " ".join(text.split())


def token_count(text):
    return len(text.split())


def is_length_ok(eng, ben):
    eng_len = token_count(eng)
    ben_len = token_count(ben)
    if not (MIN_TOKENS <= eng_len <= MAX_TOKENS):
        return False
    if not (MIN_TOKENS <= ben_len <= MAX_TOKENS):
        return False
    ratio = max(eng_len / ben_len, ben_len / eng_len)
    return ratio <= MAX_LENGTH_RATIO


def update_counter(counter, sentence):
    counter.update(sentence.split())


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_vocab(path, counter):
    with open(path, "w", encoding="utf-8") as f:
        for token, freq in counter.most_common():
            f.write(f"{token}\t{freq}\n")


def write_preview(path, pairs, n=25):
    with open(path, "w", encoding="utf-8") as f:
        for i, (eng, ben) in enumerate(pairs[:n], start=1):
            f.write(f"[{i}]\n")
            f.write(f"EN\t{eng}\n")
            f.write(f"BN\t{ben}\n\n")


def reservoir_sample_pairs():
    rng = random.Random(SEED)
    sample_pairs = []
    seen_pairs = set()

    stats = {
        "total_raw_pairs": 0,
        "empty_after_cleaning": 0,
        "filtered_by_length_or_ratio": 0,
        "exact_duplicate_pairs": 0,
        "eligible_unique_pairs": 0,
    }

    with open(RAW_EN_PATH, encoding="utf-8", errors="replace") as f_en, \
         open(RAW_BN_PATH, encoding="utf-8", errors="replace") as f_bn:
        for raw_eng, raw_ben in zip(f_en, f_bn):
            stats["total_raw_pairs"] += 1

            eng = clean_english(raw_eng.strip())
            ben = clean_bengali(raw_ben.strip())

            if not eng or not ben:
                stats["empty_after_cleaning"] += 1
                continue

            if not is_length_ok(eng, ben):
                stats["filtered_by_length_or_ratio"] += 1
                continue

            pair = (eng, ben)
            if pair in seen_pairs:
                stats["exact_duplicate_pairs"] += 1
                continue

            seen_pairs.add(pair)
            stats["eligible_unique_pairs"] += 1

            if len(sample_pairs) < TARGET_SAMPLE_SIZE:
                sample_pairs.append(pair)
            else:
                j = rng.randint(0, stats["eligible_unique_pairs"] - 1)
                if j < TARGET_SAMPLE_SIZE:
                    sample_pairs[j] = pair

    rng.shuffle(sample_pairs)
    return sample_pairs, stats


def build_summary(sample_pairs, stats):
    eng_counter = Counter()
    ben_counter = Counter()

    eng_lengths = []
    ben_lengths = []

    for eng, ben in sample_pairs:
        update_counter(eng_counter, eng)
        update_counter(ben_counter, ben)
        eng_lengths.append(token_count(eng))
        ben_lengths.append(token_count(ben))

    stats.update({
        "sample_size": len(sample_pairs),
        "english_vocab_size": len(eng_counter),
        "bengali_vocab_size": len(ben_counter),
        "english_avg_tokens": round(sum(eng_lengths) / len(eng_lengths), 3) if eng_lengths else 0,
        "bengali_avg_tokens": round(sum(ben_lengths) / len(ben_lengths), 3) if ben_lengths else 0,
        "english_max_tokens": max(eng_lengths) if eng_lengths else 0,
        "bengali_max_tokens": max(ben_lengths) if ben_lengths else 0,
    })

    return eng_counter, ben_counter, stats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Preparing New Dataset Sample")
    print("=" * 72)
    print(f"Raw English corpus : {RAW_EN_PATH}")
    print(f"Raw Bengali corpus : {RAW_BN_PATH}")
    print(f"Target sample size : {TARGET_SAMPLE_SIZE:,}")
    print(f"Token length range : [{MIN_TOKENS}, {MAX_TOKENS}]")
    print(f"Max length ratio   : {MAX_LENGTH_RATIO}")

    sample_pairs, stats = reservoir_sample_pairs()
    eng_counter, ben_counter, stats = build_summary(sample_pairs, stats)

    with open(OUT_TSV, "w", encoding="utf-8") as f:
        for eng, ben in sample_pairs:
            f.write(f"{eng}\t{ben}\n")

    write_lines(OUT_EN, [eng for eng, _ in sample_pairs])
    write_lines(OUT_BN, [ben for _, ben in sample_pairs])
    write_vocab(OUT_VOCAB_EN, eng_counter)
    write_vocab(OUT_VOCAB_BN, ben_counter)
    write_preview(OUT_PREVIEW, sample_pairs)

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    print(f"  {OUT_TSV}")
    print(f"  {OUT_EN}")
    print(f"  {OUT_BN}")
    print(f"  {OUT_VOCAB_EN}")
    print(f"  {OUT_VOCAB_BN}")
    print(f"  {OUT_STATS}")
    print(f"  {OUT_PREVIEW}")

    print("\nSummary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
