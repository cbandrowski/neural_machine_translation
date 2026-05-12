"""
Task 2.1 data-preparation analysis for the English-Bengali NMT project.

The script reads ben-eng/ben.txt, applies the same core preprocessing described
in the report, prints dataset/dictionary statistics, and creates simple SVG
charts that can be used in the final report.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ben-eng" / "ben.txt"
OUT_DIR = BASE_DIR / "output" / "data_analysis"

TOP_N = 15

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def clean_english(text: str) -> str:
    """Lowercase and keep English alphabetic tokens only."""
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return " ".join(text.lower().split())


def clean_bengali(text: str) -> str:
    """Keep Bengali Unicode characters and whitespace only."""
    text = re.sub(r"[^\u0980-\u09FF\s]", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return text.split()


def load_pairs(path: Path) -> tuple[list[tuple[str, str]], int]:
    pairs: list[tuple[str, str]] = []
    malformed = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                malformed += 1
                continue

            english = parts[0].strip()
            bengali = parts[1].strip()
            if english and bengali:
                pairs.append((english, bengali))
            else:
                malformed += 1

    return pairs, malformed


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * pct
    floor = math.floor(k)
    ceil = math.ceil(k)
    if floor == ceil:
        return float(sorted_values[int(k)])
    lower = sorted_values[floor] * (ceil - k)
    upper = sorted_values[ceil] * (k - floor)
    return float(lower + upper)


def describe_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "p25": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
        }

    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean,
        "median": percentile(lengths, 0.50),
        "std": math.sqrt(variance),
        "p25": percentile(lengths, 0.25),
        "p75": percentile(lengths, 0.75),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
    }


def pearson(xs: list[int], ys: list[int]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else 0.0


def length_buckets(lengths: list[int]) -> Counter[str]:
    buckets: Counter[str] = Counter()
    for length in lengths:
        if length <= 3:
            buckets["1-3 tokens"] += 1
        elif length <= 7:
            buckets["4-7 tokens"] += 1
        elif length <= 12:
            buckets["8-12 tokens"] += 1
        elif length <= 20:
            buckets["13-20 tokens"] += 1
        else:
            buckets["21+ tokens"] += 1
    return buckets


def fmt(value: float) -> str:
    return f"{value:,.2f}"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_token_vocab_chart(
    path: Path,
    english_token_count: int,
    bengali_token_count: int,
    english_vocab_count: int,
    bengali_vocab_count: int,
) -> None:
    width = 900
    height = 520
    margin_left = 120
    margin_bottom = 90
    chart_top = 80
    chart_height = height - chart_top - margin_bottom
    max_value = max(english_token_count, bengali_token_count)

    bars = [
        ("English tokens", english_token_count, "#4e79a7"),
        ("Bengali tokens", bengali_token_count, "#e15759"),
        ("English unique words", english_vocab_count, "#59a14f"),
        ("Bengali unique words", bengali_vocab_count, "#f28e2b"),
    ]

    bar_width = 105
    gap = 70
    x0 = margin_left

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="36" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Token and Vocabulary Counts</text>',
        '<text x="450" y="62" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">After lowercasing, punctuation removal, and word tokenization</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 60}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{chart_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
    ]

    for tick in range(0, 5):
        value = max_value * tick / 4
        y = height - margin_bottom - (value / max_value) * chart_height
        svg.append(f'<line x1="{margin_left - 6}" y1="{y:.1f}" x2="{width - 60}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{value:,.0f}</text>')

    for idx, (label, value, color) in enumerate(bars):
        x = x0 + idx * (bar_width + gap)
        bar_height = (value / max_value) * chart_height
        y = height - margin_bottom - bar_height
        svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
        svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{value:,}</text>')
        svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{svg_escape(label)}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_sentence_length_chart(
    path: Path,
    english_lengths: list[int],
    bengali_lengths: list[int],
) -> None:
    width = 900
    height = 520
    margin_left = 90
    margin_bottom = 85
    chart_top = 80
    chart_height = height - chart_top - margin_bottom
    buckets = ["1-3 tokens", "4-7 tokens", "8-12 tokens", "13-20 tokens", "21+ tokens"]
    eng_buckets = length_buckets(english_lengths)
    ben_buckets = length_buckets(bengali_lengths)
    max_value = max(max(eng_buckets.values()), max(ben_buckets.values()))

    group_width = 140
    bar_width = 46
    gap = 24
    x0 = 135

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="36" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Sentence Length Distribution</text>',
        '<text x="450" y="62" text-anchor="middle" font-family="Arial" font-size="13" fill="#555">Number of sentence pairs by tokenized sentence length</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 55}" y2="{height - margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{chart_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333"/>',
        '<rect x="665" y="88" width="16" height="16" fill="#4e79a7"/><text x="688" y="101" font-family="Arial" font-size="13">English</text>',
        '<rect x="750" y="88" width="16" height="16" fill="#e15759"/><text x="773" y="101" font-family="Arial" font-size="13">Bengali</text>',
    ]

    for tick in range(0, 5):
        value = max_value * tick / 4
        y = height - margin_bottom - (value / max_value) * chart_height
        svg.append(f'<line x1="{margin_left - 6}" y1="{y:.1f}" x2="{width - 55}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        svg.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{value:,.0f}</text>')

    for idx, bucket in enumerate(buckets):
        group_x = x0 + idx * group_width
        for offset, value, color in [
            (0, eng_buckets[bucket], "#4e79a7"),
            (bar_width + gap, ben_buckets[bucket], "#e15759"),
        ]:
            bar_height = (value / max_value) * chart_height
            y = height - margin_bottom - bar_height
            x = group_x + offset
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
            svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:,}</text>')
        svg.append(f'<text x="{group_x + bar_width + gap / 2:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{bucket}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def make_markdown(
    total_pairs: int,
    malformed: int,
    english_tokens: list[str],
    bengali_tokens: list[str],
    english_counter: Counter[str],
    bengali_counter: Counter[str],
    english_lengths: list[int],
    bengali_lengths: list[int],
) -> str:
    eng_stats = describe_lengths(english_lengths)
    ben_stats = describe_lengths(bengali_lengths)
    eng_vocab = set(english_tokens)
    ben_vocab = set(bengali_tokens)
    eng_hapax = sum(1 for count in english_counter.values() if count == 1)
    ben_hapax = sum(1 for count in bengali_counter.values() if count == 1)
    length_corr = pearson(english_lengths, bengali_lengths)

    eng_buckets = length_buckets(english_lengths)
    ben_buckets = length_buckets(bengali_lengths)

    lines = [
        "# English-Bengali Corpus Analysis",
        "",
        "## Dataset Overview",
        "",
        f"- Source file: `{DATA_PATH.name}`",
        f"- Valid sentence pairs: {total_pairs:,}",
        f"- Malformed or empty lines skipped: {malformed:,}",
        f"- Total English tokens after cleaning: {len(english_tokens):,}",
        f"- Total Bengali tokens after cleaning: {len(bengali_tokens):,}",
        f"- Unique English words: {len(eng_vocab):,}",
        f"- Unique Bengali words: {len(ben_vocab):,}",
        f"- English words appearing once: {eng_hapax:,} ({eng_hapax / len(eng_vocab) * 100:.1f}% of English vocabulary)",
        f"- Bengali words appearing once: {ben_hapax:,} ({ben_hapax / len(ben_vocab) * 100:.1f}% of Bengali vocabulary)",
        f"- Pearson correlation between English and Bengali sentence lengths: {length_corr:.3f}",
        "",
        "## Sentence Length Summary",
        "",
        "| Language | Mean | Median | Std. Dev. | Min | Max | 75th pct. | 90th pct. | 95th pct. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| English | {fmt(eng_stats['mean'])} | {fmt(eng_stats['median'])} | "
            f"{fmt(eng_stats['std'])} | {eng_stats['min']:.0f} | {eng_stats['max']:.0f} | "
            f"{fmt(eng_stats['p75'])} | {fmt(eng_stats['p90'])} | {fmt(eng_stats['p95'])} |"
        ),
        (
            f"| Bengali | {fmt(ben_stats['mean'])} | {fmt(ben_stats['median'])} | "
            f"{fmt(ben_stats['std'])} | {ben_stats['min']:.0f} | {ben_stats['max']:.0f} | "
            f"{fmt(ben_stats['p75'])} | {fmt(ben_stats['p90'])} | {fmt(ben_stats['p95'])} |"
        ),
        "",
        "## Sentence Length Buckets",
        "",
        "| Bucket | English Sentences | Bengali Sentences |",
        "|---|---:|---:|",
    ]

    for bucket in ["1-3 tokens", "4-7 tokens", "8-12 tokens", "13-20 tokens", "21+ tokens"]:
        lines.append(f"| {bucket} | {eng_buckets[bucket]:,} | {ben_buckets[bucket]:,} |")

    lines.extend(
        [
            "",
            "## Most Frequent English Words",
            "",
            "| Rank | Word | Count |",
            "|---:|---|---:|",
        ]
    )
    for rank, (word, count) in enumerate(english_counter.most_common(15), start=1):
        lines.append(f"| {rank} | {word} | {count:,} |")

    lines.extend(
        [
            "",
            "## Most Frequent Bengali Words",
            "",
            "| Rank | Word | Count |",
            "|---:|---|---:|",
        ]
    )
    for rank, (word, count) in enumerate(bengali_counter.most_common(15), start=1):
        lines.append(f"| {rank} | {word} | {count:,} |")

    lines.extend(
        [
            "",
            "## Report Interpretation Notes",
            "",
            "- The dataset is small enough for a compact LSTM-based sequence-to-sequence model, but the vocabulary contains many rare words.",
            "- Bengali has a larger vocabulary than English in this corpus, which is expected because Bengali morphology creates more surface forms.",
            "- The sentence length percentiles help justify maximum sequence lengths and padding choices.",
            "- The length correlation shows whether English and Bengali sentence complexity tends to grow together across aligned pairs.",
            "- The rare-word counts support using an out-of-vocabulary token and vocabulary caps during tokenizer construction.",
        ]
    )

    return "\n".join(lines) + "\n"


def print_report_stats(
    total_pairs: int,
    english_tokens: list[str],
    bengali_tokens: list[str],
    english_lengths: list[int],
    bengali_lengths: list[int],
) -> None:
    english_vocab = set(english_tokens)
    bengali_vocab = set(bengali_tokens)
    eng_stats = describe_lengths(english_lengths)
    ben_stats = describe_lengths(bengali_lengths)

    print("=" * 72)
    print("TASK 2.1 DATA PREPARATION SUMMARY")
    print("=" * 72)
    print(f"Total sequence pairs: {total_pairs:,}")
    print(f"Total English tokens: {len(english_tokens):,}")
    print(f"Total Bengali tokens: {len(bengali_tokens):,}")
    print(f"Total unique English words: {len(english_vocab):,}")
    print(f"Total unique Bengali words: {len(bengali_vocab):,}")
    print(f"Average English sentence length: {eng_stats['mean']:.2f} words")
    print(f"Average Bengali sentence length: {ben_stats['mean']:.2f} words")
    print(f"Maximum English sentence length: {eng_stats['max']:.0f} words")
    print(f"Maximum Bengali sentence length: {ben_stats['max']:.0f} words")
    print("=" * 72)
    print("\nTop English words after preprocessing:")
    for rank, (word, count) in enumerate(Counter(english_tokens).most_common(TOP_N), start=1):
        print(f"{rank:>2}. {word:<15} {count:>5,}")
    print("\nTop Bengali words after preprocessing:")
    for rank, (word, count) in enumerate(Counter(bengali_tokens).most_common(TOP_N), start=1):
        print(f"{rank:>2}. {word:<15} {count:>5,}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs, malformed = load_pairs(DATA_PATH)
    english_clean = [clean_english(english) for english, _ in pairs]
    bengali_clean = [clean_bengali(bengali) for _, bengali in pairs]

    cleaned_pairs = [
        (english, bengali)
        for english, bengali in zip(english_clean, bengali_clean)
        if english and bengali
    ]

    english_tokenized = [tokenize(english) for english, _ in cleaned_pairs]
    bengali_tokenized = [tokenize(bengali) for _, bengali in cleaned_pairs]

    english_tokens = [token for sentence in english_tokenized for token in sentence]
    bengali_tokens = [token for sentence in bengali_tokenized for token in sentence]
    english_lengths = [len(sentence) for sentence in english_tokenized]
    bengali_lengths = [len(sentence) for sentence in bengali_tokenized]

    english_counter = Counter(english_tokens)
    bengali_counter = Counter(bengali_tokens)

    print_report_stats(
        total_pairs=len(cleaned_pairs),
        english_tokens=english_tokens,
        bengali_tokens=bengali_tokens,
        english_lengths=english_lengths,
        bengali_lengths=bengali_lengths,
    )

    token_vocab_chart = OUT_DIR / "word_based_token_vocab_counts.svg"
    sentence_length_chart = OUT_DIR / "word_based_sentence_length_distribution.svg"
    write_token_vocab_chart(
        token_vocab_chart,
        english_token_count=len(english_tokens),
        bengali_token_count=len(bengali_tokens),
        english_vocab_count=len(english_counter),
        bengali_vocab_count=len(bengali_counter),
    )
    write_sentence_length_chart(sentence_length_chart, english_lengths, bengali_lengths)

    print("\nCharts written:")
    print(f"- {token_vocab_chart}")
    print(f"- {sentence_length_chart}")


if __name__ == "__main__":
    main()
