# Word-Based Data Analysis Log

Tokenization: word-level / word-based.

## Dataset

- Dataset file: `ben-eng/ben.txt`
- Language pair: English-Bengali
- Main translation direction for final models: Bengali -> English

## Preprocessing

- English text converted to lowercase.
- Punctuation removed.
- Sentences tokenized into words.
- Separate word dictionaries created for Bengali and English.

## Statistics

| Statistic | Value |
| --- | ---: |
| Total sequence pairs | 7,030 |
| Total English tokens | 34,816 |
| Total Bengali tokens | 32,860 |
| Total unique English words | 2,770 |
| Total unique Bengali words | 4,632 |
| Average English sentence length | 4.95 words |
| Average Bengali sentence length | 4.67 words |
| Maximum English sentence length | 20 words |
| Maximum Bengali sentence length | 18 words |

## Charts

- `output/data_analysis/word_based_token_vocab_counts.svg`
- `output/data_analysis/word_based_sentence_length_distribution.svg`
