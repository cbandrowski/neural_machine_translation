# Word-Based Output Index

Fresh word-level outputs generated after clearing previous result artifacts.

## 2.1 Data Preparation

Logs:

- `word_based_data_analysis_log.md`

Charts:

- `output/data_analysis/word_based_token_vocab_counts.svg`
- `output/data_analysis/word_based_sentence_length_distribution.svg`

## 2.2 Word Embedding

Logs:

- `word_based_word_embedding_log.md`

Charts:

- `word_based_eng_pca_all_words.png`
- `word_based_eng_pca_top_words.png`
- `word_based_eng_nearest_neighbours.png`
- `word_based_ben_pca_all_words.png`
- `word_based_ben_pca_top_words.png`
- `word_based_ben_nearest_neighbours.png`

## 2.3 Translation With Seq2Seq LSTM

Logs:

- `word_based_SEQ2SEQ_EXPERIMENT_LOG.md`
- `word_based_PIPELINE_COMPARISON_NOTES.md`

Charts:

- `word_based_seq2seq_eng_to_ben_training_curve.png`
- `word_based_seq2seq_eng_to_ben_bleu_histogram.png`
- `word_based_ben_seq2seq_attention_training_curve.png`
- `word_based_ben_seq2seq_attention_bleu_histogram.png`
- `word_based_ben_seq2seq_attention_heatmap.png`
- `word_based_ben_seq2seq_no_attention_training_curve.png`
- `word_based_ben_seq2seq_no_attention_bleu_histogram.png`

## Fresh BLEU Summary

| Model | Direction | Greedy BLEU-4 | Beam BLEU-4 |
| --- | --- | ---: | ---: |
| Plain encoder-decoder LSTM | English -> Bengali | 0.0695 | n/a |
| Attention LSTM | Bengali -> English | 0.1495 | 0.1605 |
| No-attention LSTM baseline | Bengali -> English | 0.1476 | 0.1556 |

Current best fresh word-based result: attention Bengali -> English with beam
search, BLEU-4 = 0.1605.
