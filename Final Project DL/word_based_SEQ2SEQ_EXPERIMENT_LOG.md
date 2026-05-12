# Word-Based Seq2Seq Experiment Log

Tokenization: word-level / word-based.

Fresh run after clearing previous generated result artifacts.

## English to Bengali, plain encoder-decoder LSTM

- Script: `seq2seq_english.py`
- Tokenization: word-based
- Direction: English -> Bengali
- Split: 70% train, 30% test
- Model: single-layer LSTM encoder + single-layer LSTM decoder
- Embedding dimension: 64
- Training pairs: 4,921
- Test pairs: 2,109
- English vocabulary (encoder): 2,340
- Bengali vocabulary (decoder): 4,330
- Decoder: greedy
- Corpus BLEU-4: 0.0695
- Mean sentence BLEU: 0.0958
- Median sentence BLEU: 0.0574

## Bengali to English, attention LSTM with beam-search comparison

- Script: `seq2seq_bengali.py`
- Tokenization: word-based
- Direction: Bengali -> English
- Split: length_stratified; 70% train, 30% test; 10% of training portion used for validation
- Model: bidirectional LSTM encoder + additive attention + LSTM decoder
- Embedding dimension: 128
- Training pairs: 4,428
- Validation pairs: 493
- Test pairs: 2,109
- Bengali vocabulary (encoder): 4,118
- English vocabulary (decoder): 2,221
- Greedy corpus BLEU-4: 0.1495
- Greedy mean sentence BLEU: 0.1593
- Greedy median sentence BLEU: 0.0803
- Beam width: 3
- Beam corpus BLEU-4: 0.1605
- Beam mean sentence BLEU: 0.1678
- Beam median sentence BLEU: 0.0803
- Attention heatmap: `word_based_ben_seq2seq_attention_heatmap.png`

## Bengali to English, no-attention LSTM baseline

- Script: `seq2seq_bengali_no_attention.py`
- Tokenization: word-based
- Direction: Bengali -> English
- Split: length_stratified; 70% train, 30% test; 10% of training portion used for validation
- Model: bidirectional LSTM encoder + LSTM decoder, no attention
- Embedding dimension: 128
- Training pairs: 4,428
- Validation pairs: 493
- Test pairs: 2,109
- Bengali vocabulary (encoder): 4,118
- English vocabulary (decoder): 2,221
- Greedy corpus BLEU-4: 0.1476
- Greedy mean sentence BLEU: 0.1578
- Greedy median sentence BLEU: 0.0803
- Beam width: 3
- Beam corpus BLEU-4: 0.1556
- Beam mean sentence BLEU: 0.1662
- Beam median sentence BLEU: 0.0803
