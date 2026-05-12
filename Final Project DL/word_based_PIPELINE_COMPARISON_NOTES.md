# Word-Based Pipeline Comparison Notes

Use this document to compare our word-level tokenization pipeline against a
subword-level tokenization pipeline.

## Dataset And Preprocessing

- Dataset file: `ben-eng/ben.txt`
- Total sentence pairs: 7,030
- Main translation direction: Bengali -> English
- Tokenization: word-level
- English text is lowercased.
- Punctuation is removed.
- Separate Bengali and English word dictionaries are created.
- English decoder targets use `<start>` and `<end>` tokens.

## Dataset Statistics

| Statistic | Value |
| --- | ---: |
| Total sequence pairs | 7,030 |
| Total English tokens | 34,816 |
| Total Bengali tokens | 32,860 |
| Unique English words | 2,770 |
| Unique Bengali words | 4,632 |
| Average English sentence length | 4.95 words |
| Average Bengali sentence length | 4.67 words |
| Maximum English sentence length | 20 words |
| Maximum Bengali sentence length | 18 words |

## Split

- Train/test split: 70% train, 30% test
- Bengali -> English models: length-stratified split
- Validation split: 10% of the training portion

| Split | Sentence Pairs |
| --- | ---: |
| Train | 4,428 |
| Validation | 493 |
| Test | 2,109 |

## Shared Bengali -> English Hyperparameters

| Hyperparameter | Value |
| --- | ---: |
| Encoder | Bidirectional LSTM |
| Encoder units | 128 forward + 128 backward |
| Decoder | LSTM |
| Decoder units | 256 |
| Embedding dimension | 128 |
| Batch size | 64 |
| Max epochs | 50 |
| Dropout | 0.25 |
| Recurrent dropout | 0.10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Gradient clipping | clipnorm = 1.0 |
| Beam width | 3 |

## Fresh Translation Results

| Model | Direction | Greedy BLEU-4 | Beam BLEU-4 |
| --- | --- | ---: | ---: |
| Plain encoder-decoder LSTM | English -> Bengali | 0.0695 | n/a |
| Attention LSTM | Bengali -> English | 0.1495 | 0.1605 |
| No-attention LSTM baseline | Bengali -> English | 0.1476 | 0.1556 |

## Current Best Fresh Word-Based Result

- Model: attention Bengali -> English seq2seq LSTM
- Tokenization: word-level
- Embedding dimension: 128
- Decoder: beam search, width 3
- Corpus BLEU-4: 0.1605

## Notes For Subword Comparison

- Keep the same dataset.
- Keep Bengali -> English as the main direction.
- Use the same 70/30 length-stratified split if possible.
- Use the same train/validation/test counts if possible.
- Compare both greedy and beam decoding.
- Use corpus BLEU-4 on the same test set.
