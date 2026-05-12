# Architecture Comparison Audit

Purpose: keep the word-based and subword-based experiments aligned for an
apples-to-apples comparison.

## Directory Structure

- Word-based helper/source folder: `word_based/`
- Subword-based helper/source folder: `subword_based/`
- Word-based outputs: `output/word_based/`
- Subword-based outputs: `output/subword_based/`

## Clean Comparison Pair

The best current word-vs-subword comparison is:

- Word-based attention model: `seq2seq_bengali.py`
- Subword-based attention model: `classDS/seq2seq_bengali_subword.py`

Both are Bengali -> English seq2seq LSTM models with additive attention.

## Settings Aligned

| Setting | Word-Based | Subword-Based | Status |
| --- | --- | --- | --- |
| Direction | Bengali -> English | Bengali -> English | Aligned |
| Split | length-stratified 70/30 | length-stratified 70/30 | Aligned |
| Validation | 10% of training portion | 10% of training portion | Aligned |
| Encoder | Bidirectional LSTM | Bidirectional LSTM | Aligned |
| Decoder | LSTM | LSTM | Aligned |
| Attention | Additive attention | Additive attention | Aligned |
| LSTM units | 256 | 256 | Aligned |
| Embedding dimension | 128 | 128 | Aligned |
| Batch size | 64 | 64 | Aligned |
| Max epochs | 50 | 50 | Aligned |
| Dropout | 0.25 | 0.25 | Aligned |
| Recurrent dropout | 0.10 | 0.10 | Aligned |
| Optimizer | Adam | Adam | Aligned |
| Learning rate | 0.001 | 0.001 | Aligned |
| Gradient clipping | clipnorm = 1.0 | clipnorm = 1.0 | Aligned |
| Beam width | 3 | 3 | Aligned |
| Output folder | `output/word_based/` | `output/subword_based/` | Aligned |

## Discrepancies Found And Fixed

1. Subword embedding dimension was `96`, while word-based attention used `128`.
   - Fixed: subword now uses `EMBEDDING_DIM = 128`.

2. Subword max epochs were `60`, while word-based attention used `50`.
   - Fixed: subword now uses `EPOCHS = 50`.

3. Subword dropout was `0.20`, while word-based attention used `0.25`.
   - Fixed: subword now uses `DROPOUT = 0.25`.

4. Subword recurrent dropout was `0.0`, while word-based attention used `0.10`.
   - Fixed: subword now uses `RECURRENT_DROPOUT = 0.10`.

5. Subword beam width was `4`, while word-based attention used `3`.
   - Fixed: subword now uses `BEAM_WIDTH = 3`.

6. Output files were mixed across root folders and script folders.
   - Fixed going forward:
     - word-based scripts write to `output/word_based/`
     - subword script writes to `output/subword_based/`

7. Subword script printed results but did not save a markdown experiment log.
   - Fixed: subword now writes `output/subword_based/subword_based_SEQ2SEQ_EXPERIMENT_LOG.md`.

8. Subword training/evaluation used multiple English references for some
   Bengali sources, while word-based used one target/reference.
   - Fixed: subword now chooses one canonical English target per Bengali source
     and evaluates BLEU against that single reference.

## Remaining Differences To Be Aware Of

These are intentional or still need a decision:

1. Tokenization differs by design.
   - Word-based: Keras word tokenizer.
   - Subword-based: WordPiece tokenizer.

2. Vocabulary sizes differ by design.
   - Word-based caps: Bengali 8,000, English 5,000.
   - Subword-based caps: Bengali 2,500, English 2,000.

3. Subword beam search uses a length penalty with alpha `0.7`.
   - Word-based beam search uses average log probability normalization.
   - These are similar but not identical. For strict comparison, decoding
     normalization should be standardized.

4. There is no subword no-attention baseline yet.
   - Word-based has both attention and no-attention.
   - Subword currently has attention only.

7. `seq2seq_english.py` is not part of the clean word-vs-subword comparison.
   - Direction is English -> Bengali.
   - Encoder is not bidirectional.
   - Embedding dimension is 64.
   - Split is random 70/30 without the same validation structure.

## Recommended Next Step

For the cleanest final comparison:

1. Run `seq2seq_bengali.py` for word-based attention.
2. Run `classDS/seq2seq_bengali_subword.py` for subword-based attention.
3. Compare BLEU, example translations, training curves, BLEU histograms, and
   attention heatmaps.
4. Optionally create a subword no-attention baseline only if the report needs
   attention-vs-no-attention under both tokenization methods.
