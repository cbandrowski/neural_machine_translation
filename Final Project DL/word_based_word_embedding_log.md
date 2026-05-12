# Word-Based Word Embedding Log

Tokenization: word-level / word-based.

## Setup

- Separate CBOW-style embedding models trained for English and Bengali.
- Embedding dimension: 50
- Context window size: 2
- Epochs: 150
- Fixed random seed: 42

## Fresh English Nearest-Neighbor Examples

| Query | Nearest words from rerun |
| --- | --- |
| can | could, until, must, wont, away |
| what | how, much, which, pretty, men |
| want | need, wanted, youve, difficult, silent |
| know | wondering, care, knew, lie, tell |
| like | seem, want, forget, happy, whether |
| you | they, christmas, we, everyone, sleeping |

## Fresh Bengali Nearest-Neighbor Examples

| Query | Meaning | Nearest words from rerun |
| --- | --- | --- |
| আপনি | you formal | তুমি, লাল, ওনারা, সত্যিই, তুই |
| তুমি | you informal | আপনি, তুই, খাওয়ার, বলেছেন, করেছিলাম |
| এটা | this / it | সাহায্যের, চিৎকার, ওটা, টিকিট, সহ্য |
| চাই | want | আজকেই, কতক্ষণ, চান, দিল, চাও |
| সে | he / she | টুপি, যাস, দেখেছেন, শত্রু, সম্পুর্ণ |
| আমি | I | শক্তিশালী, কাগজ, স্কুলে, টেলিভিশন, সবাই |

## Charts

- `word_based_eng_pca_all_words.png`
- `word_based_eng_pca_top_words.png`
- `word_based_eng_nearest_neighbours.png`
- `word_based_ben_pca_all_words.png`
- `word_based_ben_pca_top_words.png`
- `word_based_ben_nearest_neighbours.png`
