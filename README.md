# Autoregressive next-word generation on WikiText-2

**Foundations of Generative AI (ICT4403), Assignment 2**  
**Student:** Aditya Singh (230953458)

## Problem and objective

Train and compare Simple RNN, Bidirectional RNN, LSTM, and GRU models for next-word prediction under matched conditions. Each model receives up to 20 preceding tokens and predicts the next token. For generation, a sampled token is appended to the context and the process is repeated. The experiment compares held-out loss, perplexity, accuracy, training time, and the coherence, diversity, contextual consistency, and repetition of generated text at three temperatures.

## Dataset and source

The dataset is [Salesforce WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1), drawn from Wikipedia articles. The **official train, validation, and test split files** are included in `data/`, and the script downloads them from that source if they are absent. The dataset card lists CC BY-SA 3.0 and GFDL licensing. Source attribution: Salesforce WikiText dataset and its Wikipedia contributors.

The raw split files contain 36,718 training rows, 3,760 validation rows, and 4,358 test rows, including blank lines and headings. For a reproducible laptop-scale experiment, this run uses the first 240,000 processed tokens of the official training split and the first 60,000 processed tokens from **each** official held-out split. It does not mix the supplied splits.

### Preprocessing and input-target construction

The script discards empty lines and standalone article/section heading lines, joins the remaining text in each split, converts WikiText `@-@`, `@,@`, and `@.@` placeholders to punctuation, and lowercases it. A regular expression extracts alphabetic words (allowing an internal ASCII apostrophe), numbers, and individual punctuation marks. This is a **word-plus-punctuation** tokenizer; the results are not directly comparable to published WikiText-2 benchmark perplexities that use other tokenization or the full corpus.

The vocabulary contains the 7,998 most common **training** tokens plus `<pad>` and `<unk>` (8,000 entries). Tokens outside this vocabulary become `<unk>`. Unknown rates are 5.97% in training, 13.04% in validation, and 12.53% in test. The higher held-out rates are an important limitation when interpreting accuracy and generated text.

Each input is a 20-token context and its target is the immediately following token. Windows start every four tokens, giving 59,995 training examples and 14,995 examples in each held-out split. Windows cannot cross split boundaries. Nearby windows within a split overlap, so they are correlated observations.

## Models and controlled conditions

Every model uses a 96-dimensional embedding, one recurrent layer with 128 units per direction, a linear output head, batch size 128, Adam with learning rate 0.002, gradient clipping at 1.0, the same shuffled example order, and up to four epochs. The random seed is 4403. Cross-entropy is optimized. For each architecture, the checkpoint with **lowest validation loss** among the four epochs is selected for test evaluation and generation; the same selection rule is used for all models.

The Bidirectional RNN processes **only the already observed context window** in both directions. The next target token is never part of that window, so generation still proceeds one token at a time without target leakage. The architectures have different parameter counts because their cells and directionality differ. No architecture-specific hyperparameter tuning was performed.

## How to run

1. Install Python 3.10 or later and run `pip install -r requirements.txt`.
2. Run `python train.py` from this directory. The script uses CUDA if available, otherwise CPU.
3. Inspect `results/dataset.json`, `results/metrics.json`, `results/training_curves.png`, and `results/samples.md`. The four `.pt` files are the validation-selected model checkpoints; `results/vocabulary.json` stores the learned vocabulary.

The reported run used PyTorch 2.12.0 and an NVIDIA GeForce RTX 5060 Laptop GPU. Training time is wall-clock time for all four epochs and validation passes after device warm-up. It excludes corpus preparation, final test evaluation, checkpoint saving, and generation. Timings depend on hardware and system load.

## Quantitative evaluation

| Model | Parameters | Selected epoch | Validation loss | Validation accuracy | Test loss | Test perplexity | Test accuracy | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple RNN | 1,828,928 | 2 | 5.540 | 17.19% | 5.640 | 281.55 | 16.74% | 17.52 s |
| Bidirectional RNN | 2,881,856 | 1 | 5.645 | 15.53% | 5.722 | 305.45 | 15.48% | 19.68 s |
| LSTM | 1,915,712 | 3 | 5.473 | **18.77%** | 5.547 | 256.47 | **18.40%** | 17.63 s |
| GRU | 1,886,784 | 2 | **5.437** | 18.68% | **5.526** | **251.18** | 18.26% | 17.95 s |

Loss is mean next-token cross-entropy; perplexity is `exp(loss)`; accuracy is exact top-1 next-token accuracy. The **GRU** has the best validation and test loss and perplexity. The **LSTM** has marginally better accuracy. Accuracy rewards only the top-ranked token, while loss measures the probability assigned to the correct token. The differences between LSTM and GRU are small in one run, so they do not establish a universal ranking.

Overfitting appears quickly: for the Bidirectional RNN, training loss falls to 3.271 by epoch 4, while validation loss rises from 5.645 at epoch 1 to 6.036. GRU validation loss is lowest at epoch 2 and rises thereafter. Selecting checkpoints by validation loss prevents later overfitted weights from determining test results.

## Generated-text comparison

`results/samples.md` contains **all 36 continuations**: four models × three shared seeds (`the history of`, `in the united states`, `the first season of`) × temperatures 0.5, 1.0, and 1.5. Each continuation adds 50 tokens. `results/samples.json` includes these samples and per-sample diversity and adjacent-repeat measurements.

At temperature 0.5, the models often reproduce encyclopedia-like constructions such as “the state of” and “the first ... of.” This helps local grammatical shape but causes repetitive scaffolding. The Bidirectional RNN repeats high-frequency connectors particularly heavily; its weaker held-out probability metrics agree with its less convincing samples. The Simple RNN also creates plausible short fragments but shifts topics abruptly. LSTM and GRU generally produce more natural local phrasing, though neither maintains a factual, consistent article over 50 tokens. Some samples combine unrelated subjects or put numbers and names in implausible places. The GRU has the strongest overall probability estimates, while LSTM is a close alternative for short-form readability and top-1 accuracy.

## Sampling-temperature analysis

Generation samples from `softmax(logits / temperature)`. The same three seeds and a fixed pseudo-random seed per prompt are used across models and temperatures.

| Temperature | Mean distinct-token ratio | Mean adjacent-repeat rate | Qualitative effect |
|---:|---:|---:|---|
| 0.5 | 0.523 | 0.019 | More predictable local phrases, but more repeated formulaic wording. |
| 1.0 | 0.848 | 0.000 | Better variety while retaining some encyclopedia-like phrasing; topics still drift. |
| 1.5 | 0.968 | 0.000 | Very diverse words but frequent semantic jumps and weaker grammar. |

These means cover 12 samples at each temperature. Distinct-token ratio is unique generated tokens divided by 50; adjacent-repeat rate counts immediately repeated identical tokens. Adjacent-repeat rate does **not** measure repeated multiword phrases, which remain visible in the samples. Temperature **1.0** offers the most useful balance in this experiment. None of the three values guarantees coherent long-form output from these small recurrent models.

## Conclusion and limits

For this WikiText-2 subset and training budget, **GRU is the most effective autoregressive generator** by held-out loss and perplexity, with accuracy nearly equal to LSTM and similar training time. The LSTM is a defensible choice when top-1 accuracy is the main criterion. The Bidirectional RNN has more parameters yet overfits earliest and ranks last on held-out metrics.

This is a controlled educational comparison, not a full WikiText-2 benchmark. It trains on a fraction of the official corpus, evaluates on truncated official held-out splits, uses a custom tokenizer and `<unk>` mapping, and runs one random seed. The source dataset’s topic shifts and the 20-token context limit also restrict sustained coherence. Longer training with early stopping, broader hyperparameter tuning, a larger vocabulary, and multiple random seeds would strengthen the conclusions.
