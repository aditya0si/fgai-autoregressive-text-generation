# Autoregressive next-word generation with recurrent networks

**Foundations of Generative AI (ICT4403), Assignment 2**  
**Student:** Aditya Singh (230953458)

## Problem and objective

Train and compare Simple RNN, Bidirectional RNN, LSTM, and GRU next-word models under matched experimental settings. Given a sequence of up to 20 previous tokens, each model predicts a probability distribution over the next token. Generated text is produced one token at a time by feeding each sampled token back into the context window.

## Dataset and source

The dataset is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), a collection of Shakespeare play text including dialogue, speaker labels, punctuation, and stage directions. The [exact source file](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) is included at `data/tinyshakespeare.txt` so the experiment can be rerun without a download. The file has 1,115,394 characters. This experiment uses its first 120,000 word/punctuation tokens to keep all four training runs practical on a laptop GPU.

Text is lowercased, then split into alphabetic words with optional internal apostrophes, numbers, and individual punctuation marks. Newlines and capitalization are not retained. The 5,000-token vocabulary is built **only from training data**; it includes `<pad>` and `<unk>`. Unknown-token rates are 2.42% in training, 9.45% in validation, and 10.41% in test. The higher held-out rates reflect new names and less common words encountered later in the corpus.

Tokens are split chronologically: 96,000 training, 12,000 validation, and 12,000 test. Windows never cross split boundaries. A 20-token input is paired with its immediately following target token; windows begin every four tokens. This gives 23,995 training and 2,995 examples in each held-out split. Nearby examples within a split overlap, so the number of statistically independent observations is smaller than the window count.

## Models and controlled conditions

All models have a 96-dimensional embedding, one recurrent layer with 128 hidden units per direction, a linear vocabulary head, batch size 128, Adam learning rate 0.002, gradient clipping at 1.0, three epochs, the same shuffled training order, and the same train/validation/test windows. The random seed is 4403. Cross-entropy is the training objective. The bidirectional RNN reads the **past context window only** in both directions; the target word is never supplied to it. It is therefore valid for sequential next-word generation, although its backward pass adds computation and parameters.

Parameter counts vary because the recurrent cells and bidirectional encoder differ: Simple RNN 1,153,928; Bidirectional RNN 1,822,856; LSTM 1,240,712; GRU 1,211,784. No hyperparameter search or early stopping was performed. Model checkpoints represent epoch 3 for each architecture.

## How to run

1. Install Python 3.10 or later and the packages in `requirements.txt` (for example, `pip install -r requirements.txt`).
2. Run `python train.py` from this directory. A CUDA GPU is used when available; otherwise the script uses CPU.
3. Review `results/metrics.json`, `results/training_curves.png`, and `results/samples.md`. `results/dataset.json` records the preprocessing configuration and split sizes. The `.pt` files are trained model checkpoints.

The reported run used PyTorch 2.12.0 with an NVIDIA GeForce RTX 5060 Laptop GPU. Training time is wall-clock time for the three training epochs and validation passes after a kernel warm-up; it excludes data preparation, final test evaluation, checkpoint saving, and sample generation. Runtime will vary by hardware and system load.

## Quantitative results

| Model | Validation loss | Validation accuracy | Test loss | Test perplexity | Test accuracy | Training time |
|---|---:|---:|---:|---:|---:|---:|
| Simple RNN | 5.423 | 13.89% | 5.426 | 227.14 | 13.52% | 8.09 s |
| Bidirectional RNN | 5.587 | 12.35% | 5.577 | 264.30 | 12.25% | 7.56 s |
| LSTM | 5.407 | **14.32%** | 5.422 | 226.43 | **13.69%** | **6.21 s** |
| GRU | **5.369** | 13.96% | **5.376** | **216.10** | 13.62% | 6.57 s |

Loss is mean cross-entropy per next token; perplexity is `exp(loss)`; accuracy is exact top-1 next-token accuracy. The LSTM has the best accuracy, but the GRU has the lowest held-out loss and perplexity. Accuracy captures only whether the most probable word is right, while loss also evaluates the probability assigned to the correct word. The bidirectional model overfits: its training loss reaches 4.223 while validation loss rises from 5.495 in epoch 2 to 5.587 in epoch 3. This is one short run, so the small differences between Simple RNN, LSTM, and GRU accuracy should not be treated as statistically conclusive.

## Generated-text comparison

`results/samples.md` includes **all 36 continuations**: four models × three shared seeds × three temperatures, 50 generated tokens each. All models can reproduce features of play dialogue, such as speaker names and courtly language. None maintains a consistent dramatic scene for 50 words. At temperature 0.5, the Simple RNN visibly falls into repeated dashes on the `the king of england` seed, while the GRU often repeats phrases such as “my lord.” The bidirectional RNN produces some locally plausible phrases but has weaker measured next-word performance and often drifts into disconnected dialogue. The LSTM and GRU generally produce the most readable short phrases. The GRU is the best overall choice here because its test loss and perplexity are lowest, its test accuracy is near the LSTM, and its samples retain recognizable dialogue patterns at moderate temperature.

## Sampling temperature

Sampling uses `softmax(logits / temperature)` with temperatures **0.5, 1.0, and 1.5**. Every model is prompted with `the king of england`, `i shall tell you`, and `what is the matter`. For each prompt, the pseudo-random generator is reset to the same seed across temperature settings; the output still changes because the probability distributions change.

| Temperature | Mean distinct-token ratio | Mean adjacent-repeat rate | Observed effect |
|---:|---:|---:|---|
| 0.5 | 0.612 | 0.024 | More common words and locally plausible phrases, but noticeable loops and formulaic repetition. |
| 1.0 | 0.837 | 0.005 | Better balance of variety and local readability; context still drifts. |
| 1.5 | 0.950 | 0.000 | Largest vocabulary variety, but many continuations become semantically disconnected or ungrammatical. |

These averages cover all 12 samples at each temperature. Distinct-token ratio is unique generated tokens divided by 50. Adjacent-repeat rate measures identical neighboring tokens; it does not capture longer repeated phrases, so the text samples are also necessary. For this corpus and training budget, **temperature 1.0** gives the most useful overall balance. Temperature 0.5 can be useful for more predictable local phrasing; 1.5 is useful for exploratory variety but reduces coherence.

## Limitations and conclusion

The experiment uses only the first 120,000 tokens, a small 20-token context, and three epochs. Shakespeare play text changes speakers and scenes abruptly. Lowercasing and removing line breaks also erase cues that would help identify speakers and verse structure. The chronological split creates vocabulary shift, and model parameter counts are not exactly equal. These factors limit absolute quality and prevent broad claims about which architecture is universally best.

Within this controlled run, **GRU provides the most effective autoregressive generation** because it achieves the strongest held-out probability estimates with competitive accuracy and produces readable short dialogue fragments at temperature 1.0. The LSTM is a close alternative when top-1 next-word accuracy is the priority.
