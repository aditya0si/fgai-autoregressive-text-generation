"""Comparable next-word RNN experiments on WikiText-2 raw."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as parquet
import torch
from torch import nn


DATA_URL = "https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1"
SPLIT_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{}-00000-of-00001.parquet"
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", re.UNICODE)
MODELS = {"Simple RNN": nn.RNN, "Bidirectional RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
SEEDS = ["the history of", "in the united states", "the first season of"]
TEMPERATURES = [0.5, 1.0, 1.5]


def load_split(path: Path, name: str) -> tuple[str, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(SPLIT_URL.format(name), path)
    rows = parquet.read_table(path, columns=["text"]).column("text").to_pylist()
    nonempty = [row.strip() for row in rows if row.strip()]
    body = [row for row in nonempty if not re.fullmatch(r"=+\s*.*?\s*=+", row)]
    text = " ".join(body).replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
    return text, {"rows": len(rows), "nonempty_rows": len(nonempty), "body_rows": len(body), "characters": len(text)}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().replace("\r\n", "\n"))


def windows(ids: list[int], context: int, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = range(0, len(ids) - context, stride)
    x = torch.tensor([ids[i : i + context] for i in starts], dtype=torch.long)
    y = torch.tensor([ids[i + context] for i in starts], dtype=torch.long)
    return x, y


class NextWordModel(nn.Module):
    def __init__(self, kind: str, vocab_size: int, embedding: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding)
        bidirectional = kind == "Bidirectional RNN"
        self.recurrent = MODELS[kind](embedding, hidden, batch_first=True, bidirectional=bidirectional)
        self.output = nn.Linear(hidden * (2 if bidirectional else 1), vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.recurrent(self.embedding(x))
        if self.recurrent.bidirectional:
            features = torch.cat((seq[:, -1, :self.recurrent.hidden_size], seq[:, 0, self.recurrent.hidden_size:]), dim=1)
        else:
            features = seq[:, -1]
        return self.output(features)


@torch.no_grad()
def evaluate(model, x, y, batch_size, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    for start in range(0, len(y), batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        logits = model(xb)
        total_loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
    loss = total_loss / len(y)
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": total_correct / len(y)}


def detokenize(words: list[str]) -> str:
    s = " ".join(words)
    return re.sub(r"\s+([,.;:!?])", r"\1", s)


@torch.no_grad()
def generate(model, seed, token_to_id, id_to_token, context, temperature, length, rng, device):
    model.eval()
    words = tokenize(seed)
    generated = []
    for _ in range(length):
        recent = words[-context:]
        ids = [token_to_id.get(w, token_to_id["<unk>"]) for w in recent]
        ids = [token_to_id["<pad>"]] * (context - len(ids)) + ids
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0]
        probabilities = torch.softmax(logits / temperature, dim=0).cpu().numpy()
        probabilities[token_to_id["<pad>"]] = 0
        probabilities[token_to_id["<unk>"]] = 0
        probabilities /= probabilities.sum()
        token = id_to_token[int(rng.choice(len(probabilities), p=probabilities))]
        generated.append(token)
        words.append(token)
    return detokenize(generated), generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-tokens", type=int, default=240000)
    parser.add_argument("--evaluation-tokens", type=int, default=60000)
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--generated-words", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(4403)
    np.random.seed(4403)
    random.seed(4403)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4403)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).parent
    results = root / "results"
    results.mkdir(exist_ok=True)

    raw_splits = {}
    source_stats = {}
    for name in ("train", "validation", "test"):
        raw_splits[name], source_stats[name] = load_split(root / "data" / f"wikitext2_{name}.parquet", name)
    limits = {"train": args.train_tokens, "validation": args.evaluation_tokens, "test": args.evaluation_tokens}
    splits = {name: tokenize(text)[:limits[name]] for name, text in raw_splits.items()}
    common = Counter(splits["train"]).most_common(args.vocab_size - 2)
    vocabulary = ["<pad>", "<unk>"] + [word for word, _ in common]
    token_to_id = {word: i for i, word in enumerate(vocabulary)}
    arrays = {name: windows([token_to_id.get(t, 1) for t in part], args.context, args.stride)
              for name, part in splits.items()}
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = (arrays[k] for k in ("train", "validation", "test"))
    metadata = {"source": DATA_URL, "source_split_stats": source_stats,
                "split_tokens_used": {k: len(v) for k, v in splits.items()}, "vocabulary_size": len(vocabulary),
                "unknown_rate": {k: sum(t not in token_to_id for t in v) / len(v) for k, v in splits.items()},
                "examples": {k: len(v[1]) for k, v in arrays.items()}, "device": str(device), "config": vars(args)}
    (results / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (results / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    all_results = {}
    samples = []
    for kind in MODELS:
        torch.manual_seed(4403)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(4403)
        model = NextWordModel(kind, len(vocabulary), args.embedding, args.hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        history = []
        best_state, best_epoch, best_validation_loss = None, None, float("inf")
        # Prime device kernels before timing without updating model weights.
        warm_x = x_train[:args.batch_size].to(device)
        warm_y = y_train[:args.batch_size].to(device)
        nn.functional.cross_entropy(model(warm_x), warm_y).backward()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(4403 + epoch))
            train_loss = 0.0
            for batch_ids in order.split(args.batch_size):
                xb, yb = x_train[batch_ids].to(device), y_train[batch_ids].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(batch_ids)
            val = evaluate(model, x_val, y_val, args.batch_size, device)
            record = {"epoch": epoch + 1, "train_loss": train_loss / len(y_train), "validation_loss": val["loss"],
                      "validation_accuracy": val["accuracy"]}
            history.append(record)
            if val["loss"] < best_validation_loss:
                best_validation_loss = val["loss"]
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            print(kind, record, flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start_time
        model.load_state_dict(best_state)
        test = evaluate(model, x_test, y_test, args.batch_size, device)
        all_results[kind] = {"parameters": sum(p.numel() for p in model.parameters()),
                             "selected_epoch": best_epoch,
                             "train_seconds": seconds, "validation": evaluate(model, x_val, y_val, args.batch_size, device),
                             "test": test, "history": history}
        torch.save({"model": model.state_dict(), "kind": kind, "config": vars(args)}, results / f"{kind.lower().replace(' ', '_')}.pt")
        for seed_idx, seed in enumerate(SEEDS):
            for temperature in TEMPERATURES:
                rng = np.random.default_rng(4403 + seed_idx)
                continuation, words = generate(model, seed, token_to_id, vocabulary, args.context,
                                               temperature, args.generated_words, rng, device)
                samples.append({"model": kind, "seed": seed, "temperature": temperature,
                                "continuation": continuation, "distinct_1": len(set(words)) / len(words),
                                "adjacent_repeat_rate": sum(a == b for a, b in zip(words, words[1:])) / (len(words)-1)})
        (results / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        (results / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind, record in all_results.items():
        h = record["history"]
        axes[0].plot([v["epoch"] for v in h], [v["validation_loss"] for v in h], marker="o", label=kind)
        axes[1].plot([v["epoch"] for v in h], [v["validation_accuracy"] for v in h], marker="o", label=kind)
    axes[0].set(title="Validation loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Top-1 accuracy")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "training_curves.png", dpi=180)
    sample_lines = ["# Generated continuations", "", "Each continuation adds 50 words to the same seed; sampling uses a fixed random seed per prompt.", ""]
    for sample in samples:
        sample_lines += [f"## {sample['model']} | temperature {sample['temperature']} | seed: {sample['seed']}",
                         "", sample["continuation"], ""]
    (results / "samples.md").write_text("\n".join(sample_lines), encoding="utf-8")
    print("Finished. Results saved to", results, flush=True)


if __name__ == "__main__":
    main()
