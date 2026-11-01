#!/usr/bin/env python3
"""Build the offline intent corpus for the intent-replica Triton model.

Encodes every utterance from CLINC150 + banking77 with bert-base-uncased,
mean-pools over the attention mask, L2-normalises, and writes three files
that the Triton python backend loads once at initialize():

    corpus.npy        float32 [N, 768], already unit-length
    corpus.txt        one raw utterance per line, row-aligned with corpus.npy
    corpus_labels.txt one intent label per line, row-aligned

Because the vectors are unit-length, cosine similarity is a plain dot
product, so the serving side needs only corpus @ q.

Run on the GPU host, not on the client:
    /AI/triton/my_env/bin/python build_corpus.py

This is offline tooling. Nothing here runs inside a request.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

# Public intent corpora. Both are plain (utterance, intent) pairs, which is
# the shape of the production corpus. Note the label column is named
# "intent" in CLINC150 and "label" in banking77, so it is detected per file.
SOURCES = {
    "banking77": [
        "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/{split}-00000-of-00001.parquet",
        ["train", "test"],
    ],
    "clinc_oos": [
        "https://huggingface.co/api/datasets/clinc/clinc_oos/parquet/imbalanced/{split}/0.parquet",
        ["train", "validation", "test"],
    ],
    # CLINC "plus" is a superset of "imbalanced"; only its extra rows survive
    # the dedupe below, but pulling it costs one request and adds ~4.6k rows.
    "clinc_oos_plus": [
        "https://huggingface.co/api/datasets/clinc/clinc_oos/parquet/plus/{split}/0.parquet",
        ["train"],
    ],
}


def fetch(url: str, dest: Path) -> Path:
    """Download a parquet file if it is not already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import urllib.request

    print(f"  fetching {dest.name}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    return dest


def label_names(table, column: str) -> list[str] | None:
    """Extract class names that HF datasets embeds in the parquet schema.

    Both datasets store the label column as ClassLabel, whose names ride in
    the schema metadata. Reading them beats hardcoding a list: a wrong
    index->name mapping would produce plausible-looking but false intents.
    """
    meta = table.schema.metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return None
    try:
        info = json.loads(raw)
        names = info["info"]["features"][column]["names"]
    except (KeyError, TypeError, ValueError):
        return None
    return names if isinstance(names, list) else None


def label_column(table, tag: str) -> str:
    """Return the intent column, which differs between the two datasets."""
    for candidate in ("intent", "label"):
        if candidate in table.schema.names:
            return candidate
    raise ValueError(f"{tag}: no intent/label column in {table.schema.names}")


def load_rows(work: Path) -> list[tuple[str, str]]:
    """Return (utterance, intent) pairs from every configured split."""
    rows: list[tuple[str, str]] = []

    for dataset, (template, splits) in SOURCES.items():
        for split in splits:
            tag = f"{dataset}_{split}"
            path = fetch(template.format(split=split), work / f"{tag}.parquet")
            table = pq.read_table(path)
            column = label_column(table, tag)
            texts = table["text"].to_pylist()
            labels = table[column].to_pylist()
            names = label_names(table, column)
            print(f"  {tag}: {len(texts)} rows, "
                  f"{len(names) if names else 'no'} intent names")
            for text, label in zip(texts, labels):
                name = names[label] if names and label < len(names) else f"{tag}_{label}"
                rows.append((text.strip(), name))

    return rows


def dedupe(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Normalise whitespace, then drop exact-duplicate utterances.

    Whitespace is collapsed because corpus.txt is line-oriented: a single
    utterance containing an embedded newline would occupy two lines and
    shift every later row out of alignment with its vector and label. A few
    rows in banking77 do contain newlines.

    Duplicates are dropped because they make the argmax ambiguous, so two
    identical rows can return either label.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for text, label in rows:
        text = " ".join(text.split())
        label = " ".join(str(label).split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append((text, label))
    return out


def load_encoder(model_dir: str, device: str):
    """Load tokenizer + model once; encode() and the drift check share them."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def encode_batch(tokenizer, model, texts: list[str], batch_size: int, device: str,
                 verbose: bool = True) -> np.ndarray:
    """Mean-pool bert-base-uncased over the attention mask, then L2-normalise.

    Pooling and normalisation match the serving-side model.py exactly, so a
    corpus vector and a query vector live in the same space.
    """
    out = np.empty((len(texts), 768), dtype=np.float32)
    start = time.time()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # Mask before summing: pad positions have non-zero hidden states, so
        # summing raw hidden and dividing by the mask length would let padding
        # leak in. That error scales with batch padding, so it differs between
        # a 1-row request and a 256-row build batch. Must match model.py.
        pooled = (hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        vec = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        out[i : i + len(chunk)] = vec
        done = i + len(chunk)
        if verbose and (done % 5000 < batch_size or done == len(texts)):
            rate = done / max(time.time() - start, 1e-9)
            print(f"  encoded {done}/{len(texts)}  ({rate:.0f} utt/s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/corpus", help="cache dir for downloaded parquet")
    ap.add_argument("--model", default="google-bert/bert-base-uncased",
                    help="HF model id or local snapshot dir")
    ap.add_argument("--hf-home", default="/tmp/hfcache",
                    help="HF cache to use, so the snapshot is not re-downloaded")
    ap.add_argument("--out", default=None, help="output dir (default: ../1 next to this script)")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only the first N utterances (for corpus-size sweeps)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Must be set before transformers touches the network.
    os.environ.setdefault("HF_HOME", args.hf_home)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "1"
    out.mkdir(parents=True, exist_ok=True)

    print("loading datasets")
    rows = dedupe(load_rows(work))
    if args.limit:
        rows = rows[: args.limit]
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    print(f"{len(rows)} unique utterances, "
          f"{len(set(labels))} distinct intents")

    print(f"encoding on {args.device}")
    tokenizer, model = load_encoder(args.model, args.device)
    vectors = encode_batch(tokenizer, model, texts, args.batch_size, args.device)

    # Sanity: unit length means the serving side can skip normalising the
    # corpus, so cosine there is a bare dot product.
    norms = np.linalg.norm(vectors, axis=1)
    print(f"norms min={norms.min():.6f} max={norms.max():.6f}")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise SystemExit("corpus is not unit-length; serving side would need to renormalise")

    # Retrieval self-check: each row's best match must be itself at
    # similarity ~1.0. A wrong argmax or a misaligned corpus would still
    # produce plausible strings, so assert it here rather than discovering it
    # later through suspicious-looking latency numbers. Chunked because the
    # full N x N similarity matrix would be ~5 GB at 37k rows.
    exact_hits = 0
    runner_ups = np.empty(len(texts), dtype=np.float32)
    for i in range(0, len(texts), 2048):
        block = vectors[i : i + 2048]
        sims = block @ vectors.T
        rows = np.arange(i, i + len(block))
        exact_hits += int((sims.argmax(axis=1) == rows).sum())
        sims[rows - i, rows] = -1.0
        runner_ups[rows] = sims.max(axis=1)
    exact = exact_hits / len(texts) * 100
    print(f"exact self-match {exact:.2f}% (should be 100%)")
    if exact < 99.0:
        raise SystemExit("self-match too low: corpus rows are not distinguishable")
    # The runner-up score shows how close the nearest *distinct* utterance
    # sits. Low margins mean the argmax is fragile, not broken.
    print(f"median runner-up similarity {np.median(runner_ups):.4f}, "
          f"max {runner_ups.max():.4f}")

    # Single-row re-encode check. The batched encode above pads every chunk
    # to its longest row, so if pooling ever lets padding leak in, the error
    # scales with batch composition and is invisible to a self-check that
    # only compares batched vectors against each other. Re-encoding a sample
    # one row at a time reproduces what a 1-row Triton request does, so any
    # drift here shows up as a score below 1.0 before the corpus ships.
    sample = np.linspace(0, len(texts) - 1, min(64, len(texts)), dtype=int)
    drift = []
    for i in sample:
        one = encode_batch(tokenizer, model, [texts[i]], 1, args.device, verbose=False)
        drift.append(float(one[0] @ vectors[i]))
    drift_arr = np.array(drift, dtype=np.float32)
    print(f"single-row vs batched agreement min={drift_arr.min():.6f} "
          f"median={float(np.median(drift_arr)):.6f}")
    if drift_arr.min() < 0.999:
        raise SystemExit(
            "encoding depends on batch padding: query and corpus vectors "
            "would live in different spaces"
        )

    np.save(out / "corpus.npy", vectors)
    (out / "corpus.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    (out / "corpus_labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    # Re-read from disk and confirm one line per vector. The serving side
    # raises on misalignment, but failing here points at the real cause.
    for name in ("corpus.txt", "corpus_labels.txt"):
        got = len((out / name).read_text(encoding="utf-8").splitlines())
        if got != len(texts):
            raise SystemExit(f"{name}: {got} lines for {len(texts)} vectors")

    print(f"wrote {out}/corpus.npy {vectors.shape} {vectors.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
