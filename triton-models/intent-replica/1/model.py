"""Triton python-backend intent classifier: utterance in, best corpus match out.

Reproduces the production pipeline shape:

    TEXT (BYTES) -> tokenize -> BERT encode -> mean pool -> L2 normalise
                 -> cosine vs a server-side corpus -> argmax
                 -> MATCH / INTENT / SCORE

The corpus is model state, not request state. It is loaded once in
initialize() and never crosses the wire, so a request carries one short
string and the response carries the matched string back.

Two parameters exist so the serving strategy can be swept without editing
this file:

    COSINE_DEVICE  "cpu" reproduces production (numpy dot product on the
                   host). "cuda" moves the same matmul onto the GPU, which
                   is the A/B that shows whether the retrieval stage is the
                   real bottleneck.
    TORCH_THREADS  pins torch's CPU thread pool. With numpy cosine on the
                   host, instances contend for the same cores, and the
                   default (all cores) makes that contention invisible in
                   per-instance timings.

The corpus vectors are already unit-length (see tools/build_corpus.py), so
cosine here is a bare dot product with no normalisation on the hot path.
"""

import json
import os
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        params = self._params(args)

        self.cosine_device = params.get("COSINE_DEVICE", "cpu")
        # Off by default: looping per request is what the production python
        # wrapper does, so the default reproduces it. Turning this on measures
        # what the already-happening dynamic batching is worth if it is
        # actually exploited.
        self.batch_across_requests = params.get("BATCH_ACROSS_REQUESTS", "false") == "true"
        threads = int(params.get("TORCH_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to("cuda").eval()

        # Corpus load. Each Triton python-backend instance is its own
        # process, so this memory is multiplied by instance_group.count.
        t0 = time.time()
        self.corpus = np.load(os.path.join(model_dir, "corpus.npy"))
        with open(os.path.join(model_dir, "corpus.txt"), encoding="utf-8") as fh:
            self.corpus_texts = [line.rstrip("\n") for line in fh]
        with open(os.path.join(model_dir, "corpus_labels.txt"), encoding="utf-8") as fh:
            self.corpus_labels = [line.rstrip("\n") for line in fh]

        if not (len(self.corpus) == len(self.corpus_texts) == len(self.corpus_labels)):
            raise ValueError(
                f"corpus row misalignment: {len(self.corpus)} vectors vs "
                f"{len(self.corpus_texts)} texts vs {len(self.corpus_labels)} labels"
            )

        if self.cosine_device == "cuda":
            self.corpus_gpu = torch.from_numpy(self.corpus).to("cuda")
        else:
            self.corpus_gpu = None

        pb_utils.Logger.log_info(
            f"intent-replica ready: {len(self.corpus)} rows, "
            f"cosine={self.cosine_device}, torch_threads={threads or 'default'}, "
            f"batch_across_requests={self.batch_across_requests}, "
            f"load={time.time() - t0:.2f}s"
        )

    @staticmethod
    def _params(args):
        """Decode the string parameters from config.pbtxt.

        The python backend hands over model_config as a JSON string, and the
        values inside `parameters` show up either as {"string_value": ...}
        dicts or as bare strings depending on how the config was parsed.
        Values may also keep their JSON quotes, which would silently select
        the wrong branch, so strip them unconditionally.
        """
        config = args.get("model_config", {}) or {}
        if isinstance(config, str):
            config = json.loads(config)
        out = {}
        for key, value in (config.get("parameters", {}) or {}).items():
            if isinstance(value, dict):
                value = value.get("string_value", "")
            out[key] = str(value).strip().strip('"')
        return out

    @torch.no_grad()
    def _encode(self, texts):
        """Mean-pool over the attention mask, then L2-normalise.

        Must match tools/build_corpus.py exactly, or query and corpus
        vectors live in different spaces and every score is wrong.
        """
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to("cuda")
        hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # Mask before summing: pad positions have non-zero hidden states, so
        # summing raw hidden and dividing by the mask length would let padding
        # leak in. That error scales with batch padding, so it differs between
        # a 1-row request and a 256-row build batch.
        pooled = (hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()

    def _retrieve(self, queries):
        """Return (index, score) for each query vector."""
        if self.corpus_gpu is not None:
            q = torch.from_numpy(queries).to("cuda")
            sims = self.corpus_gpu @ q.T
            scores, idx = sims.max(dim=0)
            return idx.cpu().numpy(), scores.cpu().numpy()
        # Production path: numpy on the host. This reads the whole corpus per
        # request, so it is memory-bandwidth bound rather than FLOP bound.
        sims = self.corpus @ queries.T
        idx = sims.argmax(axis=0)
        return idx, sims[idx, np.arange(len(queries))]

    def execute(self, requests):
        if self.batch_across_requests:
            return self._execute_batched(requests)
        responses = []
        for request in requests:
            texts, shape = self._decode(request)
            responses.append(self._respond(texts, shape))
        return responses

    @staticmethod
    def _decode(request):
        in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
        raw = in_tensor.as_numpy()
        texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t)
                 for t in raw.flatten()]
        return texts, raw.shape

    def _respond(self, texts, shape):
        queries = self._encode(texts)
        idx, scores = self._retrieve(queries)
        return self._make_response(idx, scores, shape)

    def _execute_batched(self, requests):
        """Encode and retrieve once for the whole request group.

        Dynamic batching already groups requests before execute() is called,
        but looping over them one at a time throws that work away: each row
        pays its own encoder call and its own full corpus read. Concatenating
        turns N corpus reads into one and N small GPU launches into one.
        """
        decoded = [self._decode(r) for r in requests]
        texts = [t for chunk, _ in decoded for t in chunk]
        queries = self._encode(texts)
        idx, scores = self._retrieve(queries)

        responses = []
        offset = 0
        for _, shape in decoded:
            n = int(np.prod(shape)) if shape else 1
            responses.append(self._make_response(
                idx[offset:offset + n], scores[offset:offset + n], shape))
            offset += n
        return responses

    def _make_response(self, idx, scores, shape):
        matches = np.array([self.corpus_texts[i] for i in idx], dtype=object)
        intents = np.array([self.corpus_labels[i] for i in idx], dtype=object)
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("MATCH", matches.reshape(shape)),
            pb_utils.Tensor("INTENT", intents.reshape(shape)),
            pb_utils.Tensor("SCORE", scores.astype(np.float32).reshape(shape)),
        ])

    def finalize(self):
        pass
