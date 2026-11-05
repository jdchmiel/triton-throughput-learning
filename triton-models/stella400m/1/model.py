"""Triton python-backend embedding model: text in, unit-length 1024-d vector out.

Pipeline (NovaSearch/stella_en_400M_v5, the 400M Stella encoder):

    TEXT (BYTES) -> prepend instruct prompt -> tokenize -> NewModel encode
                 -> masked mean pool -> 2_Dense_1024 projection
                 -> L2 normalise -> EMBEDDING [1024] FP32

The instruct prompt is applied server-side (PROMPT_MODE=always by default) so
callers send bare text. Stella is asymmetric: the prompt belongs on queries
and must not be used when indexing the corpus. Either set PROMPT_MODE=never on
the indexing model, or send the optional USE_PROMPT input per request.

Two repairs run after from_pretrained(). Both are transformers-5.x shims for
this 2024-era remote code, and both fail silently if removed -- the model
still loads, still runs, and still returns unit-length vectors that are just
nearly the same vector for every input.

  1. Non-persistent buffers. Transformers 5 builds the model on the meta
     device and materialises it with to_empty(), which does not re-run the
     __init__ lines that fill buffers registered with persistent=False. Three
     matter here: NewEmbeddings.position_ids (an arange) and the rope
     inv_freq/cos_cached/sin_cached. position_ids came back as uninitialised
     memory, so rope_cos[position_ids] gathered out of bounds and killed the
     process with a CUDA device-side assert. The caches came back as memory
     that happened to be in range, which is worse: no crash, just degenerate
     attention. Every pairwise cosine landed in 0.999x and the ranking was
     inverted -- unrelated text scored highest.

  2. get_extended_attention_mask was removed from PreTrainedModel in
     transformers 5, but modeling.py still calls self.get_extended_attention_mask.
     It is monkeypatched back onto the instance. modeling.py passes device=
     but not dtype=, so the shim has to infer the dtype itself.
"""

import json
import os
import types

import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


def _repair_nonpersistent_buffers(model):
    """Re-fill the buffers that to_empty() left as uninitialised memory.

    Returns the worst absolute error between the rebuilt rope cache and the
    analytic NTK value. That check exists because a silent regression here is
    invisible in the output shape and only shows up as garbage similarity.
    """
    emb = model.embeddings
    max_pos = int(model.config.max_position_embeddings)
    emb.register_buffer(
        "position_ids",
        torch.arange(max_pos, device=emb.word_embeddings.weight.device),
        persistent=False,
    )

    rope = emb.rotary_emb
    # max_seq_len_cached is a plain python int, so it survives to_empty() and
    # tells us the length __init__ built the cache for (8192 * 2.0 = 16384).
    seq_len = int(rope.max_seq_len_cached)
    dtype = next(model.parameters()).dtype
    rope._set_cos_sin_cache(seq_len=seq_len, device=emb.position_ids.device, dtype=dtype)

    # Analytic reference, mirroring NTKScalingRotaryEmbedding._set_cos_sin_cache
    # for the seq_len > max_position_embeddings branch.
    err = float("inf")
    try:
        dim = int(rope.dim)
        if seq_len > int(rope.max_position_embeddings) and rope.mixed_b is None:
            base = rope.base * rope.scaling_factor
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            inv_freq = inv_freq / rope.scaling_factor ** (2 / dim)
            t = torch.arange(seq_len, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            ref = torch.cat((freqs, freqs), dim=-1).cos()
            err = float((rope.cos_cached.cpu().float() - ref).abs().max())
    except Exception as exc:  # never let a verification helper break the load
        pb_utils.Logger.log_warn(f"stella400m: rope verification skipped: {exc}")
    return err


def _restore_extended_attention_mask(model):
    """Put back the PreTrainedModel helper that transformers 5 removed."""

    def _get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        # modeling.py calls this with device= only, so dtype has to come from
        # the parameters or torch.finfo(None) raises TypeError.
        dtype = dtype or next(self.parameters()).dtype
        return (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min

    model.get_extended_attention_mask = types.MethodType(_get_extended_attention_mask, model)


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        params = self._params(args)

        self.prompt_mode = params.get("PROMPT_MODE", "always")
        self.max_length = int(params.get("MAX_LENGTH", "512"))
        dtype_name = params.get("DTYPE", "float16")
        self.dtype = getattr(torch, dtype_name)
        threads = int(params.get("TORCH_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)

        self.prompt = params.get(
            "QUERY_PROMPT",
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query.\nQuery: ",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        load_kwargs = dict(
            trust_remote_code=True,
            # config.json ships both of these as true. Both require xformers,
            # which the miniLM env does not have, and modeling.py hard-asserts
            # rather than falling back.
            use_memory_efficient_attention=False,
            unpad_inputs=False,
            torch_dtype=self.dtype,
        )
        attn_impl = params.get("ATTN_IMPLEMENTATION", "")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModel.from_pretrained(model_dir, **load_kwargs)
        _restore_extended_attention_mask(self.model)
        self.model = self.model.to("cuda").eval()
        rope_err = _repair_nonpersistent_buffers(self.model)

        # 2_Dense_1024 is Stella's 1024-d output head. Weights are stored under
        # linear.*, so the prefix has to go before load_state_dict. The head is
        # a separate module, so it needs its own move to the GPU.
        # 1_Pooling/config.json asks for mean tokens only, which is what
        # _encode does.
        head_dir = os.path.join(model_dir, params.get("POOLING_HEAD", "2_Dense_1024"))
        head_sd = {
            k[len("linear."):]: v
            for k, v in load_file(os.path.join(head_dir, "model.safetensors")).items()
        }
        self.head = torch.nn.Linear(head_sd["weight"].shape[1], head_sd["weight"].shape[0])
        self.head.load_state_dict(head_sd)
        self.head = self.head.to("cuda").to(self.dtype).eval()

        self.out_dim = int(self.head.out_features)
        self._warmup()

        pb_utils.Logger.log_info(
            f"stella400m ready: dim={self.out_dim}, dtype={dtype_name}, "
            f"attn={attn_impl or 'default'}, prompt_mode={self.prompt_mode}, "
            f"max_length={self.max_length}, rope_cache_err={rope_err:.2e}"
        )
        # Tolerance is dtype-scaled: the cache is built in the model dtype, so
        # fp16 quantisation alone moves cos by ~1e-3. Anything above that means
        # the cache really is wrong, which shows up as degenerate similarity.
        tol = 1e-2 if self.dtype == torch.float16 else 1e-4
        if rope_err > tol:
            pb_utils.Logger.log_warn(
                f"stella400m: rope cache does not match the analytic value "
                f"(max_err={rope_err:.4f} > {tol:g}); embeddings will be degenerate"
            )

    def _warmup(self):
        """Prime cuBLAS and the CUDA caching allocator before the first request."""
        for n in (1, 8):
            self._encode(["warmup"] * n)

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
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        ).to("cuda")
        hidden = self.model(**enc)[0]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # Pad positions carry non-zero hidden states, so pool against the mask
        # rather than the padded length or batch padding leaks into the vector.
        pooled = (hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        projected = self.head(pooled.to(self.dtype)).float()
        return F.normalize(projected, p=2, dim=1)

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            raw = in_tensor.as_numpy()
            texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t)
                     for t in raw.flatten()]

            use_prompt = self.prompt_mode == "always"
            opt = pb_utils.get_input_tensor_by_name(request, "USE_PROMPT")
            if opt is not None:
                flags = opt.as_numpy().flatten().tolist()
                if flags:
                    use_prompt = bool(int(flags[0]))
            if use_prompt:
                texts = [self.prompt + t for t in texts]

            vectors = self._encode(texts).cpu().numpy().astype(np.float32)
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("EMBEDDING", vectors.reshape(len(texts), self.out_dim))
            ]))
        return responses

    def finalize(self):
        pass
