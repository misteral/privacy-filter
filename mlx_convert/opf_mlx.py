"""MLX inference backend for OpenAI Privacy Filter.

This module intentionally mirrors the small PyTorch runtime in ``opf._model`` but
keeps the integration separate: it loads the original HuggingFace checkpoint
(``original/config.json`` + ``original/model.safetensors`` after OPF's downloader
promotes the subtree) directly with MLX and reuses the existing tokenizer,
Viterbi decoder, and span-rendering helpers.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import tiktoken
import torch

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on non-Apple CI
    raise ModuleNotFoundError(
        "The MLX backend requires `mlx`. Install with `uv pip install -e '.[mlx]'` "
        "on Apple Silicon."
    ) from exc

from opf._api import RedactionResult, _redact_text, _warning_for_prediction
from opf._common.checkpoint_download import ensure_default_checkpoint
from opf._common.constants import OUTPUT_MODES, SCHEMA_VERSION
from opf._common.label_space import resolve_label_space_from_config
from opf._core.decoding import build_sequence_decoder
from opf._core.runtime import (
    DetectedSpan,
    PredictionResult,
    build_detection_summary,
)
from opf._core.sequence_labeling import TokenizedExample, build_label_info, example_to_windows
from opf._core.spans import (
    decode_text_with_offsets,
    discard_overlapping_spans_by_label,
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration fields used by the OPF transformer checkpoint."""

    model_type: str = "privacy_filter"
    encoding: str = "o200k_base"
    num_hidden_layers: int = 8
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 200064
    num_labels: int = 33
    hidden_size: int = 640
    intermediate_size: int = 640
    head_dim: int = 64
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    sliding_window: int = 257
    bidirectional_context: bool = True
    bidirectional_left_context: int = 128
    bidirectional_right_context: int = 128
    initial_context_length: int = 4096
    max_position_embeddings: int = 131072
    default_n_ctx: int = 128000
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    param_dtype: str = "bfloat16"
    swiglu_limit: float = 7.0
    packed_geglu: bool = False
    moe_chunk_size: int = 8

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        fields = cls.__dataclass_fields__
        values = {k: v for k, v in payload.items() if k in fields}
        return cls(**values)


_NEG_INF = -1.0e9


def _linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    """Apply a PyTorch-layout linear weight: ``[out, in]``."""
    y = mx.matmul(x.astype(weight.dtype), weight.T)
    if bias is not None:
        y = y + bias
    return y


def _rms_norm(x: mx.array, scale: mx.array, eps: float = 1e-5) -> mx.array:
    dtype = x.dtype
    t = x.astype(mx.float32)
    t = t * mx.rsqrt(mx.mean(t * t, axis=-1, keepdims=True) + eps)
    return (t * scale.astype(mx.float32)).astype(dtype)


def _swiglu(x: mx.array, *, limit: float = 7.0, packed: bool = False) -> mx.array:
    if packed:
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
    else:
        half = x.shape[-1] // 2
        x_glu, x_linear = x[..., :half], x[..., half:]
    x_glu = mx.minimum(x_glu, limit)
    x_linear = mx.clip(x_linear, -limit, limit)
    return (x_glu * mx.sigmoid(1.702 * x_glu)) * (x_linear + 1.0)


def _batched_linear_expert(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """Apply expert weights stored as ``[batch, in, out]`` to ``[batch, in]``."""
    y = mx.matmul(x.astype(weight.dtype)[:, None, :], weight)[:, 0, :]
    return y + bias


class RotaryEmbedding:
    """YaRN RoPE cache compatible with the PyTorch OPF implementation."""

    def __init__(self, cfg: ModelConfig):
        self.head_dim = int(cfg.head_dim)
        self.base = float(cfg.rope_theta)
        self.initial_context_length = int(cfg.initial_context_length)
        self.scaling_factor = float(cfg.rope_scaling_factor)
        self.ntk_alpha = float(cfg.rope_ntk_alpha)
        self.ntk_beta = float(cfg.rope_ntk_beta)
        self.max_position_embeddings = max(
            int(self.initial_context_length * self.scaling_factor),
            self.initial_context_length,
        )
        self._cos_cache, self._sin_cache = self._compute_cos_sin(
            self.max_position_embeddings
        )

    def _concentration_and_inv_freq(self) -> tuple[float, mx.array]:
        d_half = self.head_dim / 2
        freq = self.base ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim)
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            low = d_half * math.log(
                self.initial_context_length / (self.ntk_beta * 2 * math.pi)
            ) / math.log(self.base)
            high = d_half * math.log(
                self.initial_context_length / (self.ntk_alpha * 2 * math.pi)
            ) / math.log(self.base)
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (np.arange(d_half, dtype=np.float32) - low) / (high - low)
            mask = 1.0 - np.clip(ramp, 0.0, 1.0)
            inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        return concentration, mx.array(inv_freq, dtype=mx.float32)

    def _compute_cos_sin(self, num_tokens: int) -> tuple[mx.array, mx.array]:
        concentration, inv_freq = self._concentration_and_inv_freq()
        t = mx.arange(num_tokens, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        return mx.cos(freqs) * concentration, mx.sin(freqs) * concentration

    def _ensure_len(self, num_tokens: int) -> None:
        if num_tokens <= self._cos_cache.shape[0]:
            return
        self._cos_cache, self._sin_cache = self._compute_cos_sin(num_tokens)

    @staticmethod
    def _apply(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: [B, T, H, D], cos/sin: [T, D/2]
        cos = cos[None, :, None, :].astype(x.dtype)
        sin = sin[None, :, None, :].astype(x.dtype)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return mx.stack([y1, y2], axis=-1).reshape(x.shape)

    def __call__(self, query: mx.array, key: mx.array) -> tuple[mx.array, mx.array]:
        batch, tokens, _ = query.shape
        self._ensure_len(tokens)
        cos = self._cos_cache[:tokens]
        sin = self._sin_cache[:tokens]
        q_shape = query.shape
        k_shape = key.shape
        query = query.reshape(batch, tokens, -1, self.head_dim)
        key = key.reshape(batch, tokens, -1, self.head_dim)
        return self._apply(query, cos, sin).reshape(q_shape), self._apply(key, cos, sin).reshape(k_shape)


class AttentionBlock:
    def __init__(self, cfg: ModelConfig, weights: dict[str, mx.array], prefix: str):
        self.cfg = cfg
        self.sinks = weights[f"{prefix}.sinks"]
        self.norm_scale = weights[f"{prefix}.norm.scale"]
        self.qkv_weight = weights[f"{prefix}.qkv.weight"]
        self.qkv_bias = weights[f"{prefix}.qkv.bias"]
        self.out_weight = weights[f"{prefix}.out.weight"]
        self.out_bias = weights[f"{prefix}.out.bias"]
        self.rope = RotaryEmbedding(cfg)
        self.qk_scale = 1.0 / math.sqrt(math.sqrt(cfg.head_dim))

    def _local_sdpa(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        # q: [B, T, Hkv, Qmult, D], k/v: [B, T, Hkv, D]
        bsz, n_tokens, n_kv_heads, q_mult, d_head = q.shape
        left = int(self.cfg.bidirectional_left_context)
        right = int(self.cfg.bidirectional_right_context)
        window = left + right + 1

        k_padded = mx.pad(k, ((0, 0), (left, right), (0, 0), (0, 0)))
        v_padded = mx.pad(v, ((0, 0), (left, right), (0, 0), (0, 0)))
        offsets = mx.arange(window) - left
        positions = mx.arange(n_tokens)[:, None] + offsets[None, :]
        valid = (positions >= 0) & (positions < n_tokens)
        gather = mx.clip(positions + left, 0, n_tokens + left + right - 1).astype(mx.int32)

        k_win = k_padded[:, gather, :, :]
        v_win = v_padded[:, gather, :, :]
        scores = mx.einsum("bthqd,btwhd->bthqw", q, k_win).astype(mx.float32)
        scores = mx.where(valid[None, :, None, None, :], scores, _NEG_INF)

        sink_scores = (self.sinks * math.log(2.0)).reshape(n_kv_heads, q_mult)
        sink_scores = mx.broadcast_to(
            sink_scores[None, None, :, :, None],
            (bsz, n_tokens, n_kv_heads, q_mult, 1),
        ).astype(mx.float32)
        scores = mx.concatenate([scores, sink_scores], axis=-1)
        probs = mx.softmax(scores, axis=-1)[..., :-1].astype(v.dtype)
        attn = mx.einsum("bthqw,btwhd->bthqd", probs, v_win)
        return attn.reshape(bsz, n_tokens, n_kv_heads * q_mult * d_head)

    def __call__(self, x: mx.array) -> mx.array:
        cfg = self.cfg
        t = _rms_norm(x, self.norm_scale)
        qkv = _linear(t, self.qkv_weight, self.qkv_bias)
        q_end = cfg.num_attention_heads * cfg.head_dim
        k_end = q_end + cfg.num_key_value_heads * cfg.head_dim
        v_end = k_end + cfg.num_key_value_heads * cfg.head_dim
        q = qkv[:, :, :q_end]
        k = qkv[:, :, q_end:k_end]
        v = qkv[:, :, k_end:v_end]
        q, k = self.rope(q, k)
        q = q * self.qk_scale
        k = k * self.qk_scale
        bsz, n_tokens, _ = q.shape
        q_mult = cfg.num_attention_heads // cfg.num_key_value_heads
        q = q.reshape(bsz, n_tokens, cfg.num_key_value_heads, q_mult, cfg.head_dim)
        k = k.reshape(bsz, n_tokens, cfg.num_key_value_heads, cfg.head_dim)
        v = v.reshape(bsz, n_tokens, cfg.num_key_value_heads, cfg.head_dim)
        attn_out = self._local_sdpa(q, k, v)
        proj = _linear(attn_out, self.out_weight, self.out_bias).astype(x.dtype)
        return x + proj


class MLPBlock:
    def __init__(self, cfg: ModelConfig, weights: dict[str, mx.array], prefix: str):
        self.cfg = cfg
        self.norm_scale = weights[f"{prefix}.norm.scale"]
        self.gate_weight = weights[f"{prefix}.gate.weight"]
        self.gate_bias = weights[f"{prefix}.gate.bias"]
        self.w1 = weights[f"{prefix}.swiglu.weight"]
        self.b1 = weights[f"{prefix}.swiglu.bias"]
        self.w2 = weights[f"{prefix}.out.weight"]
        self.b2 = weights[f"{prefix}.out.bias"]

    def __call__(self, x: mx.array) -> mx.array:
        cfg = self.cfg
        batch_shape = x.shape[:-1]
        t = _rms_norm(x, self.norm_scale).reshape(-1, x.shape[-1])
        gate = _linear(t.astype(mx.float32), self.gate_weight.astype(mx.float32), self.gate_bias.astype(mx.float32))
        # MLX topk APIs have changed over time; argsort is stable and fine for 128 experts.
        expert_indices = mx.argsort(-gate, axis=-1)[:, : cfg.experts_per_token]
        expert_values = mx.take_along_axis(gate, expert_indices, axis=-1)
        expert_weights = mx.softmax(expert_values, axis=-1)

        outputs: list[mx.array] = []
        chunk_size = max(1, int(cfg.moe_chunk_size))
        n_tokens = int(t.shape[0])
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            t_chunk = t[start:end]
            idx_chunk = expert_indices[start:end]
            weight_chunk = expert_weights[start:end]
            accum = mx.zeros((end - start, cfg.hidden_size), dtype=mx.float32)
            for slot in range(cfg.experts_per_token):
                idx = idx_chunk[:, slot]
                h = _batched_linear_expert(t_chunk, self.w1[idx], self.b1[idx])
                h = _swiglu(h, limit=cfg.swiglu_limit, packed=cfg.packed_geglu)
                o = _batched_linear_expert(h, self.w2[idx], self.b2[idx]).astype(mx.float32)
                accum = accum + o * weight_chunk[:, slot : slot + 1].astype(mx.float32)
            outputs.append(accum.astype(x.dtype))
        y = mx.concatenate(outputs, axis=0).reshape(*batch_shape, -1)
        return x + y


class TransformerBlock:
    def __init__(self, cfg: ModelConfig, weights: dict[str, mx.array], layer_idx: int):
        self.attn = AttentionBlock(cfg, weights, f"block.{layer_idx}.attn")
        self.mlp = MLPBlock(cfg, weights, f"block.{layer_idx}.mlp")

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(self.attn(x))


class Transformer:
    def __init__(self, cfg: ModelConfig, weights: dict[str, mx.array]):
        self.cfg = cfg
        self.embedding = weights["embedding.weight"]
        self.blocks = [TransformerBlock(cfg, weights, i) for i in range(cfg.num_hidden_layers)]
        self.norm_scale = weights["norm.scale"]
        self.unembedding = weights["unembedding.weight"]

    @classmethod
    def from_checkpoint(cls, checkpoint: str | os.PathLike[str], *, moe_chunk_size: int = 8) -> "Transformer":
        checkpoint_path = Path(checkpoint).expanduser()
        cfg = ModelConfig.from_file(checkpoint_path / "config.json")
        cfg = ModelConfig(**{**cfg.__dict__, "moe_chunk_size": int(moe_chunk_size)})
        weights: dict[str, mx.array] = {}
        files = sorted(checkpoint_path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {checkpoint_path}")
        for file in files:
            weights.update(mx.load(str(file)))
        return cls(cfg, weights)

    def __call__(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError("Transformer expects [batch, tokens] int token ids")
        x = self.embedding[token_ids]
        for block in self.blocks:
            x = block(x)
            mx.eval(x)
        x = _rms_norm(x, self.norm_scale)
        return _linear(x, self.unembedding, None)


def resolve_checkpoint_path(model: str | os.PathLike[str] | None) -> str:
    if model is not None:
        return str(Path(model).expanduser())
    env_value = os.environ.get("OPF_CHECKPOINT")
    if env_value:
        return str(Path(env_value).expanduser())
    return ensure_default_checkpoint()


class OPFMLX:
    """Reusable MLX redactor for local Apple Silicon inference."""

    def __init__(
        self,
        *,
        model: str | os.PathLike[str] | None = None,
        context_window_length: int = 4096,
        trim_whitespace: bool = True,
        output_mode: Literal["typed", "redacted"] = "typed",
        decode_mode: Literal["viterbi", "argmax"] = "viterbi",
        discard_overlapping_predicted_spans: bool = False,
        output_text_only: bool = False,
        moe_chunk_size: int = 8,
    ) -> None:
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode!r}")
        self.checkpoint = resolve_checkpoint_path(model)
        with (Path(self.checkpoint) / "config.json").open("r", encoding="utf-8") as handle:
            self.config_json = json.load(handle)
        self.config = ModelConfig.from_file(Path(self.checkpoint) / "config.json")
        self.n_ctx = int(context_window_length)
        if self.n_ctx <= 0:
            raise ValueError("context_window_length must be positive")
        self.trim_whitespace = bool(trim_whitespace)
        self.discard_overlapping_predicted_spans = bool(discard_overlapping_predicted_spans)
        self.output_mode = str(output_mode)
        self.output_text_only = bool(output_text_only)
        self.encoding = tiktoken.get_encoding(str(self.config_json.get("encoding", self.config.encoding)))
        _category_version, _span_names, ner_names = resolve_label_space_from_config(
            self.config_json, context=str(Path(self.checkpoint) / "config.json")
        )
        self.label_info = build_label_info(ner_names)
        self.decoder, _biases = build_sequence_decoder(
            decode_mode=str(decode_mode),
            label_info=self.label_info,
            viterbi_calibration_path=None,
            checkpoint_dir=self.checkpoint,
        )
        self.model = Transformer.from_checkpoint(self.checkpoint, moe_chunk_size=moe_chunk_size)

    def predict_text(self, text: str) -> PredictionResult:
        token_ids = tuple(int(tok) for tok in self.encoding.encode(text, allowed_special="all"))
        if not token_ids:
            return PredictionResult(text=text, spans=(), decoded_mismatch=False)

        background = int(self.label_info.background_token_label)
        example = TokenizedExample(
            tokens=token_ids,
            labels=tuple(background for _ in token_ids),
            example_id="mlx-example",
            text=text,
        )
        token_positions: list[int] = []
        score_vectors: list[torch.Tensor] = []
        for window in example_to_windows(example, self.n_ctx):
            if not window.tokens:
                continue
            token_array = mx.array([list(window.tokens)], dtype=mx.int32)
            logits = self.model(token_array).astype(mx.float32)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            log_probs_np = np.array(log_probs[0])
            if log_probs_np.shape[0] != len(window.tokens):
                raise ValueError("Logprob output length does not match window length")
            for token_pos, is_valid in enumerate(window.mask):
                if not bool(is_valid):
                    continue
                token_positions.append(int(window.offsets[token_pos]))
                score_vectors.append(torch.from_numpy(log_probs_np[token_pos].astype(np.float32)))

        if not score_vectors:
            return PredictionResult(text=text, spans=(), decoded_mismatch=False)
        stacked_scores = torch.stack(score_vectors, dim=0)
        if self.decoder is not None:
            decoded_labels = self.decoder.decode(stacked_scores)
            if len(decoded_labels) != len(token_positions):
                decoded_labels = stacked_scores.argmax(dim=1).tolist()
        else:
            decoded_labels = stacked_scores.argmax(dim=1).tolist()
        predicted_labels_by_index = {
            token_idx: int(label) for token_idx, label in zip(token_positions, decoded_labels)
        }
        predicted_token_spans = labels_to_spans(predicted_labels_by_index, self.label_info)

        decoded_text, char_starts, char_ends = decode_text_with_offsets(token_ids, self.encoding)
        decoded_mismatch = decoded_text != text
        source_text = decoded_text if decoded_mismatch else text
        predicted_char_spans = token_spans_to_char_spans(predicted_token_spans, char_starts, char_ends)
        if self.trim_whitespace:
            predicted_char_spans = trim_char_spans_whitespace(predicted_char_spans, source_text)
        if self.discard_overlapping_predicted_spans:
            predicted_char_spans = discard_overlapping_spans_by_label(predicted_char_spans)

        detected: list[DetectedSpan] = []
        for label_idx, start, end in predicted_char_spans:
            if not (0 <= start < end <= len(source_text)):
                continue
            label = str(self.label_info.span_class_names[int(label_idx)])
            normalized = label.upper().replace("-", "_")
            detected.append(
                DetectedSpan(
                    label=label,
                    start=int(start),
                    end=int(end),
                    text=source_text[start:end],
                    placeholder=f"<{normalized}>",
                )
            )
        detected = _select_non_overlapping_spans(detected)
        if self.output_mode == "redacted":
            detected = [
                DetectedSpan(
                    label="redacted",
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    placeholder="<REDACTED>",
                )
                for s in detected
            ]
        return PredictionResult(text=source_text, spans=tuple(detected), decoded_mismatch=decoded_mismatch)

    def redact(self, text: str) -> str | RedactionResult:
        prediction = self.predict_text(text)
        redacted_text = _redact_text(prediction.text, prediction.spans)
        if self.output_text_only:
            return redacted_text
        labels = [span.label for span in prediction.spans]
        return RedactionResult(
            schema_version=SCHEMA_VERSION,
            summary=build_detection_summary(
                output_mode=self.output_mode,
                labels=labels,
                decoded_mismatch=prediction.decoded_mismatch,
            ),
            text=prediction.text,
            detected_spans=prediction.spans,
            redacted_text=redacted_text,
            warning=_warning_for_prediction(prediction),
        )


def _select_non_overlapping_spans(spans: Sequence[DetectedSpan]) -> list[DetectedSpan]:
    ordered = sorted(spans, key=lambda span: (span.start, -(span.end - span.start), span.label))
    kept: list[DetectedSpan] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor or span.end <= span.start:
            continue
        kept.append(span)
        cursor = span.end
    return kept
