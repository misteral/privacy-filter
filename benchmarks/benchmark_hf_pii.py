"""Benchmark OPF backends on a Hugging Face PII dataset.

Compares the original PyTorch OPF backend against the experimental MLX backend
on accuracy (exact span+label and any-PII span F1) and speed (examples/sec,
chars/sec, tokens/sec).
"""

from __future__ import annotations

import argparse
import json
import resource
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _peak_memory_mb() -> float:
    """Peak resident-set size of the current process in MB.

    `ru_maxrss` is bytes on macOS but kilobytes on Linux/BSD — normalize to MB.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


REPO_ROOT = Path(__file__).resolve().parent.parent
MLX_CONVERT_DIR = REPO_ROOT / "mlx_convert"


SOURCE_LABEL_TO_OPF: dict[str, str] = {
    # private_person
    "FIRSTNAME": "private_person",
    "MIDDLENAME": "private_person",
    "LASTNAME": "private_person",
    "TITLE": "private_person",
    # private_email
    "EMAIL": "private_email",
    # private_phone
    "PHONENUMBER": "private_phone",
    # private_date
    "DATE": "private_date",
    "DOB": "private_date",
    "TIME": "private_date",
    "AGE": "private_date",
    # private_address
    "STREET": "private_address",
    "BUILDINGNUMBER": "private_address",
    "CITY": "private_address",
    "STATE": "private_address",
    "ZIPCODE": "private_address",
    # account_number
    "ACCOUNTNUMBER": "account_number",
    "AADHAAR": "account_number",
    "BIC": "account_number",
    "BITCOINADDRESS": "account_number",
    "CREDITCARDISSUER": "account_number",
    "CREDITCARDNUMBER": "account_number",
    "DRIVERLICENSENUM": "account_number",
    "ETHEREUMADDRESS": "account_number",
    "IBAN": "account_number",
    "IDCARDNUM": "account_number",
    "IPV4": "account_number",
    "MASKEDNUMBER": "account_number",
    "NATIONAL_INSURANCE": "account_number",
    "PASSPORTNUM": "account_number",
    "SSN": "account_number",
    "TAXNUM": "account_number",
    "UPI_ID": "account_number",
    "USERNAME": "account_number",
    "VEHICLEVIN": "account_number",
    "VEHICLEVRM": "account_number",
    "BUSINESS_REGISTRATION": "account_number",
    # secret
    "PASSWORD": "secret",
    "PIN": "secret",
    "CVV": "secret",
}


@dataclass
class GoldExample:
    """One benchmark example with gold spans mapped to OPF labels."""

    text: str
    spans: list[tuple[str, int, int]]  # (opf_label, start, end)
    n_chars: int
    n_tokens: int | None


@dataclass
class BackendMetrics:
    """Aggregated metrics for one backend run."""

    name: str
    load_seconds: float
    inference_seconds: float
    n_examples: int
    n_chars: int
    n_tokens: int
    n_gold_spans: int
    n_pred_spans: int
    exact_tp: int
    any_tp: int
    warmup_examples: int = 0
    peak_memory_mb: float | None = None
    error: str | None = None
    per_example: list[dict[str, Any]] = field(default_factory=list)

    def derived(self) -> dict[str, float | int | str | None]:
        total_pred = self.n_pred_spans
        total_gold = self.n_gold_spans
        exact_p = (self.exact_tp / total_pred) if total_pred > 0 else None
        exact_r = (self.exact_tp / total_gold) if total_gold > 0 else None
        exact_f1 = _f1(exact_p, exact_r)
        any_p = (self.any_tp / total_pred) if total_pred > 0 else None
        any_r = (self.any_tp / total_gold) if total_gold > 0 else None
        any_f1 = _f1(any_p, any_r)
        examples_per_sec = (
            self.n_examples / self.inference_seconds
            if self.inference_seconds > 0 and self.n_examples > 0
            else None
        )
        chars_per_sec = (
            self.n_chars / self.inference_seconds
            if self.inference_seconds > 0 and self.n_chars > 0
            else None
        )
        tokens_per_sec: float | None
        if self.inference_seconds > 0 and self.n_tokens > 0:
            tokens_per_sec = self.n_tokens / self.inference_seconds
        else:
            tokens_per_sec = None
        return {
            "backend": self.name,
            "load_seconds": self.load_seconds,
            "inference_seconds": self.inference_seconds,
            "warmup_examples": self.warmup_examples,
            "n_examples": self.n_examples,
            "n_chars": self.n_chars,
            "n_tokens": self.n_tokens,
            "n_gold_spans": self.n_gold_spans,
            "n_pred_spans": self.n_pred_spans,
            "exact_tp": self.exact_tp,
            "any_tp": self.any_tp,
            "exact.precision": exact_p,
            "exact.recall": exact_r,
            "exact.f1": exact_f1,
            "any.precision": any_p,
            "any.recall": any_r,
            "any.f1": any_f1,
            "examples_per_sec": examples_per_sec,
            "chars_per_sec": chars_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "error": self.error,
        }


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    if p + r <= 0:
        return 0.0
    return 2 * p * r / (p + r)


def _normalize_privacy_mask(value: object) -> list[dict[str, Any]]:
    """Return privacy_mask as a list of dicts (some HF rows store it as JSON string)."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _normalize_privacy_mask(parsed)
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.append(item)
        return out
    return []


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_gold_spans(
    text: str,
    privacy_mask: list[dict[str, Any]],
) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    text_len = len(text)
    for entry in privacy_mask:
        raw_label = entry.get("label")
        if not isinstance(raw_label, str):
            continue
        opf_label = SOURCE_LABEL_TO_OPF.get(raw_label.strip().upper())
        if opf_label is None:
            continue
        start = _coerce_int(entry.get("start"))
        end = _coerce_int(entry.get("end"))
        if start is None or end is None:
            continue
        if start < 0 or end <= start or end > text_len:
            continue
        spans.append((opf_label, start, end))
    spans.sort(key=lambda item: (item[1], item[2], item[0]))
    return spans


def _load_examples(
    *,
    dataset_name: str,
    split: str,
    limit: int | None,
    encoder: Any | None,
) -> list[GoldExample]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install with `uv pip install datasets`"
            " or run with `uv run --with datasets ...`."
        ) from exc

    print(f"Loading dataset {dataset_name!r} split={split!r}...", flush=True)
    ds = load_dataset(dataset_name, split=split)
    examples: list[GoldExample] = []
    n = len(ds) if limit is None else min(limit, len(ds))
    for idx in range(n):
        row = ds[idx]
        text = row.get("source_text")
        if not isinstance(text, str) or not text:
            continue
        privacy_mask = _normalize_privacy_mask(row.get("privacy_mask"))
        gold_spans = _build_gold_spans(text, privacy_mask)
        token_count: int | None = None
        if encoder is not None:
            try:
                token_count = len(encoder.encode(text, allowed_special="all"))
            except Exception:
                token_count = None
        examples.append(
            GoldExample(
                text=text,
                spans=gold_spans,
                n_chars=len(text),
                n_tokens=token_count,
            )
        )
    return examples


def _try_get_tokenizer() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def _build_original_backend(args: argparse.Namespace) -> Any:
    from opf import OPF

    return OPF(
        model=args.checkpoint,
        context_window_length=args.context,
        device=args.device,
        decode_mode=args.decode_mode,
    )


def _build_mlx_backend(args: argparse.Namespace) -> Any:
    if str(MLX_CONVERT_DIR) not in sys.path:
        sys.path.insert(0, str(MLX_CONVERT_DIR))
    from opf_mlx import OPFMLX  # type: ignore[import-not-found]

    return OPFMLX(
        model=args.checkpoint,
        context_window_length=args.context,
        moe_chunk_size=args.moe_chunk_size,
        decode_mode=args.decode_mode,
    )


@dataclass
class _SwiftDaemonSpan:
    label: str
    start: int
    end: int


@dataclass
class _SwiftDaemonResult:
    detected_spans: list[_SwiftDaemonSpan]


class SwiftDaemonBackend:
    """Benchmark adapter that talks to the Swift opf-mlx-daemon over a Unix socket.

    The daemon owns the model and returns argmax token labels for a batch=1
    window of pre-tokenized o200k_base ids. This adapter handles tokenization,
    windowing, label-to-span conversion, and span trimming using the existing
    OPF helpers — so the result shape matches what the benchmark expects from
    the original/mlx backends (an object with ``detected_spans``).
    """

    def __init__(
        self,
        *,
        checkpoint: str | None,
        n_ctx: int,
        socket_path: str,
    ) -> None:
        # Imports are deferred so that benchmark startup doesn't require these
        # packages until the swift-mlx-daemon backend is actually selected.
        import tiktoken
        from opf._common.checkpoint_download import ensure_default_checkpoint
        from opf._common.label_space import resolve_label_space_from_config
        from opf._core.sequence_labeling import build_label_info

        if checkpoint is None:
            checkpoint_path = Path(ensure_default_checkpoint())
        else:
            checkpoint_path = Path(checkpoint).expanduser()
        config_path = checkpoint_path / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            self.config_json = json.load(handle)

        self.checkpoint = str(checkpoint_path)
        self.n_ctx = int(n_ctx)
        if self.n_ctx <= 0:
            raise ValueError("context_window_length must be positive")

        encoding_name = str(self.config_json.get("encoding", "o200k_base"))
        self.encoding = tiktoken.get_encoding(encoding_name)

        _category_version, _span_names, ner_names = resolve_label_space_from_config(
            self.config_json, context=str(config_path)
        )
        self.label_info = build_label_info(ner_names)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._recv_buf = bytearray()

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def _send_request(self, tokens: Sequence[int]) -> list[int]:
        payload = json.dumps({"tokens": list(tokens)}).encode("utf-8")
        self._sock.sendall(payload + b"\n")
        line = self._read_line()
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"daemon returned invalid JSON: {line!r}") from exc
        if "error" in response:
            raise RuntimeError(f"daemon error: {response['error']}")
        labels = response.get("labels")
        if not isinstance(labels, list):
            raise RuntimeError(f"daemon response missing labels: {response!r}")
        if len(labels) != len(tokens):
            raise RuntimeError(
                f"daemon returned {len(labels)} labels for window of {len(tokens)} tokens"
            )
        return [int(v) for v in labels]

    def _read_line(self) -> bytes:
        while True:
            idx = self._recv_buf.find(b"\n")
            if idx >= 0:
                line = bytes(self._recv_buf[:idx])
                del self._recv_buf[: idx + 1]
                return line
            chunk = self._sock.recv(65536)
            if not chunk:
                raise RuntimeError("daemon closed connection")
            self._recv_buf.extend(chunk)

    def redact(self, text: str) -> _SwiftDaemonResult:
        from opf._core.sequence_labeling import TokenizedExample, example_to_windows
        from opf._core.spans import (
            decode_text_with_offsets,
            labels_to_spans,
            token_spans_to_char_spans,
            trim_char_spans_whitespace,
        )

        token_ids = tuple(int(tok) for tok in self.encoding.encode(text, allowed_special="all"))
        if not token_ids:
            return _SwiftDaemonResult(detected_spans=[])

        background = int(self.label_info.background_token_label)
        example = TokenizedExample(
            tokens=token_ids,
            labels=tuple(background for _ in token_ids),
            example_id="swift-mlx-example",
            text=text,
        )

        labels_by_index: dict[int, int] = {}
        for window in example_to_windows(example, self.n_ctx):
            if not window.tokens:
                continue
            window_labels = self._send_request(window.tokens)
            for token_pos, is_valid in enumerate(window.mask):
                if not bool(is_valid):
                    continue
                labels_by_index[int(window.offsets[token_pos])] = int(window_labels[token_pos])

        if not labels_by_index:
            return _SwiftDaemonResult(detected_spans=[])

        token_spans = labels_to_spans(labels_by_index, self.label_info)
        decoded_text, char_starts, char_ends = decode_text_with_offsets(token_ids, self.encoding)
        source_text = decoded_text if decoded_text != text else text
        char_spans = token_spans_to_char_spans(token_spans, char_starts, char_ends)
        char_spans = trim_char_spans_whitespace(char_spans, source_text)

        detected: list[_SwiftDaemonSpan] = []
        for label_idx, start, end in char_spans:
            if not (0 <= start < end <= len(source_text)):
                continue
            label = str(self.label_info.span_class_names[int(label_idx)])
            detected.append(_SwiftDaemonSpan(label=label, start=int(start), end=int(end)))
        return _SwiftDaemonResult(detected_spans=_select_non_overlapping(detected))


def _select_non_overlapping(spans: Sequence[_SwiftDaemonSpan]) -> list[_SwiftDaemonSpan]:
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), s.label))
    kept: list[_SwiftDaemonSpan] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor or span.end <= span.start:
            continue
        kept.append(span)
        cursor = span.end
    return kept


def _build_swift_daemon_backend(args: argparse.Namespace) -> Any:
    return SwiftDaemonBackend(
        checkpoint=args.checkpoint,
        n_ctx=args.context,
        socket_path=args.swift_socket,
    )


class SwiftTextDaemonBackend:
    """Benchmark adapter that talks to the full Swift text daemon over a Unix socket.

    Unlike `SwiftDaemonBackend`, this backend sends raw text and trusts the Swift
    daemon to tokenize, run the model, decode (argmax/Viterbi), and return the
    final character-level detected spans. The daemon's startup `--decode-mode`
    sets the default; per-request `decode_mode` overrides it.
    """

    def __init__(self, *, socket_path: str, decode_mode: str | None) -> None:
        self.socket_path = socket_path
        self.decode_mode = decode_mode
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._recv_buf = bytearray()

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def _read_line(self) -> bytes:
        while True:
            idx = self._recv_buf.find(b"\n")
            if idx >= 0:
                line = bytes(self._recv_buf[:idx])
                del self._recv_buf[: idx + 1]
                return line
            chunk = self._sock.recv(65536)
            if not chunk:
                raise RuntimeError("daemon closed connection")
            self._recv_buf.extend(chunk)

    def redact(self, text: str) -> _SwiftDaemonResult:
        payload: dict[str, Any] = {"text": text}
        if self.decode_mode is not None:
            payload["decode_mode"] = self.decode_mode
        self._sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        line = self._read_line()
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"daemon returned invalid JSON: {line!r}") from exc
        if "error" in response:
            raise RuntimeError(f"daemon error: {response['error']}")
        spans_raw = response.get("detected_spans", [])
        spans: list[_SwiftDaemonSpan] = []
        for entry in spans_raw:
            label = entry.get("label")
            start = entry.get("start")
            end = entry.get("end")
            if not isinstance(label, str) or start is None or end is None:
                continue
            spans.append(_SwiftDaemonSpan(label=label, start=int(start), end=int(end)))
        return _SwiftDaemonResult(detected_spans=spans)


def _build_swift_text_daemon_backend(args: argparse.Namespace) -> Any:
    return SwiftTextDaemonBackend(
        socket_path=args.swift_text_socket,
        decode_mode=args.decode_mode,
    )


def _spans_from_result(result: Any) -> list[tuple[str, int, int]]:
    detected = getattr(result, "detected_spans", None)
    if detected is None:
        return []
    out: list[tuple[str, int, int]] = []
    for span in detected:
        label = getattr(span, "label", None)
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        if not isinstance(label, str) or start is None or end is None:
            continue
        out.append((label, int(start), int(end)))
    return out


def _count_matches(
    pred: Sequence[tuple[str, int, int]],
    gold: Sequence[tuple[str, int, int]],
) -> tuple[int, int]:
    """Return (exact_tp, any_tp) one-to-one matches between pred and gold."""
    exact_tp = 0
    used_gold_exact: set[int] = set()
    for p_lbl, p_s, p_e in pred:
        for g_idx, (g_lbl, g_s, g_e) in enumerate(gold):
            if g_idx in used_gold_exact:
                continue
            if p_lbl == g_lbl and p_s == g_s and p_e == g_e:
                used_gold_exact.add(g_idx)
                exact_tp += 1
                break

    any_tp = 0
    used_gold_any: set[int] = set()
    for _p_lbl, p_s, p_e in pred:
        for g_idx, (_g_lbl, g_s, g_e) in enumerate(gold):
            if g_idx in used_gold_any:
                continue
            if p_s == g_s and p_e == g_e:
                used_gold_any.add(g_idx)
                any_tp += 1
                break
    return exact_tp, any_tp


def _run_backend(
    *,
    name: str,
    builder,
    examples: list[GoldExample],
    warmup: int,
) -> BackendMetrics:
    print(f"\n[{name}] loading backend...", flush=True)
    load_start = time.perf_counter()
    try:
        backend = builder()
    except Exception as exc:  # noqa: BLE001 - surface load errors without aborting peers
        load_seconds = time.perf_counter() - load_start
        return BackendMetrics(
            name=name,
            load_seconds=load_seconds,
            inference_seconds=0.0,
            n_examples=0,
            n_chars=0,
            n_tokens=0,
            n_gold_spans=0,
            n_pred_spans=0,
            exact_tp=0,
            any_tp=0,
            warmup_examples=0,
            error=f"load failed: {exc!r}",
        )
    load_seconds = time.perf_counter() - load_start
    print(f"[{name}] loaded in {load_seconds:.2f}s", flush=True)

    effective_warmup = max(0, min(warmup, max(0, len(examples) - 1)))
    if effective_warmup != warmup:
        print(
            f"[{name}] reducing warmup {warmup} -> {effective_warmup} so at least one"
            " example is timed.",
            flush=True,
        )

    if effective_warmup > 0:
        print(f"[{name}] warmup {effective_warmup} example(s)...", flush=True)
        for ex in examples[:effective_warmup]:
            try:
                backend.redact(ex.text)
            except Exception as exc:  # noqa: BLE001 - keep warmup non-fatal
                print(f"[{name}] warmup error: {exc!r}", flush=True)

    timed = examples[effective_warmup:]
    n_examples = len(timed)
    n_chars = 0
    n_tokens = 0
    n_gold_spans = 0
    n_pred_spans = 0
    exact_tp = 0
    any_tp = 0
    per_example: list[dict[str, Any]] = []

    print(f"[{name}] timing {n_examples} example(s)...", flush=True)
    inference_start = time.perf_counter()
    for ex in timed:
        try:
            result = backend.redact(ex.text)
        except Exception as exc:  # noqa: BLE001 - surface bad inputs without aborting
            per_example.append({"error": repr(exc), "text_len": ex.n_chars})
            continue
        pred_spans = _spans_from_result(result)
        ex_exact, ex_any = _count_matches(pred_spans, ex.spans)
        n_chars += ex.n_chars
        if ex.n_tokens is not None:
            n_tokens += ex.n_tokens
        n_gold_spans += len(ex.spans)
        n_pred_spans += len(pred_spans)
        exact_tp += ex_exact
        any_tp += ex_any
    inference_seconds = time.perf_counter() - inference_start
    print(
        f"[{name}] done: {n_examples} examples in {inference_seconds:.2f}s",
        flush=True,
    )

    return BackendMetrics(
        name=name,
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        n_examples=n_examples,
        n_chars=n_chars,
        n_tokens=n_tokens,
        n_gold_spans=n_gold_spans,
        n_pred_spans=n_pred_spans,
        exact_tp=exact_tp,
        any_tp=any_tp,
        warmup_examples=effective_warmup,
        peak_memory_mb=_peak_memory_mb(),
        per_example=per_example,
    )


def _format_optional_float(value: float | None, fmt: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def _print_backend_section(metrics: BackendMetrics) -> None:
    derived = metrics.derived()
    print(f"\n=== {metrics.name} ===")
    if metrics.error:
        print(f"  error: {metrics.error}")
        return
    print(f"  load_seconds:      {metrics.load_seconds:.3f}")
    print(f"  inference_seconds: {metrics.inference_seconds:.3f}")
    print(
        "  warmup/timed:      "
        f"{metrics.warmup_examples} warmup / {metrics.n_examples} timed"
    )
    print(f"  gold_spans:        {metrics.n_gold_spans}")
    print(f"  pred_spans:        {metrics.n_pred_spans}")
    print(f"  exact_tp/any_tp:   {metrics.exact_tp} / {metrics.any_tp}")
    print(f"  exact.precision:   {_format_optional_float(derived['exact.precision'])}")
    print(f"  exact.recall:      {_format_optional_float(derived['exact.recall'])}")
    print(f"  exact.f1:          {_format_optional_float(derived['exact.f1'])}")
    print(f"  any.precision:     {_format_optional_float(derived['any.precision'])}")
    print(f"  any.recall:        {_format_optional_float(derived['any.recall'])}")
    print(f"  any.f1:            {_format_optional_float(derived['any.f1'])}")
    print(f"  examples/sec:      {_format_optional_float(derived['examples_per_sec'], '.3f')}")
    print(f"  chars/sec:         {_format_optional_float(derived['chars_per_sec'], '.1f')}")
    print(f"  tokens/sec:        {_format_optional_float(derived['tokens_per_sec'], '.1f')}")
    print(f"  peak_memory_mb:    {_format_optional_float(derived['peak_memory_mb'], '.1f')}")


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return numerator / denominator


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _print_comparison(
    a: BackendMetrics,
    b: BackendMetrics,
) -> dict[str, Any]:
    """Print and return a comparison summary (a - b deltas, a/b speed ratios)."""
    a_d = a.derived()
    b_d = b.derived()
    ratios = {
        "examples_per_sec_ratio": _safe_ratio(
            a_d["examples_per_sec"], b_d["examples_per_sec"]
        ),
        "chars_per_sec_ratio": _safe_ratio(
            a_d["chars_per_sec"], b_d["chars_per_sec"]
        ),
        "tokens_per_sec_ratio": _safe_ratio(
            a_d["tokens_per_sec"], b_d["tokens_per_sec"]
        ),
        "load_seconds_delta": _safe_delta(a_d["load_seconds"], b_d["load_seconds"]),
        "inference_seconds_delta": _safe_delta(
            a_d["inference_seconds"], b_d["inference_seconds"]
        ),
        "exact_f1_delta": _safe_delta(a_d["exact.f1"], b_d["exact.f1"]),
        "any_f1_delta": _safe_delta(a_d["any.f1"], b_d["any.f1"]),
    }
    print(f"\n=== comparison: {a.name} vs {b.name} ===")
    print(
        "  speed (a/b):       "
        f"examples={_format_optional_float(ratios['examples_per_sec_ratio'], '.3f')}x"
        f"  chars={_format_optional_float(ratios['chars_per_sec_ratio'], '.3f')}x"
        f"  tokens={_format_optional_float(ratios['tokens_per_sec_ratio'], '.3f')}x"
    )
    print(
        "  inference_delta:   "
        f"{_format_optional_float(ratios['inference_seconds_delta'], '.3f')}s"
        f"  load_delta={_format_optional_float(ratios['load_seconds_delta'], '.3f')}s"
    )
    print(
        "  f1 deltas (a-b):   "
        f"exact={_format_optional_float(ratios['exact_f1_delta'])}"
        f"  any={_format_optional_float(ratios['any_f1_delta'])}"
    )
    return {
        "a": a.name,
        "b": b.name,
        **ratios,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OPF backends on a HF PII dataset."
    )
    parser.add_argument(
        "--backend",
        choices=("original", "mlx", "swift-mlx-daemon", "swift-mlx-text-daemon", "both"),
        default="both",
        help=(
            "Backend to benchmark. 'both' runs original+mlx; the swift-mlx-daemon "
            "and swift-mlx-text-daemon backends each require a separately-started "
            "Swift daemon process."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="Ari-S-123/pii-detection-english-consolidated",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--decode-mode",
        default="viterbi",
        choices=("viterbi", "argmax"),
    )
    parser.add_argument("--moe-chunk-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--swift-socket",
        type=str,
        default="/tmp/opf-mlx-swift.sock",
        help="Unix socket path for the swift-mlx-daemon (token) backend.",
    )
    parser.add_argument(
        "--swift-text-socket",
        type=str,
        default="/tmp/opf-mlx-swift-text.sock",
        help="Unix socket path for the swift-mlx-text-daemon backend.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.backend == "swift-mlx-daemon" and args.decode_mode != "argmax":
        print(
            "error: --backend swift-mlx-daemon requires --decode-mode argmax "
            "(the token-only Swift daemon does not run Viterbi). Use the "
            "--backend swift-mlx-text-daemon backend for Viterbi.",
            file=sys.stderr,
        )
        return 2

    encoder = _try_get_tokenizer()
    examples = _load_examples(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        encoder=encoder,
    )
    if not examples:
        print("No usable examples loaded; aborting.", file=sys.stderr)
        return 2
    print(
        f"Loaded {len(examples)} example(s); total chars={sum(e.n_chars for e in examples)}",
        flush=True,
    )

    backends_to_run: list[tuple[str, Any]] = []
    if args.backend in ("original", "both"):
        backends_to_run.append(("original", lambda: _build_original_backend(args)))
    if args.backend in ("mlx", "both"):
        backends_to_run.append(("mlx", lambda: _build_mlx_backend(args)))
    if args.backend == "swift-mlx-daemon":
        backends_to_run.append(
            ("swift-mlx-daemon", lambda: _build_swift_daemon_backend(args))
        )
    if args.backend == "swift-mlx-text-daemon":
        backends_to_run.append(
            ("swift-mlx-text-daemon", lambda: _build_swift_text_daemon_backend(args))
        )

    results: dict[str, BackendMetrics] = {}
    for name, builder in backends_to_run:
        results[name] = _run_backend(
            name=name,
            builder=builder,
            examples=examples,
            warmup=args.warmup,
        )

    for name in ("original", "mlx", "swift-mlx-daemon", "swift-mlx-text-daemon"):
        if name in results:
            _print_backend_section(results[name])

    comparison: dict[str, Any] | None = None
    if "original" in results and "mlx" in results and not (
        results["original"].error or results["mlx"].error
    ):
        comparison = _print_comparison(results["mlx"], results["original"])

    if args.output_json:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "n_examples_loaded": len(examples),
            "backends": {name: results[name].derived() for name in results},
        }
        if comparison is not None:
            payload["comparison"] = comparison
        out_path = Path(args.output_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote metrics JSON to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
