# swift-mlx — OPF MLX Swift spike

Native Swift + [MLX Swift](https://github.com/ml-explore/mlx-swift) spike for the
OpenAI Privacy Filter (OPF) checkpoint. The Swift binary now owns the full
inference pipeline: it loads the `o200k_base` tokenizer (via HuggingFace
[swift-transformers](https://github.com/huggingface/swift-transformers)), runs
the OPF transformer in MLX, decodes labels with argmax or Viterbi using
`viterbi_calibration.json`, and renders spans + redacted text — matching the
Python MLX reference at `mlx_convert/opf_mlx.py` and the OPF Python pipeline.
There is still no batch>1 path and no SwiftPM-only build (MLX needs Xcode).

## Layout

```
swift-mlx/
  Package.swift
  README.md
  Sources/OPFMLXDaemon/
    main.swift      # CLI dispatcher
    Model.swift     # OPF transformer: config, RoPE, attention, MoE, blocks
    Forward.swift   # `forward` subcommand (token-list one-shot)
    Daemon.swift    # legacy `serve-tokens` Unix-socket token daemon
    Tokenizer.swift # swift-transformers wrapper + byte-level offsets
    Labels.swift    # OPF v2 label space (33 BIESO classes)
    Spans.swift     # labels-to-spans, char spans, redaction renderer
    Viterbi.swift   # CRF Viterbi decoder + calibration loader
    Pipeline.swift  # text -> tokens -> model -> spans -> redacted_text
    Redact.swift    # `redact` one-shot text command
    Serve.swift     # `serve` text daemon (Unix socket, JSON over text)
    Tokenize.swift  # `tokenize` diagnostic helper (no model load)
```

The executable product is `opf-mlx-daemon`. It implements:

```
opf-mlx-daemon inspect [--checkpoint <path>]
opf-mlx-daemon forward --tokens <id1,id2,...>
                       [--checkpoint <path>] [--context <N>] [--moe-chunk-size <N>]
opf-mlx-daemon serve-tokens [--checkpoint <path>] [--socket <path>]
                            [--context <N>] [--moe-chunk-size <N>]
opf-mlx-daemon redact --text <text> [--checkpoint <path>] [--context <N>]
                      [--moe-chunk-size <N>] [--decode-mode argmax|viterbi]
                      [--tokenizer <path>] [--no-download]
opf-mlx-daemon serve [--checkpoint <path>] [--socket <path>]
                     [--context <N>] [--moe-chunk-size <N>]
                     [--decode-mode argmax|viterbi]
                     [--tokenizer <path>] [--no-download]
opf-mlx-daemon tokenize --text <text> [--checkpoint <path>] [--tokenizer <path>]
```

## Requirements

- Apple Silicon Mac (MLX uses Metal).
- macOS 14+ (matches MLX Swift's deployment target).
- Xcode 15+ / Swift 5.10+ toolchain.

The default OPF checkpoint is expected at `~/.opf/privacy_filter` and contains:

- `config.json`
- `model.safetensors`
- `viterbi_calibration.json`

If you don't have it, run any OPF Python entry point (e.g. `opf` CLI) once to
let `ensure_default_checkpoint` populate it.

A `tokenizer.json` is also required for the text commands (`redact`, `serve`,
`tokenize`). If `<checkpoint>/tokenizer.json` is missing, the binary downloads
it from
`https://huggingface.co/openai/privacy-filter/resolve/main/tokenizer.json`
on first use. Pass `--no-download` to disable that, or `--tokenizer <path>`
to load it from elsewhere.

## Build

> **Important**: `swift build` / `swift run` cannot fully build MLX Swift on
> macOS. SwiftPM's command-line build does not compile the Metal shaders that
> MLX needs at startup, so the binary it produces fails immediately with
> `Failed to load the default metallib`. This is a known limitation of MLX
> Swift (see the upstream README) — use `xcodebuild` instead.

### Recommended: xcodebuild

```bash
cd swift-mlx
xcodebuild build \
  -scheme OPFMLXDaemon \
  -destination 'platform=macOS' \
  -derivedDataPath .build/xcode
```

The first build resolves the `mlx-swift` SwiftPM dependency and compiles MLX
itself, which takes a few minutes. Subsequent builds are incremental.

The build also produces `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`
next to the binary, which MLX picks up at runtime via SwiftPM bundle search.

### Run inspect

```bash
.build/xcode/Build/Products/Debug/opf-mlx-daemon inspect \
  --checkpoint ~/.opf/privacy_filter
```

Or, with the default path:

```bash
.build/xcode/Build/Products/Debug/opf-mlx-daemon inspect
```

The output prints:

- the checkpoint path,
- a few key fields from `config.json` (`model_type`, `encoding`,
  `num_hidden_layers`, `hidden_size`, `num_experts`, `experts_per_token`,
  `num_labels`, `vocab_size`, `bidirectional_{left,right}_context`),
- the number of tensors loaded and total parameter count,
- the first ~20 tensor names with shape + dtype,
- presence checks for `embedding.weight`, `block.0.attn.qkv.weight`,
  `block.0.mlp.swiglu.weight`, `norm.scale`, and `unembedding.weight`.

The process exits non-zero if the config or safetensors file is missing.

### Run forward

```bash
.build/xcode/Build/Products/Debug/opf-mlx-daemon forward \
  --checkpoint ~/.opf/privacy_filter \
  --tokens 9906,673,9981 \
  --context 64 \
  --moe-chunk-size 1
```

Output:

```
input tokens: 3
logits shape: [1, 3, 33]
argmax labels: 0,0,0
latency_ms: 3298.35
```

`forward` parses comma-separated o200k_base token ids, runs one batch=1 forward
through the OPF transformer (8 blocks, 128 experts, top-4 routing, YaRN RoPE,
local bidirectional attention with sink logits), and prints:

- the input token count,
- the logits shape `[1, T, num_labels]` (33 labels),
- the per-token argmax label list,
- the wall time of the forward pass in ms.

Non-zero exit on missing files, missing tensors, shape mismatches, or
out-of-range token ids.

### Run redact (one-shot text)

```bash
.build/xcode-release/Build/Products/Release/opf-mlx-daemon redact \
  --text "Alice was born on 1990-01-02. Her email is alice@example.com." \
  --checkpoint ~/.opf/privacy_filter \
  --context 128 --moe-chunk-size 2 --decode-mode viterbi
```

Output (pretty-printed JSON):

```json
{
  "text": "Alice was born on 1990-01-02. Her email is alice@example.com.",
  "detected_spans": [
    {"label": "private_date",  "start": 18, "end": 28, "text": "1990-01-02",       "placeholder": "<PRIVATE_DATE>"},
    {"label": "private_email", "start": 43, "end": 60, "text": "alice@example.com","placeholder": "<PRIVATE_EMAIL>"}
  ],
  "redacted_text": "Alice was born on <PRIVATE_DATE>. Her email is <PRIVATE_EMAIL>.",
  "latency_ms": 888.5,
  "token_count": 19,
  "decode_mode": "viterbi"
}
```

`redact` performs the full pipeline end-to-end:

1. Tokenize with `swift-transformers` (`o200k_base` BPE, byte-level pre/decoder),
   matching `tiktoken.get_encoding("o200k_base")` token ids exactly.
2. Run the OPF transformer in fixed-size windows of `--context` tokens.
3. Decode either argmax or Viterbi (using `<checkpoint>/viterbi_calibration.json`
   or zero biases if missing).
4. Convert token labels to character spans (BIESO rules), trim whitespace, and
   pick a non-overlapping left-to-right subset.
5. Render `redacted_text` using `<LABEL>` placeholders (e.g. `<PRIVATE_DATE>`).

### Run serve (text daemon)

```bash
.build/xcode-release/Build/Products/Release/opf-mlx-daemon serve \
  --checkpoint ~/.opf/privacy_filter \
  --socket /tmp/opf-mlx-swift-text.sock \
  --context 128 --moe-chunk-size 2 --decode-mode viterbi
```

The daemon loads `config.json`, `model.safetensors`, `tokenizer.json`, and
`viterbi_calibration.json` once at startup, then handles requests sequentially
over a Unix domain socket. Newline-delimited JSON in both directions.

Request:

```json
{"text":"Alice was born on 1990-01-02.", "request_id":"optional", "decode_mode":"viterbi"}
```

Successful response:

```json
{
  "request_id": "optional",
  "text": "Alice was born on 1990-01-02.",
  "detected_spans": [
    {"label":"private_date","start":18,"end":28,"text":"1990-01-02","placeholder":"<PRIVATE_DATE>"}
  ],
  "redacted_text": "Alice was born on <PRIVATE_DATE>.",
  "latency_ms": 12.3,
  "token_count": 12,
  "decode_mode": "viterbi"
}
```

`decode_mode` in the request is optional and overrides the daemon's startup
default for that request.

Error response:

```json
{"request_id":"optional", "error":"missing or non-string 'text'"}
```

Smoke test:

```bash
python3 - <<'PY'
import socket, json
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect('/tmp/opf-mlx-swift-text.sock')
s.sendall(json.dumps({'text':'Alice was born on 1990-01-02. Her email is alice@example.com.','request_id':'smoke'}).encode() + b'\n')
buf = b''
while b'\n' not in buf:
    buf += s.recv(65536)
print(buf.decode())
PY
```

### Run serve-tokens (token-only daemon, legacy)

```bash
.build/xcode/Build/Products/Debug/opf-mlx-daemon serve-tokens \
  --checkpoint ~/.opf/privacy_filter \
  --socket /tmp/opf-mlx-swift.sock \
  --context 128 \
  --moe-chunk-size 2
```

The daemon loads `config.json` + `model.safetensors` and builds the OPF
transformer once at startup, then listens on a Unix domain socket. It speaks a
**newline-delimited JSON** protocol — one JSON object per line, in both
directions. Connections may send multiple requests; requests are handled
sequentially.

Request:

```json
{"tokens":[13225,11,922], "request_id":"optional"}
```

Successful response:

```json
{"request_id":"optional","labels":[0,0,0],"latency_ms":12.3}
```

Error response:

```json
{"request_id":"optional","error":"tokens length 256 exceeds --context 128"}
```

The daemon validates that `tokens` is a non-empty array of integers in
`[0, vocab_size)` and that `len(tokens) <= --context`. `latency_ms` measures the
forward pass only (model load and socket I/O are excluded). Stale socket files
at `--socket` are unlinked on startup.

Smoke test from another shell:

```bash
python3 - <<'PY'
import socket, json
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect('/tmp/opf-mlx-swift.sock')
s.sendall(json.dumps({'tokens': [9906, 673, 9981], 'request_id': 'smoke'}).encode() + b'\n')
buf = b''
while b'\n' not in buf:
    buf += s.recv(4096)
print(buf.decode())
PY
```

Expected `labels` is `[0,0,0]` (the same value the `forward` subcommand prints
for these tokens).

#### Verifying parity with Python MLX

The Swift forward should match the Python MLX reference token-for-token. Quick
check from the repo root:

```bash
uv run --python 3.12 --with mlx python - <<'PY'
import sys, numpy as np
sys.path.insert(0, 'mlx_convert')
import mlx.core as mx
from opf_mlx import Transformer
ids = [9906, 673, 9981]
model = Transformer.from_checkpoint('/Users/aleksandrbobrov/.opf/privacy_filter', moe_chunk_size=1)
logits = model(mx.array([ids], dtype=mx.int32)).astype(mx.float32)
labels = np.array(mx.argmax(logits[0], axis=-1)).tolist()
print('python_labels=', ','.join(map(str, labels)))
print('shape=', logits.shape)
PY
```

Last verified parity (2026-04-28):

| Tokens                                                                             | Swift labels                                       | Python labels                                      |
| ---------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| `9906,673,9981`                                                                    | `0,0,0`                                            | `0,0,0`                                            |
| `13225,11,922,1308,382,5928,16627,326,922,3719,382,64626,640,68671,81309,1136`     | `0,0,0,0,0,17,19,0,0,0,0,13,14,14,14,15`           | `0,0,0,0,0,17,19,0,0,0,0,13,14,14,14,15`           |

The 16-token example is the o200k_base tokenization of
`"Hello, my name is John Smith and my email is john.smith@example.com"`.

### `swift build` (compile-only check)

`swift build` still works for verifying that the code compiles, but the
resulting `.build/debug/opf-mlx-daemon` will fail at runtime as described
above. Don't run it; use the xcodebuild output.

## Caveats / known limitations

- Batch size is hardcoded to 1 by the CLI. The model code itself does not
  fundamentally require batch=1 (shapes are parameterized), but it has only
  been exercised at batch=1.
- Tokenizer offsets are computed by decoding each token individually to its
  UTF-8 bytes (via `swift-transformers`'s ByteLevel decoder) and matching
  byte ranges in the input — the same algorithm `tiktoken` uses on the Python
  side. Token ids match `tiktoken.get_encoding("o200k_base")` exactly on the
  test sentences (see "Verifying parity" below).
- Latency is dominated by the MoE block. With `--moe-chunk-size 1` (the
  default and the safest value) each token does 4 expert forward passes; the
  first forward also pays a model load + Metal kernel JIT cost. Subsequent
  forwards in the same process are much faster than the first. `2` is a safe
  speedup; larger values have not been tuned.
- The local SDPA window is materialised explicitly via `take` + `einsum` on a
  zero-padded K/V copy, mirroring the Python MLX reference. This is fine for
  short contexts (<= 128) but is not the most efficient layout for longer
  sequences.
- `viterbi_calibration.json` is honored when present, but the default OPF
  checkpoint ships zero biases — Viterbi here mainly enforces BIESO transition
  structure.

## Status / next steps

- [x] Package skeleton with MLX Swift + swift-transformers dependencies.
- [x] `inspect` subcommand reading config + safetensors.
- [x] Model forward pass with YaRN RoPE, local bidirectional attention with
      sink logits, MoE SwiGLU, and final unembedding — argmax-equivalent to
      the Python MLX reference on small inputs.
- [x] Token-only `serve-tokens` daemon (legacy compatibility).
- [x] Native Swift tokenizer (`swift-transformers` loading `tokenizer.json`),
      with byte-level char-offset reconstruction.
- [x] CRF Viterbi decoder ported from `opf._core.decoding`, BIESO transition
      constraints over 33 OPF v2 classes, calibration biases loaded from
      `viterbi_calibration.json`.
- [x] `redact` one-shot CLI and `serve` long-lived text daemon (Unix socket,
      newline-delimited JSON; text in, detected spans + redacted_text out).
- [ ] Performance: optimize MoE expert dispatch and the local SDPA window.
