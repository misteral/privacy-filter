# OPF benchmarks

Harness for comparing OpenAI Privacy Filter backends against each other on a
public Hugging Face PII dataset. The harness measures both **accuracy**
(exact span+label and any-PII span F1) and **speed** (examples/sec, chars/sec,
tokens/sec).

Default dataset: [`Ari-S-123/pii-detection-english-consolidated`](https://huggingface.co/datasets/Ari-S-123/pii-detection-english-consolidated),
split `test`.

## Install

The benchmark needs the `datasets` package on top of the regular OPF install.
The MLX backend additionally requires `mlx` (Apple Silicon only).

```bash
# regular install
uv pip install -e '.[benchmark]'

# regular + mlx
uv pip install -e '.[benchmark,mlx]'
```

You can also avoid touching the project install and pass extras inline via `uv run`:

```bash
uv run --python 3.12 --with datasets benchmarks/benchmark_hf_pii.py --help
```

## Checkpoint

Both backends look for the same checkpoint directory:

- explicit `--checkpoint /path/to/dir`
- the `OPF_CHECKPOINT` environment variable
- otherwise `~/.opf/privacy_filter`, downloaded on first use

## Usage

```bash
# MLX backend on the first 100 examples
uv run --python 3.12 --with datasets --with mlx benchmarks/benchmark_hf_pii.py \
  --backend mlx --limit 100 --context 128 --moe-chunk-size 2

# Original PyTorch backend (CPU) on the first 100 examples
uv run --python 3.12 --with datasets benchmarks/benchmark_hf_pii.py \
  --backend original --device cpu --limit 100 --context 128

# Both backends head-to-head, write metrics JSON
uv run --python 3.12 --with datasets --with mlx benchmarks/benchmark_hf_pii.py \
  --backend both --limit 100 --context 128 --moe-chunk-size 2 --device cpu \
  --output-json /tmp/opf_bench.json
```

Smoke test (a few examples, tiny context, single MoE chunk):

```bash
uv run --python 3.12 --with datasets --with mlx benchmarks/benchmark_hf_pii.py \
  --backend mlx --limit 3 --context 64 --moe-chunk-size 1
```

## Options

| Flag | Default | Notes |
| --- | --- | --- |
| `--backend` | `both` | One of `original`, `mlx`, `both` |
| `--dataset` | `Ari-S-123/pii-detection-english-consolidated` | Any HF dataset with the same span schema |
| `--split` | `test` | |
| `--limit` | (all) | Cap on examples loaded from the split |
| `--context` | `128` | Token context window length per backend |
| `--checkpoint` | (auto) | Path to OPF checkpoint directory |
| `--device` | `cpu` | Used by the original backend (`cpu` or `cuda`) |
| `--decode-mode` | `viterbi` | `viterbi` or `argmax` |
| `--moe-chunk-size` | `2` | MLX-only MoE chunk size; lower = less RAM, slower |
| `--warmup` | `3` | Examples used to warm caches before timing |
| `--output-json` | (off) | If set, write a metrics JSON file |

When `--limit` is so small that warmup would consume all examples, the harness
automatically reduces warmup so that at least one example is timed.

## Metrics

Per backend:

- **load_seconds** — wall-clock time to construct the backend (model load).
- **inference_seconds** — wall-clock time over the timed (post-warmup)
  examples.
- **examples_per_sec / chars_per_sec / tokens_per_sec** — throughput on the
  timed examples. Token counts use the `o200k_base` tiktoken encoding, the
  same encoding used by the OPF model. If tiktoken is unavailable,
  `tokens_per_sec` is reported as `n/a`.
- **exact span+label F1** — exact match on `(label, start, end)` after the
  source dataset labels have been mapped to OPF span classes.
- **any-PII span F1** — exact match on `(start, end)` ignoring the label.

For `--backend both`, the harness also prints a comparison block with
speedup ratios (MLX/original) and F1 deltas (MLX − original).

## Label mapping

Source labels in `privacy_mask.label` are mapped to the OPF `v2` span
class space before matching. Unmapped source labels (`COMPANYNAME`,
`AMOUNT`, `GENDER`, `SEX`) are dropped from the gold set. See
`SOURCE_LABEL_TO_OPF` in `benchmark_hf_pii.py` for the full table.
