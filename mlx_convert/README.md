# OpenAI Privacy Filter — MLX backend

Experimental Apple Silicon inference path for OpenAI Privacy Filter. It loads the original OPF checkpoint directly with MLX and reuses the repo's tokenizer, Viterbi decoder, and span rendering code.

## Install

```bash
uv venv .venv
uv pip install -e '.[mlx]'
```

`mlx` requires Apple Silicon macOS.

## Checkpoint

The MLX backend expects the same checkpoint directory as the normal OPF runtime:

```text
~/.opf/privacy_filter/
  config.json
  model.safetensors
  viterbi_calibration.json
```

The first run can reuse OPF's downloader and will download `openai/privacy-filter`'s `original/*` files (~2.8 GB):

```bash
python -m opf --device cpu "Alice was born on 1990-01-02."
```

Or pass an explicit directory:

```bash
./mlx_convert/opf-mlx --checkpoint /path/to/privacy_filter "Alice was born on 1990-01-02."
```

## Usage

```bash
# One-shot
./mlx_convert/opf-mlx "Alice was born on 1990-01-02."

# JSON output
./mlx_convert/opf-mlx --json "Alice was born on 1990-01-02."

# Read stdin
cat notes.txt | ./mlx_convert/opf-mlx --text-only
```

Python:

```python
import sys
sys.path.insert(0, "mlx_convert")
from opf_mlx import OPFMLX

opf = OPFMLX(context_window_length=4096, moe_chunk_size=8)
print(opf.redact("Alice was born on 1990-01-02.").redacted_text)
```

## Notes and limitations

- No conversion step is required: the original `.safetensors` file is loaded directly.
- Default context is `4096` tokens, not the model's full `128k`, because the simple MLX local-attention implementation materializes a `[tokens × 257]` band.
- MoE is chunked (`--moe-chunk-size`) to control memory. Lower values use less memory but are slower.
- This is an experimental backend; validate against the PyTorch runtime before production use.
