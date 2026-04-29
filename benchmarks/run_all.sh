#!/usr/bin/env bash
# Run the OPF benchmark for the original (PyTorch CPU), MLX (Python), and
# swift-mlx-text-daemon backends, collect peak memory for each (via
# /usr/bin/time -l), and print a unified markdown table.
#
# macOS only — relies on /usr/bin/time -l, xcodebuild, and Apple Silicon MLX.

set -euo pipefail

LIMIT=100

print_usage() {
    cat <<'EOF'
Usage: run_all.sh [--limit N]

Options:
  --limit N    Number of examples to load from the test split (default: 100).
  -h, --help   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            [[ $# -ge 2 ]] || { echo "--limit requires a value" >&2; exit 2; }
            LIMIT="$2"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; print_usage >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
SWIFT_DIR="$REPO_ROOT/swift-mlx"
SOCKET="/tmp/opf-mlx-swift-text.sock"
DAEMON_BIN="$SWIFT_DIR/.build/xcode-release/Build/Products/Release/opf-mlx-daemon"

ORIG_OUT="$BENCH_DIR/out_original.json"
MLX_OUT="$BENCH_DIR/out_mlx.json"
SWIFT_OUT="$BENCH_DIR/out_swift.json"

ORIG_TIME="/tmp/opf-bench-original.time"
MLX_TIME="/tmp/opf-bench-mlx.time"
DAEMON_TIME="/tmp/opf-bench-swift-daemon.time"
DAEMON_LOG="/tmp/opf-bench-swift-daemon.log"
XCODE_LOG="/tmp/opf-xcodebuild.log"

TIME_PID=""
DAEMON_PID=""

cleanup() {
    if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
    fi
    if [[ -n "$TIME_PID" ]] && kill -0 "$TIME_PID" 2>/dev/null; then
        # Try to grab the daemon child if we don't already have its pid
        if [[ -z "$DAEMON_PID" ]]; then
            local child
            child="$(pgrep -P "$TIME_PID" || true)"
            [[ -n "$child" ]] && kill "$child" 2>/dev/null || true
        fi
        wait "$TIME_PID" 2>/dev/null || true
    fi
    rm -f "$SOCKET"
}
trap cleanup EXIT INT TERM

extract_maxrss_mb() {
    # /usr/bin/time -l writes "<bytes>  maximum resident set size" on macOS.
    local f="$1"
    if [[ ! -f "$f" ]]; then echo "n/a"; return; fi
    local bytes
    bytes="$(awk '/maximum resident set size/ { print $1; exit }' "$f")"
    if [[ -z "$bytes" ]]; then echo "n/a"; return; fi
    awk -v b="$bytes" 'BEGIN { printf "%.1f", b / (1024 * 1024) }'
}

rm -f "$SOCKET"

echo "==> Building Swift daemon (Release)..."
(
    cd "$SWIFT_DIR"
    xcodebuild build \
        -configuration Release \
        -scheme OPFMLXDaemon \
        -destination 'platform=macOS' \
        -derivedDataPath .build/xcode-release \
        > "$XCODE_LOG" 2>&1
) || {
    echo "Swift build failed; tail of $XCODE_LOG:" >&2
    tail -50 "$XCODE_LOG" >&2 || true
    exit 1
}
if [[ ! -x "$DAEMON_BIN" ]]; then
    echo "Daemon binary missing at $DAEMON_BIN" >&2
    exit 1
fi

echo
echo "==> Backend: original (limit=$LIMIT)"
/usr/bin/time -l \
    uv run --no-sync --python 3.12 --with datasets \
        "$BENCH_DIR/benchmark_hf_pii.py" \
        --backend original --limit "$LIMIT" \
        --output-json "$ORIG_OUT" \
    2> "$ORIG_TIME"
ORIG_MEM_MB="$(extract_maxrss_mb "$ORIG_TIME")"
echo "original peak rss: ${ORIG_MEM_MB} MB"

echo
echo "==> Backend: mlx (limit=$LIMIT)"
/usr/bin/time -l \
    uv run --no-sync --python 3.12 --with datasets --with mlx \
        "$BENCH_DIR/benchmark_hf_pii.py" \
        --backend mlx --limit "$LIMIT" \
        --output-json "$MLX_OUT" \
    2> "$MLX_TIME"
MLX_MEM_MB="$(extract_maxrss_mb "$MLX_TIME")"
echo "mlx peak rss: ${MLX_MEM_MB} MB"

echo
echo "==> Backend: swift-mlx-text-daemon (limit=$LIMIT)"
: > "$DAEMON_LOG"
: > "$DAEMON_TIME"

# Run the daemon under /usr/bin/time -l; redirect stdout to a log we can poll
# for "ready", and stderr (where time -l writes) to its own file.
/usr/bin/time -l \
    "$DAEMON_BIN" serve \
    --checkpoint "$HOME/.opf/privacy_filter" \
    --socket "$SOCKET" \
    --context 128 --moe-chunk-size 2 --decode-mode viterbi \
    > "$DAEMON_LOG" 2> "$DAEMON_TIME" &
TIME_PID=$!

echo "Waiting for daemon (time pid=$TIME_PID) to be ready..."
READY=0
for _ in $(seq 1 240); do
    if grep -q "^ready$" "$DAEMON_LOG" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 "$TIME_PID" 2>/dev/null; then
        echo "Daemon exited before becoming ready. Daemon log:" >&2
        cat "$DAEMON_LOG" >&2
        echo "--- /usr/bin/time output:" >&2
        cat "$DAEMON_TIME" >&2 || true
        exit 1
    fi
    sleep 1
done
if [[ "$READY" != "1" ]]; then
    echo "Daemon did not become ready within timeout." >&2
    exit 1
fi

DAEMON_PID="$(pgrep -P "$TIME_PID" || true)"
echo "Daemon ready (daemon pid=${DAEMON_PID:-unknown})."

uv run --no-sync --python 3.12 --with datasets --with mlx \
    "$BENCH_DIR/benchmark_hf_pii.py" \
    --backend swift-mlx-text-daemon --decode-mode viterbi \
    --limit "$LIMIT" \
    --swift-text-socket "$SOCKET" \
    --output-json "$SWIFT_OUT"

# Stop the daemon (kill the actual binary, not /usr/bin/time, so time can
# wait4() on it and write the rusage block).
if [[ -n "$DAEMON_PID" ]] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    kill "$DAEMON_PID" 2>/dev/null || true
fi
wait "$TIME_PID" 2>/dev/null || true
TIME_PID=""
DAEMON_PID=""

SWIFT_MEM_MB="$(extract_maxrss_mb "$DAEMON_TIME")"
echo "swift-mlx-text-daemon peak rss: ${SWIFT_MEM_MB} MB"

echo
echo
python3 - \
    "$ORIG_OUT" "$MLX_OUT" "$SWIFT_OUT" \
    "$ORIG_MEM_MB" "$MLX_MEM_MB" "$SWIFT_MEM_MB" \
    "$LIMIT" <<'PY'
import json, sys

orig_path, mlx_path, swift_path, o_mem, m_mem, s_mem, limit = sys.argv[1:8]


def load(path: str, name: str) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    return data.get("backends", {}).get(name, {})


def fmt(value, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


orig = load(orig_path, "original")
mlx = load(mlx_path, "mlx")
swift = load(swift_path, "swift-mlx-text-daemon")

print(f"# Benchmark Comparison (limit={limit})\n")
print("| Backend | Examples/sec | Exact F1 | Peak Memory (MB) |")
print("|---|---|---|---|")
rows = [
    ("original", orig, o_mem),
    ("mlx", mlx, m_mem),
    ("swift-mlx-text-daemon", swift, s_mem),
]
for name, data, mem in rows:
    print(
        f"| {name} | {fmt(data.get('examples_per_sec'))} | "
        f"{fmt(data.get('exact.f1'), '.4f')} | {mem} |"
    )
PY
