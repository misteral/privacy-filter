import Foundation
import MLX

// MARK: - CLI

@inline(__always)
func die(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(code)
}

func expandTilde(_ path: String) -> String {
    (path as NSString).expandingTildeInPath
}

func usage() -> String {
    """
    usage:
      opf-mlx-daemon inspect [--checkpoint <path>]
      opf-mlx-daemon forward --tokens <id1,id2,...> [--checkpoint <path>]
                             [--context <N>] [--moe-chunk-size <N>]
      opf-mlx-daemon serve-tokens [--checkpoint <path>] [--socket <path>]
                                  [--context <N>] [--moe-chunk-size <N>]
      opf-mlx-daemon redact --text <text> [--checkpoint <path>] [--context <N>]
                            [--moe-chunk-size <N>] [--decode-mode argmax|viterbi]
                            [--tokenizer <path>] [--no-download]
      opf-mlx-daemon serve [--checkpoint <path>] [--socket <path>] [--context <N>]
                           [--moe-chunk-size <N>] [--decode-mode argmax|viterbi]
                           [--tokenizer <path>] [--no-download]

    Subcommands:
      inspect       Load checkpoint config + safetensors and print a summary.
      forward       Run a single forward pass on a small token list and print
                    argmax labels + logits shape.
      serve-tokens  Long-lived Unix-socket daemon that takes pre-tokenized
                    o200k_base ids and returns argmax token labels (legacy).
      redact        One-shot end-to-end redaction: text in, JSON with detected
                    spans + redacted_text out (uses native Swift tokenizer).
      serve         Long-lived Unix-socket text daemon: text in, detected spans
                    + redacted_text out (newline-delimited JSON protocol).

    Options:
      --checkpoint <path>     Path to checkpoint directory (default: ~/.opf/privacy_filter)
      --tokens <list>         Comma-separated integer token ids (forward only).
      --socket <path>         Unix socket path. Default /tmp/opf-mlx-swift.sock for
                              serve-tokens; /tmp/opf-mlx-swift-text.sock for serve.
      --text <text>           Input text for redact.
      --context <N>           Maximum context length per request (default: 64
                              for forward, 128 elsewhere).
      --moe-chunk-size <N>    MoE chunk size (default: 1, or 2 for text commands).
      --decode-mode <mode>    'argmax' or 'viterbi' (text commands; default viterbi).
      --tokenizer <path>      Path to tokenizer.json (or directory containing it).
                              Default: <checkpoint>/tokenizer.json (downloaded if
                              missing unless --no-download is passed).
      --no-download           Do not auto-download tokenizer.json.
      -h, --help              Show this help.
    """
}

// MARK: - Argument parsing

struct InspectArgs {
    var checkpoint: String = "~/.opf/privacy_filter"
}

func parseInspectArgs(_ argv: [String]) -> InspectArgs {
    var args = InspectArgs()
    var i = 0
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--checkpoint":
            guard i + 1 < argv.count else { die("--checkpoint requires a value") }
            args.checkpoint = argv[i + 1]
            i += 2
        case "-h", "--help":
            print(usage())
            exit(0)
        default:
            die("unknown argument: \(a)\n\n\(usage())")
        }
    }
    return args
}

// MARK: - Config decoding

struct OPFConfig: Decodable {
    let model_type: String?
    let encoding: String?
    let num_hidden_layers: Int?
    let hidden_size: Int?
    let num_experts: Int?
    let experts_per_token: Int?
    let num_labels: Int?
    let vocab_size: Int?
    let bidirectional_left_context: Int?
    let bidirectional_right_context: Int?
}

// MARK: - DType helpers

func dtypeName(_ dtype: DType) -> String {
    switch dtype {
    case .bool: return "bool"
    case .uint8: return "uint8"
    case .uint16: return "uint16"
    case .uint32: return "uint32"
    case .uint64: return "uint64"
    case .int8: return "int8"
    case .int16: return "int16"
    case .int32: return "int32"
    case .int64: return "int64"
    case .float16: return "float16"
    case .float32: return "float32"
    case .bfloat16: return "bfloat16"
    case .complex64: return "complex64"
    case .float64: return "float64"
    }
}

// MARK: - Inspect command

func runInspect(_ argv: [String]) {
    let args = parseInspectArgs(argv)
    let checkpoint = expandTilde(args.checkpoint)
    let checkpointURL = URL(fileURLWithPath: checkpoint, isDirectory: true)

    let configURL = checkpointURL.appendingPathComponent("config.json")
    let safetensorsURL = checkpointURL.appendingPathComponent("model.safetensors")

    let fm = FileManager.default
    guard fm.fileExists(atPath: configURL.path) else {
        die("config.json not found at \(configURL.path)")
    }
    guard fm.fileExists(atPath: safetensorsURL.path) else {
        die("model.safetensors not found at \(safetensorsURL.path)")
    }

    // --- Load config.json
    let config: OPFConfig
    do {
        let data = try Data(contentsOf: configURL)
        config = try JSONDecoder().decode(OPFConfig.self, from: data)
    } catch {
        die("failed to parse config.json: \(error)")
    }

    // --- Load model.safetensors
    let arrays: [String: MLXArray]
    do {
        arrays = try MLX.loadArrays(url: safetensorsURL)
    } catch {
        die("failed to load model.safetensors: \(error)")
    }

    // --- Report
    print("OPF MLX Swift checkpoint inspect")
    print("================================")
    print("checkpoint: \(checkpointURL.path)")
    print("")
    print("config:")
    func line(_ key: String, _ value: Any?) {
        let v = value.map { "\($0)" } ?? "<missing>"
        print("  \(key): \(v)")
    }
    line("model_type", config.model_type)
    line("encoding", config.encoding)
    line("num_hidden_layers", config.num_hidden_layers)
    line("hidden_size", config.hidden_size)
    line("num_experts", config.num_experts)
    line("experts_per_token", config.experts_per_token)
    line("num_labels", config.num_labels)
    line("vocab_size", config.vocab_size)
    line("bidirectional_left_context", config.bidirectional_left_context)
    line("bidirectional_right_context", config.bidirectional_right_context)

    let names = arrays.keys.sorted()
    var totalParams: Int = 0
    for name in names {
        let arr = arrays[name]!
        totalParams += arr.size
    }

    print("")
    print("tensors: \(names.count)")
    print("total parameters: \(totalParams)")
    print("")

    let preview = names.prefix(20)
    print("first \(preview.count) tensors:")
    for name in preview {
        let arr = arrays[name]!
        let shapeStr = "[" + arr.shape.map(String.init).joined(separator: ", ") + "]"
        print("  \(name)  shape=\(shapeStr)  dtype=\(dtypeName(arr.dtype))")
    }

    let required = [
        "embedding.weight",
        "block.0.attn.qkv.weight",
        "block.0.mlp.swiglu.weight",
        "norm.scale",
        "unembedding.weight",
    ]
    print("")
    print("presence checks:")
    var missing: [String] = []
    for key in required {
        let present = arrays[key] != nil
        let mark = present ? "OK" : "MISSING"
        print("  [\(mark)] \(key)")
        if !present { missing.append(key) }
    }

    if !missing.isEmpty {
        print("")
        FileHandle.standardError.write(Data("warning: missing expected tensors: \(missing.joined(separator: ", "))\n".utf8))
    }
}

// MARK: - Entry point

let allArgs = CommandLine.arguments
if allArgs.count < 2 {
    print(usage())
    exit(2)
}

let command = allArgs[1]
let rest = Array(allArgs.dropFirst(2))

switch command {
case "inspect":
    runInspect(rest)
case "forward":
    runForward(rest)
case "serve-tokens":
    runServeTokens(rest)
case "redact":
    runRedact(rest)
case "serve":
    runServe(rest)
case "tokenize":
    runTokenize(rest)
case "-h", "--help":
    print(usage())
default:
    die("unknown command: \(command)\n\n\(usage())", code: 2)
}
