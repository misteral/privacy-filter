import Foundation
import MLX

struct ForwardArgs {
    var checkpoint: String = "~/.opf/privacy_filter"
    var tokens: [Int32] = []
    var context: Int = 64
    var moeChunkSize: Int = 1
}

func parseForwardArgs(_ argv: [String]) -> ForwardArgs {
    var args = ForwardArgs()
    var i = 0
    var sawTokens = false
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--checkpoint":
            guard i + 1 < argv.count else { die("--checkpoint requires a value") }
            args.checkpoint = argv[i + 1]
            i += 2
        case "--tokens":
            guard i + 1 < argv.count else { die("--tokens requires a comma-separated value") }
            let raw = argv[i + 1]
            let parts = raw.split(separator: ",", omittingEmptySubsequences: true)
            var ids: [Int32] = []
            for p in parts {
                let trimmed = p.trimmingCharacters(in: .whitespaces)
                guard let n = Int32(trimmed) else { die("invalid token id: \(trimmed)") }
                ids.append(n)
            }
            if ids.isEmpty { die("--tokens must contain at least one token id") }
            args.tokens = ids
            sawTokens = true
            i += 2
        case "--context":
            guard i + 1 < argv.count else { die("--context requires a value") }
            guard let n = Int(argv[i + 1]), n > 0 else { die("invalid --context value: \(argv[i + 1])") }
            args.context = n
            i += 2
        case "--moe-chunk-size":
            guard i + 1 < argv.count else { die("--moe-chunk-size requires a value") }
            guard let n = Int(argv[i + 1]), n > 0 else { die("invalid --moe-chunk-size value: \(argv[i + 1])") }
            args.moeChunkSize = n
            i += 2
        case "-h", "--help":
            print(usage())
            exit(0)
        default:
            die("unknown argument: \(a)\n\n\(usage())")
        }
    }
    if !sawTokens { die("--tokens is required") }
    return args
}

func runForward(_ argv: [String]) {
    let args = parseForwardArgs(argv)
    let checkpoint = expandTilde(args.checkpoint)
    let checkpointURL = URL(fileURLWithPath: checkpoint, isDirectory: true)

    let configURL = checkpointURL.appendingPathComponent("config.json")
    let safetensorsURL = checkpointURL.appendingPathComponent("model.safetensors")

    let fm = FileManager.default
    guard fm.fileExists(atPath: configURL.path) else { die("config.json not found at \(configURL.path)") }
    guard fm.fileExists(atPath: safetensorsURL.path) else { die("model.safetensors not found at \(safetensorsURL.path)") }

    // Load config
    let cfg: OPFModelConfig
    do {
        let data = try Data(contentsOf: configURL)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            die("config.json is not a JSON object")
        }
        var parsed = OPFModelConfig.from(json: obj)
        parsed.moeChunkSize = args.moeChunkSize
        cfg = parsed
    } catch {
        die("failed to parse config.json: \(error)")
    }

    if args.tokens.count > args.context {
        die("token list of \(args.tokens.count) exceeds --context \(args.context)")
    }
    for t in args.tokens {
        if t < 0 || t >= Int32(cfg.vocabSize) {
            die("token id \(t) out of range [0, \(cfg.vocabSize))")
        }
    }

    // Load weights
    let weights: [String: MLXArray]
    do {
        weights = try MLX.loadArrays(url: safetensorsURL)
    } catch {
        die("failed to load model.safetensors: \(error)")
    }

    // Build model
    let model: OPFTransformer
    do {
        model = try OPFTransformer(cfg: cfg, weights: weights)
    } catch let e as OPFError {
        die("model build failed: \(e.description)")
    } catch {
        die("model build failed: \(error)")
    }

    // Prepare input batch [1, T]
    let tokenArray = MLXArray(args.tokens, [1, args.tokens.count])

    print("input tokens: \(args.tokens.count)")
    let start = Date()
    let logits = model(tokenArray).asType(.float32)
    eval(logits)
    let elapsedMs = Date().timeIntervalSince(start) * 1000.0

    let shape = logits.shape
    let shapeStr = "[" + shape.map(String.init).joined(separator: ", ") + "]"
    print("logits shape: \(shapeStr)")

    if shape.count != 3 || shape[0] != 1 || shape[1] != args.tokens.count || shape[2] != cfg.numLabels {
        die("unexpected logits shape \(shapeStr); expected [1, \(args.tokens.count), \(cfg.numLabels)]")
    }

    // Argmax along the label axis -> shape [1, T]
    let labels = argMax(logits, axis: -1).asType(.int32)
    eval(labels)
    let labelArr: [Int32] = labels.asArray(Int32.self)
    let labelStr = labelArr.map { String($0) }.joined(separator: ",")
    print("argmax labels: \(labelStr)")
    print(String(format: "latency_ms: %.2f", elapsedMs))
}
