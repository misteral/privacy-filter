import Foundation

struct RedactArgs {
    var checkpoint: String = "~/.opf/privacy_filter"
    var text: String? = nil
    var context: Int = 128
    var moeChunkSize: Int = 2
    var decodeMode: String = "viterbi"
    var tokenizerPath: String? = nil
    var noDownload: Bool = false
}

func parseRedactArgs(_ argv: [String]) -> RedactArgs {
    var args = RedactArgs()
    var i = 0
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--checkpoint":
            guard i + 1 < argv.count else { die("--checkpoint requires a value") }
            args.checkpoint = argv[i + 1]; i += 2
        case "--text":
            guard i + 1 < argv.count else { die("--text requires a value") }
            args.text = argv[i + 1]; i += 2
        case "--context":
            guard i + 1 < argv.count else { die("--context requires a value") }
            guard let n = Int(argv[i + 1]), n > 0 else { die("invalid --context value: \(argv[i + 1])") }
            args.context = n; i += 2
        case "--moe-chunk-size":
            guard i + 1 < argv.count else { die("--moe-chunk-size requires a value") }
            guard let n = Int(argv[i + 1]), n > 0 else { die("invalid --moe-chunk-size value: \(argv[i + 1])") }
            args.moeChunkSize = n; i += 2
        case "--decode-mode":
            guard i + 1 < argv.count else { die("--decode-mode requires a value") }
            let v = argv[i + 1].lowercased()
            guard v == "viterbi" || v == "argmax" else { die("invalid --decode-mode: \(v)") }
            args.decodeMode = v; i += 2
        case "--tokenizer":
            guard i + 1 < argv.count else { die("--tokenizer requires a value") }
            args.tokenizerPath = argv[i + 1]; i += 2
        case "--no-download":
            args.noDownload = true; i += 1
        case "-h", "--help":
            print(usage()); exit(0)
        default:
            die("unknown argument: \(a)\n\n\(usage())")
        }
    }
    return args
}

func runRedact(_ argv: [String]) {
    let args = parseRedactArgs(argv)
    guard let text = args.text else { die("--text is required") }
    let checkpoint = expandTilde(args.checkpoint)

    let pipeline: InferencePipeline
    do {
        pipeline = try loadPipeline(
            checkpointDir: checkpoint,
            tokenizerPathOverride: args.tokenizerPath,
            allowDownload: !args.noDownload,
            context: args.context,
            moeChunkSize: args.moeChunkSize,
            decodeMode: args.decodeMode
        )
    } catch let e as OPFError {
        die("pipeline load failed: \(e.description)")
    } catch let e as TokenizerError {
        die("pipeline load failed: \(e.description)")
    } catch {
        die("pipeline load failed: \(error)")
    }

    let start = Date()
    let result = pipeline.run(text: text)
    let elapsedMs = Date().timeIntervalSince(start) * 1000.0

    var spansArr: [[String: Any]] = []
    for s in result.spans {
        spansArr.append([
            "label": s.label,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "placeholder": s.placeholder,
        ])
    }
    let payload: [String: Any] = [
        "text": result.tokenization.decodedMismatch ? result.tokenization.decodedText : text,
        "detected_spans": spansArr,
        "redacted_text": result.redactedText,
        "latency_ms": elapsedMs,
        "token_count": result.tokenization.tokenIds.count,
        "decode_mode": args.decodeMode,
    ]
    do {
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        if let s = String(data: data, encoding: .utf8) {
            print(s)
        } else {
            print("{\"error\":\"failed to encode payload\"}")
        }
    } catch {
        print("{\"error\":\"\(error)\"}")
    }
}
