import Foundation

/// `tokenize --text "..." [--checkpoint <path>] [--tokenizer <path>]`
/// Diagnostic helper: prints the o200k_base token ids for a text and the per-token
/// character offsets, so parity vs. Python `tiktoken` can be verified without
/// loading the full model.
func runTokenize(_ argv: [String]) {
    var text: String? = nil
    var checkpoint: String = "~/.opf/privacy_filter"
    var tokenizerPath: String? = nil
    var noDownload = false
    var i = 0
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--text":
            guard i + 1 < argv.count else { die("--text requires a value") }
            text = argv[i + 1]; i += 2
        case "--checkpoint":
            guard i + 1 < argv.count else { die("--checkpoint requires a value") }
            checkpoint = argv[i + 1]; i += 2
        case "--tokenizer":
            guard i + 1 < argv.count else { die("--tokenizer requires a value") }
            tokenizerPath = argv[i + 1]; i += 2
        case "--no-download":
            noDownload = true; i += 1
        case "-h", "--help":
            print(usage()); exit(0)
        default:
            die("unknown argument: \(a)")
        }
    }
    guard let text = text else { die("--text is required") }
    let cp = expandTilde(checkpoint)
    let path: String
    do {
        path = try OPFTokenizer.resolveTokenizerPath(
            checkpointDir: cp,
            explicitPath: tokenizerPath,
            allowDownload: !noDownload
        )
    } catch {
        die("\(error)")
    }
    let tok: OPFTokenizer
    do {
        tok = try OPFTokenizer.load(tokenizerJSONPath: path)
    } catch {
        die("\(error)")
    }
    let result = tok.tokenizeWithOffsets(text)
    var out: [String: Any] = [
        "token_count": result.tokenIds.count,
        "token_ids": result.tokenIds,
        "char_starts": result.charStarts,
        "char_ends": result.charEnds,
        "decoded_mismatch": result.decodedMismatch,
    ]
    if result.decodedMismatch {
        out["decoded_text"] = result.decodedText
    }
    if let data = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted]),
       let s = String(data: data, encoding: .utf8) {
        print(s)
    }
}
