import Foundation
import Hub
import Tokenizers

/// Result of tokenizing a UTF-8 text into o200k_base ids with per-token char ranges.
struct TokenizationResult {
    let tokenIds: [Int]
    /// Character (Swift `Character`) start offsets, one per token.
    let charStarts: [Int]
    /// Character end offsets (exclusive), one per token.
    let charEnds: [Int]
    /// Decoded text used for the offset table; equals input if decode is lossless.
    let decodedText: String
    /// Whether the decoded text differs from the original input.
    let decodedMismatch: Bool
}

enum TokenizerError: Error, CustomStringConvertible {
    case missingTokenizerJSON(String)
    case loadFailed(String)
    case downloadFailed(String)

    var description: String {
        switch self {
        case .missingTokenizerJSON(let path): return "tokenizer.json not found at \(path)"
        case .loadFailed(let msg): return "failed to load tokenizer: \(msg)"
        case .downloadFailed(let msg): return "failed to download tokenizer.json: \(msg)"
        }
    }
}

/// Wraps a `swift-transformers` `Tokenizer` and exposes byte-level offset tracking
/// matching the Python `tiktoken.decode_single_token_bytes` algorithm.
final class OPFTokenizer {
    let tokenizer: Tokenizer
    /// Cached UTF-8 byte sequences for token ids we've decoded so far.
    /// Indexed by token id; nil entry means not yet computed.
    private var tokenBytesCache: [Int: [UInt8]] = [:]
    private let cacheLock = NSLock()

    private init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Resolve a `tokenizer.json` path from a checkpoint directory and an optional override.
    /// If neither contains tokenizer.json and `allowDownload` is true, downloads from
    /// the OPF HF repo into the checkpoint directory.
    static func resolveTokenizerPath(
        checkpointDir: String,
        explicitPath: String?,
        allowDownload: Bool
    ) throws -> String {
        let fm = FileManager.default
        if let p = explicitPath {
            let expanded = expandTilde(p)
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: expanded, isDirectory: &isDir) {
                if isDir.boolValue {
                    let candidate = (expanded as NSString).appendingPathComponent("tokenizer.json")
                    if fm.fileExists(atPath: candidate) { return candidate }
                    throw TokenizerError.missingTokenizerJSON(candidate)
                }
                return expanded
            }
            throw TokenizerError.missingTokenizerJSON(expanded)
        }

        let inCheckpoint = (checkpointDir as NSString).appendingPathComponent("tokenizer.json")
        if fm.fileExists(atPath: inCheckpoint) { return inCheckpoint }

        if !allowDownload {
            throw TokenizerError.missingTokenizerJSON(inCheckpoint)
        }

        // Download to checkpoint directory.
        let url = URL(string: "https://huggingface.co/openai/privacy-filter/resolve/main/tokenizer.json")!
        FileHandle.standardError.write(Data("downloading tokenizer.json from \(url.absoluteString)...\n".utf8))
        let semaphore = DispatchSemaphore(value: 0)
        var downloadError: Error?
        let task = URLSession.shared.downloadTask(with: url) { tmpURL, response, error in
            defer { semaphore.signal() }
            if let error = error {
                downloadError = error
                return
            }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                downloadError = TokenizerError.downloadFailed("status=\((response as? HTTPURLResponse)?.statusCode ?? -1)")
                return
            }
            guard let tmpURL = tmpURL else {
                downloadError = TokenizerError.downloadFailed("no temp file")
                return
            }
            do {
                if fm.fileExists(atPath: inCheckpoint) {
                    try fm.removeItem(atPath: inCheckpoint)
                }
                try fm.createDirectory(
                    atPath: checkpointDir,
                    withIntermediateDirectories: true
                )
                try fm.moveItem(atPath: tmpURL.path, toPath: inCheckpoint)
            } catch {
                downloadError = error
            }
        }
        task.resume()
        semaphore.wait()
        if let err = downloadError {
            throw TokenizerError.downloadFailed("\(err)")
        }
        return inCheckpoint
    }

    /// Build a Tokenizer from a tokenizer.json path. Bypasses `from(modelFolder:)` so
    /// that a `tokenizer_config.json` is not required — we provide a minimal one.
    static func load(tokenizerJSONPath: String) throws -> OPFTokenizer {
        let url = URL(fileURLWithPath: tokenizerJSONPath)
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw TokenizerError.loadFailed("could not read \(tokenizerJSONPath): \(error)")
        }
        let tokenizerData: Config
        do {
            tokenizerData = try JSONDecoder().decode(Config.self, from: data)
        } catch {
            throw TokenizerError.loadFailed("could not parse tokenizer.json: \(error)")
        }
        let tokenizerConfig = Config([
            "tokenizer_class" as NSString: "PreTrainedTokenizer" as Any,
        ])
        let built: Tokenizer
        do {
            built = try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData,
                strict: false
            )
        } catch {
            throw TokenizerError.loadFailed("AutoTokenizer.from threw: \(error)")
        }
        return OPFTokenizer(tokenizer: built)
    }

    /// Encode text, returning ids without special tokens.
    func encode(_ text: String) -> [Int] {
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// Return UTF-8 bytes for a single token id, using cached results.
    func tokenBytes(_ tokenId: Int) -> [UInt8] {
        cacheLock.lock()
        if let cached = tokenBytesCache[tokenId] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()
        let decoded = tokenizer.decode(tokens: [tokenId], skipSpecialTokens: false)
        let bytes = Array(decoded.utf8)
        cacheLock.lock()
        tokenBytesCache[tokenId] = bytes
        cacheLock.unlock()
        return bytes
    }

    /// Decode tokens to text.
    func decodeAll(_ tokens: [Int]) -> String {
        return tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
    }

    /// Tokenize and compute character offsets aligned to the input text.
    /// Mirrors `opf._core.spans.decode_text_with_offsets`: build per-character
    /// byte cursors over the decoded text, then bisect each token's byte range.
    func tokenizeWithOffsets(_ text: String) -> TokenizationResult {
        let ids = encode(text)
        if ids.isEmpty {
            return TokenizationResult(
                tokenIds: [],
                charStarts: [],
                charEnds: [],
                decodedText: "",
                decodedMismatch: !text.isEmpty
            )
        }

        var tokenByteRanges: [(start: Int, end: Int)] = []
        tokenByteRanges.reserveCapacity(ids.count)
        var concatenated: [UInt8] = []
        for id in ids {
            let bytes = tokenBytes(id)
            let start = concatenated.count
            concatenated.append(contentsOf: bytes)
            tokenByteRanges.append((start: start, end: start + bytes.count))
        }
        let decodedText = String(decoding: concatenated, as: UTF8.self)

        var charByteStarts: [Int] = []
        var charByteEnds: [Int] = []
        charByteStarts.reserveCapacity(decodedText.count)
        charByteEnds.reserveCapacity(decodedText.count)
        var byteCursor = 0
        for ch in decodedText {
            charByteStarts.append(byteCursor)
            byteCursor += ch.utf8.count
            charByteEnds.append(byteCursor)
        }

        var charStarts: [Int] = []
        var charEnds: [Int] = []
        charStarts.reserveCapacity(ids.count)
        charEnds.reserveCapacity(ids.count)
        for range in tokenByteRanges {
            let startIdx = bisectRight(charByteEnds, range.start)
            var endIdx = bisectLeft(charByteStarts, range.end)
            if endIdx < startIdx { endIdx = startIdx }
            charStarts.append(startIdx)
            charEnds.append(endIdx)
        }

        return TokenizationResult(
            tokenIds: ids,
            charStarts: charStarts,
            charEnds: charEnds,
            decodedText: decodedText,
            decodedMismatch: decodedText != text
        )
    }
}

// MARK: - Bisect helpers

@inline(__always)
private func bisectRight(_ array: [Int], _ value: Int) -> Int {
    // Return the rightmost insertion point for value in sorted array.
    var lo = 0
    var hi = array.count
    while lo < hi {
        let mid = (lo + hi) >> 1
        if value < array[mid] {
            hi = mid
        } else {
            lo = mid + 1
        }
    }
    return lo
}

@inline(__always)
private func bisectLeft(_ array: [Int], _ value: Int) -> Int {
    // Return the leftmost insertion point for value in sorted array.
    var lo = 0
    var hi = array.count
    while lo < hi {
        let mid = (lo + hi) >> 1
        if array[mid] < value {
            lo = mid + 1
        } else {
            hi = mid
        }
    }
    return lo
}
