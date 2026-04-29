import Foundation
import MLX

/// High-level inference pipeline: text -> tokenizer -> model -> spans -> redacted text.
final class InferencePipeline {
    let model: OPFTransformer
    let cfg: OPFModelConfig
    let tokenizer: OPFTokenizer
    let labelInfo: LabelInfo
    let viterbiDecoder: ViterbiDecoder
    let context: Int
    let decodeMode: String

    init(
        model: OPFTransformer,
        cfg: OPFModelConfig,
        tokenizer: OPFTokenizer,
        labelInfo: LabelInfo,
        viterbiDecoder: ViterbiDecoder,
        context: Int,
        decodeMode: String
    ) {
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.labelInfo = labelInfo
        self.viterbiDecoder = viterbiDecoder
        self.context = context
        self.decodeMode = decodeMode
    }

    /// Run end-to-end inference for one text and return non-overlapping detected spans
    /// plus the redacted text built from those spans.
    func run(text: String, decodeModeOverride: String? = nil) -> (
        tokenization: TokenizationResult,
        spans: [DetectedSpan],
        redactedText: String
    ) {
        let mode = (decodeModeOverride ?? decodeMode).lowercased()
        let tk = tokenizer.tokenizeWithOffsets(text)
        if tk.tokenIds.isEmpty {
            return (tk, [], text)
        }
        let sourceText = tk.decodedMismatch ? tk.decodedText : text

        // Run model in fixed-size windows of `context` tokens.
        var labelsByIndex: [Int: Int] = [:]
        var start = 0
        while start < tk.tokenIds.count {
            let end = min(start + context, tk.tokenIds.count)
            let windowIds = Array(tk.tokenIds[start ..< end])
            let labels = decodeWindow(windowTokens: windowIds, mode: mode)
            for (i, label) in labels.enumerated() {
                labelsByIndex[start + i] = label
            }
            start = end
        }

        let tokenSpans = labelsToSpans(labelsByIndex, labelInfo: labelInfo)
        let charSpans = tokenSpansToCharSpans(
            tokenSpans,
            charStarts: tk.charStarts,
            charEnds: tk.charEnds
        )
        let trimmed = trimCharSpansWhitespace(charSpans, text: sourceText)

        // Build DetectedSpan objects with placeholder + text.
        let chars = Array(sourceText)
        var detected: [DetectedSpan] = []
        for (labelIdx, s, e) in trimmed {
            guard s >= 0, e > s, e <= chars.count else { continue }
            let labelName = (labelIdx >= 0 && labelIdx < labelInfo.spanClassNames.count)
                ? labelInfo.spanClassNames[labelIdx]
                : "label_\(labelIdx)"
            let spanText = String(chars[s ..< e])
            detected.append(DetectedSpan(
                label: labelName,
                start: s,
                end: e,
                text: spanText,
                placeholder: labelPlaceholder(labelName)
            ))
        }
        let nonOverlap = selectNonOverlappingSpans(detected)
        let redacted = buildRedactedText(sourceText, spans: nonOverlap)
        return (tk, nonOverlap, redacted)
    }

    /// Decode one [T] window of token ids into [T] label ids using either argmax or Viterbi.
    private func decodeWindow(windowTokens: [Int32], mode: String) -> [Int] {
        let tokens = windowTokens
        let inputArray = MLXArray(tokens, [1, tokens.count])
        let logits = model(inputArray).asType(.float32)

        if mode == "argmax" {
            let labelsArr = argMax(logits, axis: -1).asType(.int32)
            eval(labelsArr)
            let asInt32: [Int32] = labelsArr.asArray(Int32.self)
            return asInt32.map { Int($0) }
        }

        // Viterbi: extract logits to [T][numLabels], log-softmax, then decode.
        eval(logits)
        let shape = logits.shape  // [1, T, num_labels]
        let T = shape[1]
        let L = shape[2]
        let flat: [Float] = logits.asArray(Float.self)
        var rows: [[Float]] = []
        rows.reserveCapacity(T)
        for t in 0 ..< T {
            var row = [Float](repeating: 0, count: L)
            let base = t * L
            for j in 0 ..< L {
                row[j] = flat[base + j]
            }
            rows.append(row)
        }
        let lp = logSoftmax(rows)
        return viterbiDecoder.decode(lp)
    }

    /// Convenience overload.
    private func decodeWindow(windowTokens: [Int], mode: String) -> [Int] {
        let asInt32 = windowTokens.map { Int32($0) }
        return decodeWindow(windowTokens: asInt32, mode: mode)
    }
}

/// Load a full pipeline from a checkpoint directory.
func loadPipeline(
    checkpointDir: String,
    tokenizerPathOverride: String?,
    allowDownload: Bool,
    context: Int,
    moeChunkSize: Int,
    decodeMode: String
) throws -> InferencePipeline {
    let configURL = URL(fileURLWithPath: (checkpointDir as NSString).appendingPathComponent("config.json"))
    let safetensorsURL = URL(fileURLWithPath: (checkpointDir as NSString).appendingPathComponent("model.safetensors"))
    let fm = FileManager.default
    guard fm.fileExists(atPath: configURL.path) else {
        throw OPFError.missingTensor("config.json at \(configURL.path)")
    }
    guard fm.fileExists(atPath: safetensorsURL.path) else {
        throw OPFError.missingTensor("model.safetensors at \(safetensorsURL.path)")
    }
    let configData = try Data(contentsOf: configURL)
    guard let configObj = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        throw OPFError.shapeMismatch("config.json is not a JSON object")
    }
    var cfg = OPFModelConfig.from(json: configObj)
    cfg.moeChunkSize = moeChunkSize

    let tokenizerPath = try OPFTokenizer.resolveTokenizerPath(
        checkpointDir: checkpointDir,
        explicitPath: tokenizerPathOverride,
        allowDownload: allowDownload
    )
    let tok = try OPFTokenizer.load(tokenizerJSONPath: tokenizerPath)

    let weights = try MLX.loadArrays(url: safetensorsURL)
    let model = try OPFTransformer(cfg: cfg, weights: weights)

    let labelInfo = LabelInfo.v2()
    let biases = try loadViterbiBiases(checkpointDir: checkpointDir)
    let viterbi = ViterbiDecoder(labelInfo: labelInfo, biases: biases)

    return InferencePipeline(
        model: model,
        cfg: cfg,
        tokenizer: tok,
        labelInfo: labelInfo,
        viterbiDecoder: viterbi,
        context: context,
        decodeMode: decodeMode
    )
}
