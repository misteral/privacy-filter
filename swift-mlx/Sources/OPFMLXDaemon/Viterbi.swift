import Foundation

/// Calibration biases for the OPF Viterbi CRF decoder.
struct ViterbiBiases {
    var backgroundStay: Float = 0.0
    var backgroundToStart: Float = 0.0
    var insideToContinue: Float = 0.0
    var insideToEnd: Float = 0.0
    var endToBackground: Float = 0.0
    var endToStart: Float = 0.0

    static let zero = ViterbiBiases()
}

enum ViterbiCalibrationError: Error, CustomStringConvertible {
    case parseFailed(String)
    case missingFields(String)

    var description: String {
        switch self {
        case .parseFailed(let m): return "calibration parse failed: \(m)"
        case .missingFields(let m): return "calibration missing fields: \(m)"
        }
    }
}

/// Load Viterbi calibration biases from `viterbi_calibration.json` in a checkpoint
/// directory. Returns zero biases when the file is missing.
func loadViterbiBiases(checkpointDir: String) throws -> ViterbiBiases {
    let path = (checkpointDir as NSString).appendingPathComponent("viterbi_calibration.json")
    let fm = FileManager.default
    if !fm.fileExists(atPath: path) {
        return ViterbiBiases.zero
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw ViterbiCalibrationError.parseFailed("not a JSON object")
    }
    guard let ops = obj["operating_points"] as? [String: Any],
          let def = ops["default"] as? [String: Any],
          let biases = def["biases"] as? [String: Any] else {
        throw ViterbiCalibrationError.missingFields("expected operating_points.default.biases")
    }
    func num(_ key: String) -> Float {
        if let v = biases[key] as? Double { return Float(v) }
        if let v = biases[key] as? Int { return Float(v) }
        if let v = biases[key] as? NSNumber { return v.floatValue }
        return 0.0
    }
    return ViterbiBiases(
        backgroundStay: num("transition_bias_background_stay"),
        backgroundToStart: num("transition_bias_background_to_start"),
        insideToContinue: num("transition_bias_inside_to_continue"),
        insideToEnd: num("transition_bias_inside_to_end"),
        endToBackground: num("transition_bias_end_to_background"),
        endToStart: num("transition_bias_end_to_start")
    )
}

/// Viterbi CRF decoder with BIESO transition constraints.
/// Mirrors `opf._core.decoding.ViterbiCRFDecoder` for CPU operation.
final class ViterbiDecoder {
    private static let NEG_INF: Float = -1.0e9

    private let labelInfo: LabelInfo
    private let numClasses: Int
    private let startScores: [Float]
    private let endScores: [Float]
    /// Flat row-major `[numClasses * numClasses]`, indexed `[from * N + to]`.
    private let transitionScores: [Float]

    init(labelInfo: LabelInfo, biases: ViterbiBiases) {
        self.labelInfo = labelInfo
        self.numClasses = labelInfo.nerClassNames.count
        let N = self.numClasses
        var start = [Float](repeating: ViterbiDecoder.NEG_INF, count: N)
        var end = [Float](repeating: ViterbiDecoder.NEG_INF, count: N)
        var trans = [Float](repeating: ViterbiDecoder.NEG_INF, count: N * N)

        let backgroundTokenIdx = labelInfo.backgroundTokenLabel
        let backgroundSpanIdx = labelInfo.backgroundSpanLabel

        for idx in 0 ..< N {
            let tag = labelInfo.tokenBoundaryTags[idx] ?? nil
            if tag == "B" || tag == "S" || idx == backgroundTokenIdx {
                start[idx] = 0.0
            }
            if tag == "E" || tag == "S" || idx == backgroundTokenIdx {
                end[idx] = 0.0
            }
            let prevSpan = labelInfo.tokenToSpanLabel[idx]
            for next in 0 ..< N {
                let nextTag = labelInfo.tokenBoundaryTags[next] ?? nil
                let nextSpan = labelInfo.tokenToSpanLabel[next]
                if ViterbiDecoder.isValidTransition(
                    prevTag: tag, prevSpan: prevSpan,
                    nextTag: nextTag, nextSpan: nextSpan,
                    backgroundTokenIdx: backgroundTokenIdx,
                    backgroundSpanIdx: backgroundSpanIdx,
                    nextIdx: next
                ) {
                    trans[idx * N + next] = ViterbiDecoder.transitionBias(
                        biases: biases,
                        prevTag: tag, prevSpan: prevSpan,
                        nextTag: nextTag, nextSpan: nextSpan,
                        backgroundTokenIdx: backgroundTokenIdx,
                        backgroundSpanIdx: backgroundSpanIdx,
                        prevIdx: idx, nextIdx: next
                    )
                }
            }
        }

        self.startScores = start
        self.endScores = end
        self.transitionScores = trans
    }

    private static func isValidTransition(
        prevTag: String?, prevSpan: Int?,
        nextTag: String?, nextSpan: Int?,
        backgroundTokenIdx: Int, backgroundSpanIdx: Int,
        nextIdx: Int
    ) -> Bool {
        let nextIsBackground = (nextSpan == backgroundSpanIdx) || (nextIdx == backgroundTokenIdx)
        if (nextSpan == nil || nextTag == nil) && !nextIsBackground { return false }

        if prevSpan == nil || prevTag == nil {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }

        let prevIsBackground = (prevSpan == backgroundSpanIdx)
        if prevIsBackground {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }
        if prevTag == "E" || prevTag == "S" {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }
        if prevTag == "B" || prevTag == "I" {
            let sameSpan = prevSpan == nextSpan
            return sameSpan && (nextTag == "I" || nextTag == "E")
        }
        return false
    }

    private static func transitionBias(
        biases: ViterbiBiases,
        prevTag: String?, prevSpan: Int?,
        nextTag: String?, nextSpan: Int?,
        backgroundTokenIdx: Int, backgroundSpanIdx: Int,
        prevIdx: Int, nextIdx: Int
    ) -> Float {
        let prevIsBackground = (prevSpan == backgroundSpanIdx) || (prevIdx == backgroundTokenIdx)
        let nextIsBackground = (nextSpan == backgroundSpanIdx) || (nextIdx == backgroundTokenIdx)

        if prevIsBackground {
            if nextIsBackground { return biases.backgroundStay }
            if nextTag == "B" || nextTag == "S" { return biases.backgroundToStart }
            return 0.0
        }
        if prevTag == "B" || prevTag == "I" {
            if nextTag == "I" && prevSpan == nextSpan { return biases.insideToContinue }
            if nextTag == "E" && prevSpan == nextSpan { return biases.insideToEnd }
            return 0.0
        }
        if prevTag == "E" || prevTag == "S" {
            if nextIsBackground { return biases.endToBackground }
            if nextTag == "B" || nextTag == "S" { return biases.endToStart }
            return 0.0
        }
        return 0.0
    }

    /// Decode one `[seqLen][numClasses]` log-probability matrix into label ids.
    func decode(_ tokenLogProbs: [[Float]]) -> [Int] {
        let seqLen = tokenLogProbs.count
        if seqLen == 0 { return [] }
        let N = numClasses
        var scores = [Float](repeating: 0, count: N)
        for j in 0 ..< N {
            scores[j] = tokenLogProbs[0][j] + startScores[j]
        }

        // Backpointers: [seqLen-1][N]
        var backpointers = [[Int]](
            repeating: [Int](repeating: 0, count: N),
            count: max(0, seqLen - 1)
        )
        var nextScores = [Float](repeating: 0, count: N)

        for step in 1 ..< seqLen {
            var bp = [Int](repeating: 0, count: N)
            for nextLabel in 0 ..< N {
                var bestScore: Float = ViterbiDecoder.NEG_INF
                var bestPrev: Int = 0
                for prevLabel in 0 ..< N {
                    let candidate = scores[prevLabel] + transitionScores[prevLabel * N + nextLabel]
                    if candidate > bestScore {
                        bestScore = candidate
                        bestPrev = prevLabel
                    }
                }
                nextScores[nextLabel] = bestScore + tokenLogProbs[step][nextLabel]
                bp[nextLabel] = bestPrev
            }
            backpointers[step - 1] = bp
            // Swap scores buffer.
            for i in 0 ..< N { scores[i] = nextScores[i] }
        }

        // Check for any finite score; if not, fall back to argmax.
        var anyFinite = false
        for v in scores {
            if v > ViterbiDecoder.NEG_INF / 2 { anyFinite = true; break }
        }
        if !anyFinite {
            var path = [Int](repeating: 0, count: seqLen)
            for t in 0 ..< seqLen {
                var bestIdx = 0
                var bestVal: Float = -.infinity
                for c in 0 ..< N {
                    let v = tokenLogProbs[t][c]
                    if v > bestVal {
                        bestVal = v
                        bestIdx = c
                    }
                }
                path[t] = bestIdx
            }
            return path
        }

        // Add end scores and pick best last label.
        for j in 0 ..< N { scores[j] += endScores[j] }
        var lastLabel = 0
        var bestVal: Float = -.infinity
        for j in 0 ..< N {
            if scores[j] > bestVal {
                bestVal = scores[j]
                lastLabel = j
            }
        }

        var path = [Int](repeating: 0, count: seqLen)
        path[seqLen - 1] = lastLabel
        var cur = lastLabel
        var step = seqLen - 2
        while step >= 0 {
            cur = backpointers[step][cur]
            path[step] = cur
            step -= 1
        }
        return path
    }
}

/// Compute a numerically stable per-row log-softmax.
func logSoftmax(_ logits: [[Float]]) -> [[Float]] {
    var out: [[Float]] = []
    out.reserveCapacity(logits.count)
    for row in logits {
        var maxVal: Float = -.infinity
        for v in row { if v > maxVal { maxVal = v } }
        var sumExp: Float = 0
        for v in row { sumExp += expf(v - maxVal) }
        let logZ = maxVal + logf(sumExp)
        var lp = [Float](repeating: 0, count: row.count)
        for i in 0 ..< row.count { lp[i] = row[i] - logZ }
        out.append(lp)
    }
    return out
}
