import Foundation

/// One detected character span ready for serialization.
struct DetectedSpan {
    let label: String
    let start: Int
    let end: Int
    let text: String
    let placeholder: String
}

/// Convert per-token label ids into token-index spans using BIESO rules.
/// Mirrors `opf._core.spans.labels_to_spans`.
func labelsToSpans(_ labelsByIndex: [Int: Int], labelInfo: LabelInfo) -> [(Int, Int, Int)] {
    var spans: [(Int, Int, Int)] = []
    var currentLabel: Int? = nil
    var startIdx: Int? = nil
    var previousIdx: Int? = nil
    let backgroundSpanLabel = labelInfo.backgroundSpanLabel

    for tokenIdx in labelsByIndex.keys.sorted() {
        let labelId = labelsByIndex[tokenIdx]!
        let spanLabel = labelInfo.tokenToSpanLabel[labelId]
        let boundaryTag = labelInfo.tokenBoundaryTags[labelId] ?? nil

        if let prev = previousIdx, tokenIdx != prev + 1 {
            if let cl = currentLabel, let sIdx = startIdx {
                spans.append((cl, sIdx, prev + 1))
            }
            currentLabel = nil
            startIdx = nil
        }

        guard let spanLabelV = spanLabel else {
            previousIdx = tokenIdx
            continue
        }

        let isBackground = spanLabelV == backgroundSpanLabel
        if isBackground {
            if let cl = currentLabel, let sIdx = startIdx {
                spans.append((cl, sIdx, tokenIdx))
            }
            currentLabel = nil
            startIdx = nil
            previousIdx = tokenIdx
            continue
        }

        switch boundaryTag {
        case "S":
            if let cl = currentLabel, let sIdx = startIdx, let prev = previousIdx {
                spans.append((cl, sIdx, prev + 1))
            }
            spans.append((spanLabelV, tokenIdx, tokenIdx + 1))
            currentLabel = nil
            startIdx = nil
        case "B":
            if let cl = currentLabel, let sIdx = startIdx, let prev = previousIdx {
                spans.append((cl, sIdx, prev + 1))
            }
            currentLabel = spanLabelV
            startIdx = tokenIdx
        case "I":
            if currentLabel == nil || currentLabel != spanLabelV {
                if let cl = currentLabel, let sIdx = startIdx, let prev = previousIdx {
                    spans.append((cl, sIdx, prev + 1))
                }
                currentLabel = spanLabelV
                startIdx = tokenIdx
            }
        case "E":
            if currentLabel == nil || currentLabel != spanLabelV || startIdx == nil {
                if let cl = currentLabel, let sIdx = startIdx, let prev = previousIdx {
                    spans.append((cl, sIdx, prev + 1))
                }
                spans.append((spanLabelV, tokenIdx, tokenIdx + 1))
                currentLabel = nil
                startIdx = nil
            } else {
                spans.append((currentLabel!, startIdx!, tokenIdx + 1))
                currentLabel = nil
                startIdx = nil
            }
        default:
            if let cl = currentLabel, let sIdx = startIdx, let prev = previousIdx {
                spans.append((cl, sIdx, prev + 1))
            }
            currentLabel = nil
            startIdx = nil
        }

        previousIdx = tokenIdx
    }

    if let cl = currentLabel, let sIdx = startIdx, let prev = previousIdx {
        spans.append((cl, sIdx, prev + 1))
    }
    return spans
}

/// Convert token-index spans into character-index spans using char_starts / char_ends.
func tokenSpansToCharSpans(
    _ spans: [(Int, Int, Int)],
    charStarts: [Int],
    charEnds: [Int]
) -> [(Int, Int, Int)] {
    var out: [(Int, Int, Int)] = []
    for (label, tokenStart, tokenEnd) in spans {
        guard tokenStart >= 0, tokenEnd > tokenStart, tokenEnd <= charStarts.count else { continue }
        let cs = charStarts[tokenStart]
        let ce = charEnds[tokenEnd - 1]
        if ce <= cs { continue }
        out.append((label, cs, ce))
    }
    return out
}

/// Trim leading and trailing whitespace from char spans, indexed in `Character` space.
func trimCharSpansWhitespace(
    _ spans: [(Int, Int, Int)],
    text: String
) -> [(Int, Int, Int)] {
    let chars = Array(text)
    var out: [(Int, Int, Int)] = []
    for (label, sIn, eIn) in spans {
        var s = sIn
        var e = eIn
        if s < 0 || e <= s || e > chars.count { continue }
        while s < e, chars[s].isWhitespace { s += 1 }
        while e > s, chars[e - 1].isWhitespace { e -= 1 }
        if e > s {
            out.append((label, s, e))
        }
    }
    return out
}

/// Greedy left-to-right non-overlapping span selection.
/// Mirrors `_select_non_overlapping_spans` in `opf._core.runtime`.
func selectNonOverlappingSpans(_ spans: [DetectedSpan]) -> [DetectedSpan] {
    let ordered = spans.sorted { lhs, rhs in
        if lhs.start != rhs.start { return lhs.start < rhs.start }
        let lhsLen = lhs.end - lhs.start
        let rhsLen = rhs.end - rhs.start
        if lhsLen != rhsLen { return lhsLen > rhsLen }
        return lhs.label < rhs.label
    }
    var kept: [DetectedSpan] = []
    var cursor = 0
    for span in ordered {
        if span.start < cursor || span.end <= span.start { continue }
        kept.append(span)
        cursor = span.end
    }
    return kept
}

/// Build redacted text by replacing kept spans with their placeholder tokens.
/// `text` is the source (decoded) text and `spans` are non-overlapping char spans
/// indexed in Swift `Character` space.
func buildRedactedText(_ text: String, spans: [DetectedSpan]) -> String {
    if spans.isEmpty { return text }
    let chars = Array(text)
    var out = ""
    var cursor = 0
    let ordered = spans.sorted { $0.start < $1.start }
    for span in ordered {
        if span.start < cursor { continue }
        if span.end > chars.count { break }
        out.append(contentsOf: chars[cursor ..< span.start])
        out.append(span.placeholder)
        cursor = span.end
    }
    if cursor < chars.count {
        out.append(contentsOf: chars[cursor ..< chars.count])
    }
    return out
}
