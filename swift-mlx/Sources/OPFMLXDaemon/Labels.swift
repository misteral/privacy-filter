import Foundation

/// OPF v2 span labels (33 BIESO classes when expanded).
/// Mirrors `opf._common.label_space` v2 taxonomy.
enum LabelSpace {
    static let backgroundClass: String = "O"
    static let boundaryPrefixes: [String] = ["B", "I", "E", "S"]

    /// v2 span class names, with `O` at index 0.
    static let v2SpanClassNames: [String] = [
        "O",
        "account_number",
        "private_address",
        "private_date",
        "private_email",
        "private_person",
        "private_phone",
        "private_url",
        "secret",
    ]

    /// v2 token-level (BIESO) class names: index 0 == "O", then for each non-O span
    /// label expand into B/I/E/S in order, matching `_expand_with_boundary_markers`.
    static var v2NerClassNames: [String] {
        var out: [String] = ["O"]
        for base in v2SpanClassNames where base != "O" {
            for prefix in boundaryPrefixes {
                out.append("\(prefix)-\(base)")
            }
        }
        return out
    }
}

/// Resolved label-space mappings for one taxonomy.
struct LabelInfo {
    /// Span class names (index 0 == "O").
    let spanClassNames: [String]
    /// Token-level NER labels (BIESO expansion); index = token label id.
    let nerClassNames: [String]
    /// Map from token label id -> span class id.
    let tokenToSpanLabel: [Int: Int]
    /// Map from token label id -> boundary tag ("B"/"I"/"E"/"S") or nil for "O".
    let tokenBoundaryTags: [Int: String?]
    /// Span class id of the "O" background label.
    let backgroundSpanLabel: Int
    /// Token class id of the "O" background label.
    let backgroundTokenLabel: Int

    /// Build a label-info for the OPF v2 33-label taxonomy.
    static func v2() -> LabelInfo {
        let spanNames = LabelSpace.v2SpanClassNames
        let nerNames = LabelSpace.v2NerClassNames
        var tokenToSpan: [Int: Int] = [:]
        var boundary: [Int: String?] = [:]
        var backgroundIdx: Int = 0
        var spanLookup: [String: Int] = [:]
        for (i, name) in spanNames.enumerated() { spanLookup[name] = i }
        for (idx, name) in nerNames.enumerated() {
            if name == LabelSpace.backgroundClass {
                tokenToSpan[idx] = spanLookup[LabelSpace.backgroundClass]!
                boundary[idx] = nil
                backgroundIdx = idx
                continue
            }
            let parts = name.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
            let prefix = String(parts[0])
            let base = String(parts[1])
            tokenToSpan[idx] = spanLookup[base]!
            boundary[idx] = prefix
        }
        return LabelInfo(
            spanClassNames: spanNames,
            nerClassNames: nerNames,
            tokenToSpanLabel: tokenToSpan,
            tokenBoundaryTags: boundary,
            backgroundSpanLabel: spanLookup[LabelSpace.backgroundClass]!,
            backgroundTokenLabel: backgroundIdx
        )
    }
}

/// Convert a span label name into its placeholder token ("private_email" -> "<PRIVATE_EMAIL>").
func labelPlaceholder(_ label: String) -> String {
    let upper = label.uppercased()
    var normalized = ""
    var inSep = false
    for ch in upper {
        if ch.isLetter || ch.isNumber {
            normalized.append(ch)
            inSep = false
        } else if !inSep {
            normalized.append("_")
            inSep = true
        }
    }
    while normalized.hasPrefix("_") { normalized.removeFirst() }
    while normalized.hasSuffix("_") { normalized.removeLast() }
    if normalized.isEmpty { normalized = "REDACTED" }
    return "<\(normalized)>"
}
