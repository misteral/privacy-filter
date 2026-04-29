import Foundation
import MLX

// MARK: - Model config

struct OPFModelConfig {
    var modelType: String = "privacy_filter"
    var encoding: String = "o200k_base"
    var numHiddenLayers: Int = 8
    var numExperts: Int = 128
    var expertsPerToken: Int = 4
    var vocabSize: Int = 200064
    var numLabels: Int = 33
    var hiddenSize: Int = 640
    var intermediateSize: Int = 640
    var headDim: Int = 64
    var numAttentionHeads: Int = 14
    var numKeyValueHeads: Int = 2
    var bidirectionalLeftContext: Int = 128
    var bidirectionalRightContext: Int = 128
    var initialContextLength: Int = 4096
    var ropeTheta: Float = 150000.0
    var ropeScalingFactor: Float = 32.0
    var ropeNtkAlpha: Float = 1.0
    var ropeNtkBeta: Float = 32.0
    var swigluLimit: Float = 7.0
    var packedGeglu: Bool = false
    var moeChunkSize: Int = 1

    static func from(json: [String: Any]) -> OPFModelConfig {
        var c = OPFModelConfig()
        if let v = json["model_type"] as? String { c.modelType = v }
        if let v = json["encoding"] as? String { c.encoding = v }
        if let v = json["num_hidden_layers"] as? Int { c.numHiddenLayers = v }
        if let v = json["num_experts"] as? Int { c.numExperts = v }
        if let v = json["experts_per_token"] as? Int { c.expertsPerToken = v }
        if let v = json["vocab_size"] as? Int { c.vocabSize = v }
        if let v = json["num_labels"] as? Int { c.numLabels = v }
        if let v = json["hidden_size"] as? Int { c.hiddenSize = v }
        if let v = json["intermediate_size"] as? Int { c.intermediateSize = v }
        if let v = json["head_dim"] as? Int { c.headDim = v }
        if let v = json["num_attention_heads"] as? Int { c.numAttentionHeads = v }
        if let v = json["num_key_value_heads"] as? Int { c.numKeyValueHeads = v }
        if let v = json["bidirectional_left_context"] as? Int { c.bidirectionalLeftContext = v }
        if let v = json["bidirectional_right_context"] as? Int { c.bidirectionalRightContext = v }
        if let v = json["initial_context_length"] as? Int { c.initialContextLength = v }
        if let v = json["rope_theta"] as? Double { c.ropeTheta = Float(v) }
        if let v = json["rope_theta"] as? Int { c.ropeTheta = Float(v) }
        if let v = json["rope_scaling_factor"] as? Double { c.ropeScalingFactor = Float(v) }
        if let v = json["rope_scaling_factor"] as? Int { c.ropeScalingFactor = Float(v) }
        if let v = json["rope_ntk_alpha"] as? Double { c.ropeNtkAlpha = Float(v) }
        if let v = json["rope_ntk_alpha"] as? Int { c.ropeNtkAlpha = Float(v) }
        if let v = json["rope_ntk_beta"] as? Double { c.ropeNtkBeta = Float(v) }
        if let v = json["rope_ntk_beta"] as? Int { c.ropeNtkBeta = Float(v) }
        if let v = json["swiglu_limit"] as? Double { c.swigluLimit = Float(v) }
        if let v = json["swiglu_limit"] as? Int { c.swigluLimit = Float(v) }
        if let v = json["packed_geglu"] as? Bool { c.packedGeglu = v }
        return c
    }
}

// MARK: - Functional building blocks

private let NEG_INF: Float = -1.0e9

func opfLinear(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray? = nil) -> MLXArray {
    var y = matmul(x.asType(weight.dtype), weight.T)
    if let bias = bias {
        y = y + bias
    }
    return y
}

func opfRMSNorm(_ x: MLXArray, _ scale: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let dtype = x.dtype
    let t = x.asType(.float32)
    let normed = t * rsqrt(mean(t * t, axis: -1, keepDims: true) + eps)
    return (normed * scale.asType(.float32)).asType(dtype)
}

func opfSwiGLU(_ x: MLXArray, limit: Float, packed: Bool) -> MLXArray {
    let xGLU: MLXArray
    let xLinear: MLXArray
    if packed {
        xGLU = x[.ellipsis, .stride(from: 0, by: 2)]
        xLinear = x[.ellipsis, .stride(from: 1, by: 2)]
    } else {
        let half = x.shape.last! / 2
        xGLU = x[.ellipsis, 0 ..< half]
        xLinear = x[.ellipsis, half ..< (2 * half)]
    }
    let glu = minimum(xGLU, limit)
    let lin = clip(xLinear, min: -limit, max: limit)
    return (glu * sigmoid(Float(1.702) * glu)) * (lin + Float(1.0))
}

/// Apply expert weights stored as `[batch, in, out]` to inputs of shape `[batch, in]`.
func opfBatchedExpertLinear(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
    let xExpanded = expandedDimensions(x.asType(weight.dtype), axis: 1)
    let y = matmul(xExpanded, weight)
    return squeezed(y, axis: 1) + bias
}

// MARK: - YaRN RoPE

final class OPFRotary {
    let headDim: Int
    let base: Float
    let initialContextLength: Int
    let scalingFactor: Float
    let ntkAlpha: Float
    let ntkBeta: Float
    let maxPositionEmbeddings: Int
    private var cosCache: MLXArray
    private var sinCache: MLXArray

    init(_ cfg: OPFModelConfig) {
        self.headDim = cfg.headDim
        self.base = cfg.ropeTheta
        self.initialContextLength = cfg.initialContextLength
        self.scalingFactor = cfg.ropeScalingFactor
        self.ntkAlpha = cfg.ropeNtkAlpha
        self.ntkBeta = cfg.ropeNtkBeta
        self.maxPositionEmbeddings = max(
            Int(Float(cfg.initialContextLength) * cfg.ropeScalingFactor),
            cfg.initialContextLength
        )
        let (cosA, sinA) = OPFRotary.computeCosSin(
            numTokens: self.maxPositionEmbeddings,
            headDim: self.headDim,
            base: self.base,
            initialContextLength: self.initialContextLength,
            scalingFactor: self.scalingFactor,
            ntkAlpha: self.ntkAlpha,
            ntkBeta: self.ntkBeta
        )
        self.cosCache = cosA
        self.sinCache = sinA
    }

    private static func concentrationAndInvFreq(
        headDim: Int, base: Float, initialContextLength: Int,
        scalingFactor: Float, ntkAlpha: Float, ntkBeta: Float
    ) -> (Float, [Float]) {
        let dHalf = Float(headDim) / 2.0
        // freq[i] = base ** ( (2i) / head_dim ) for i in 0..d/2
        var freq = [Float](repeating: 0, count: Int(dHalf))
        for i in 0 ..< Int(dHalf) {
            let exponent = Float(2 * i) / Float(headDim)
            freq[i] = powf(base, exponent)
        }
        if scalingFactor > 1.0 {
            let concentration = 0.1 * logf(scalingFactor) + 1.0
            let logBase = logf(base)
            let twoPi = Float(2.0 * Double.pi)
            let low = Double(dHalf) * Double(logf(Float(initialContextLength) / (ntkBeta * twoPi))) / Double(logBase)
            let high = Double(dHalf) * Double(logf(Float(initialContextLength) / (ntkAlpha * twoPi))) / Double(logBase)
            var invFreq = [Float](repeating: 0, count: freq.count)
            for i in 0 ..< freq.count {
                let interpolation = 1.0 / (scalingFactor * freq[i])
                let extrapolation = 1.0 / freq[i]
                let ramp = (Double(i) - low) / (high - low)
                let mask = Float(1.0 - min(max(ramp, 0.0), 1.0))
                invFreq[i] = interpolation * (1.0 - mask) + extrapolation * mask
            }
            return (concentration, invFreq)
        } else {
            var invFreq = [Float](repeating: 0, count: freq.count)
            for i in 0 ..< freq.count { invFreq[i] = 1.0 / freq[i] }
            return (1.0, invFreq)
        }
    }

    private static func computeCosSin(
        numTokens: Int, headDim: Int, base: Float, initialContextLength: Int,
        scalingFactor: Float, ntkAlpha: Float, ntkBeta: Float
    ) -> (MLXArray, MLXArray) {
        let (concentration, invFreqArr) = concentrationAndInvFreq(
            headDim: headDim, base: base, initialContextLength: initialContextLength,
            scalingFactor: scalingFactor, ntkAlpha: ntkAlpha, ntkBeta: ntkBeta
        )
        let invFreq = MLXArray(invFreqArr).asType(.float32)
        let t = arange(numTokens).asType(.float32)
        let freqs = outer(t, invFreq)
        let cosA = cos(freqs) * concentration
        let sinA = sin(freqs) * concentration
        return (cosA, sinA)
    }

    private func ensureLen(_ numTokens: Int) {
        if numTokens <= cosCache.shape[0] { return }
        let (c, s) = OPFRotary.computeCosSin(
            numTokens: numTokens,
            headDim: headDim, base: base, initialContextLength: initialContextLength,
            scalingFactor: scalingFactor, ntkAlpha: ntkAlpha, ntkBeta: ntkBeta
        )
        cosCache = c
        sinCache = s
    }

    private static func apply(_ x: MLXArray, cosA: MLXArray, sinA: MLXArray) -> MLXArray {
        // x: [B, T, H, D], cos/sin: [T, D/2]
        let cosE = expandedDimensions(cosA, axes: [0, 2]).asType(x.dtype)
        let sinE = expandedDimensions(sinA, axes: [0, 2]).asType(x.dtype)
        let x1 = x[.ellipsis, .stride(from: 0, by: 2)]
        let x2 = x[.ellipsis, .stride(from: 1, by: 2)]
        let y1 = x1 * cosE - x2 * sinE
        let y2 = x2 * cosE + x1 * sinE
        return stacked([y1, y2], axis: -1).reshaped(x.shape)
    }

    func callAsFunction(_ query: MLXArray, _ key: MLXArray) -> (MLXArray, MLXArray) {
        let batch = query.shape[0]
        let tokens = query.shape[1]
        ensureLen(tokens)
        let cosA = cosCache[0 ..< tokens]
        let sinA = sinCache[0 ..< tokens]
        let qShape = query.shape
        let kShape = key.shape
        let q4 = query.reshaped([batch, tokens, -1, headDim])
        let k4 = key.reshaped([batch, tokens, -1, headDim])
        let qOut = OPFRotary.apply(q4, cosA: cosA, sinA: sinA).reshaped(qShape)
        let kOut = OPFRotary.apply(k4, cosA: cosA, sinA: sinA).reshaped(kShape)
        return (qOut, kOut)
    }
}

// MARK: - Attention block

final class OPFAttention {
    let cfg: OPFModelConfig
    let sinks: MLXArray
    let normScale: MLXArray
    let qkvWeight: MLXArray
    let qkvBias: MLXArray
    let outWeight: MLXArray
    let outBias: MLXArray
    let rope: OPFRotary
    let qkScale: Float

    init(cfg: OPFModelConfig, weights: [String: MLXArray], prefix: String) throws {
        self.cfg = cfg
        self.sinks = try OPFAttention.required(weights, "\(prefix).sinks")
        self.normScale = try OPFAttention.required(weights, "\(prefix).norm.scale")
        self.qkvWeight = try OPFAttention.required(weights, "\(prefix).qkv.weight")
        self.qkvBias = try OPFAttention.required(weights, "\(prefix).qkv.bias")
        self.outWeight = try OPFAttention.required(weights, "\(prefix).out.weight")
        self.outBias = try OPFAttention.required(weights, "\(prefix).out.bias")
        self.rope = OPFRotary(cfg)
        // Python: 1.0 / sqrt(sqrt(head_dim))
        self.qkScale = 1.0 / sqrtf(sqrtf(Float(cfg.headDim)))
    }

    private static func required(_ weights: [String: MLXArray], _ key: String) throws -> MLXArray {
        guard let w = weights[key] else {
            throw OPFError.missingTensor(key)
        }
        return w
    }

    private func localSDPA(q: MLXArray, k: MLXArray, v: MLXArray) -> MLXArray {
        // q: [B, T, Hkv, Qmult, D]; k/v: [B, T, Hkv, D]
        let bsz = q.shape[0]
        let nTokens = q.shape[1]
        let nKvHeads = q.shape[2]
        let qMult = q.shape[3]
        let dHead = q.shape[4]
        let left = cfg.bidirectionalLeftContext
        let right = cfg.bidirectionalRightContext
        let window = left + right + 1

        let widths: [IntOrPair] = [[0, 0], [left, right], [0, 0], [0, 0]]
        let kPadded = padded(k, widths: widths)
        let vPadded = padded(v, widths: widths)

        let offsets = arange(window) - left
        // positions: [T, window]
        let positions = expandedDimensions(arange(nTokens), axis: 1) + expandedDimensions(offsets, axis: 0)
        let valid = (positions .>= 0) .&& (positions .< nTokens)
        let gather = clip(positions + left, min: 0, max: nTokens + left + right - 1).asType(.int32)

        // k_padded: [B, T+left+right, Hkv, D]; gather: [T, window]
        // kWin: [B, T, window, Hkv, D]
        let kWin = take(kPadded, gather, axis: 1)
        let vWin = take(vPadded, gather, axis: 1)

        // scores: [B, T, Hkv, Qmult, window]
        let scores = einsum("bthqd,btwhd->bthqw", q, kWin).asType(.float32)
        let validMask = expandedDimensions(valid, axes: [0, 2, 3])  // [1, T, 1, 1, window]
        let scoresMasked = which(validMask, scores, NEG_INF)

        // Python: sink_scores = (self.sinks * log(2.0)).reshape(n_kv_heads, q_mult)
        let sinkBase = (sinks * Float(log(2.0))).reshaped([nKvHeads, qMult])
        // Expand to [B, T, Hkv, Qmult, 1]
        var sinkExpanded = expandedDimensions(sinkBase, axes: [0, 1, 4])  // [1,1,Hkv,Qmult,1]
        sinkExpanded = broadcast(sinkExpanded, to: [bsz, nTokens, nKvHeads, qMult, 1]).asType(.float32)

        let scoresAll = concatenated([scoresMasked, sinkExpanded], axis: -1)
        let probsAll = softmax(scoresAll, axis: -1)
        // drop the trailing sink slot
        let probs = probsAll[.ellipsis, 0 ..< window].asType(v.dtype)
        let attn = einsum("bthqw,btwhd->bthqd", probs, vWin)
        return attn.reshaped([bsz, nTokens, nKvHeads * qMult * dHead])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let t = opfRMSNorm(x, normScale)
        let qkv = opfLinear(t, qkvWeight, qkvBias)
        let qEnd = cfg.numAttentionHeads * cfg.headDim
        let kEnd = qEnd + cfg.numKeyValueHeads * cfg.headDim
        let vEnd = kEnd + cfg.numKeyValueHeads * cfg.headDim
        let qSlice = qkv[0..., 0..., 0 ..< qEnd]
        let kSlice = qkv[0..., 0..., qEnd ..< kEnd]
        let vSlice = qkv[0..., 0..., kEnd ..< vEnd]
        var (qRot, kRot) = rope(qSlice, kSlice)
        qRot = qRot * qkScale
        kRot = kRot * qkScale
        let bsz = qRot.shape[0]
        let nTokens = qRot.shape[1]
        let qMult = cfg.numAttentionHeads / cfg.numKeyValueHeads
        let qReshaped = qRot.reshaped([bsz, nTokens, cfg.numKeyValueHeads, qMult, cfg.headDim])
        let kReshaped = kRot.reshaped([bsz, nTokens, cfg.numKeyValueHeads, cfg.headDim])
        let vReshaped = vSlice.reshaped([bsz, nTokens, cfg.numKeyValueHeads, cfg.headDim])
        let attnOut = localSDPA(q: qReshaped, k: kReshaped, v: vReshaped)
        let proj = opfLinear(attnOut, outWeight, outBias).asType(x.dtype)
        return x + proj
    }
}

// MARK: - MoE MLP

final class OPFMLP {
    let cfg: OPFModelConfig
    let normScale: MLXArray
    let gateWeight: MLXArray
    let gateBias: MLXArray
    let w1: MLXArray
    let b1: MLXArray
    let w2: MLXArray
    let b2: MLXArray

    init(cfg: OPFModelConfig, weights: [String: MLXArray], prefix: String) throws {
        self.cfg = cfg
        self.normScale = try OPFMLP.required(weights, "\(prefix).norm.scale")
        self.gateWeight = try OPFMLP.required(weights, "\(prefix).gate.weight")
        self.gateBias = try OPFMLP.required(weights, "\(prefix).gate.bias")
        self.w1 = try OPFMLP.required(weights, "\(prefix).swiglu.weight")
        self.b1 = try OPFMLP.required(weights, "\(prefix).swiglu.bias")
        self.w2 = try OPFMLP.required(weights, "\(prefix).out.weight")
        self.b2 = try OPFMLP.required(weights, "\(prefix).out.bias")
    }

    private static func required(_ weights: [String: MLXArray], _ key: String) throws -> MLXArray {
        guard let w = weights[key] else { throw OPFError.missingTensor(key) }
        return w
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchShape = Array(x.shape.dropLast())
        let hidden = x.shape.last!
        let normed = opfRMSNorm(x, normScale).reshaped([-1, hidden])

        let gate = opfLinear(
            normed.asType(.float32),
            gateWeight.asType(.float32),
            gateBias.asType(.float32)
        )
        // Top-k via argSort on negated gates.
        let sorted = argSort(-gate, axis: -1)
        let expertIndices = sorted[0..., 0 ..< cfg.expertsPerToken]
        let expertValues = takeAlong(gate, expertIndices, axis: -1)
        let expertWeights = softmax(expertValues, axis: -1)

        var outputs: [MLXArray] = []
        let chunkSize = max(1, cfg.moeChunkSize)
        let nTokens = normed.shape[0]
        for start in stride(from: 0, to: nTokens, by: chunkSize) {
            let end = min(start + chunkSize, nTokens)
            let tChunk = normed[start ..< end]
            let idxChunk = expertIndices[start ..< end]
            let weightChunk = expertWeights[start ..< end]
            var accum = zeros([end - start, cfg.hiddenSize], dtype: .float32)
            for slot in 0 ..< cfg.expertsPerToken {
                let idx = idxChunk[0..., slot]
                let w1Sel = take(w1, idx, axis: 0)
                let b1Sel = take(b1, idx, axis: 0)
                let w2Sel = take(w2, idx, axis: 0)
                let b2Sel = take(b2, idx, axis: 0)
                var h = opfBatchedExpertLinear(tChunk, w1Sel, b1Sel)
                h = opfSwiGLU(h, limit: cfg.swigluLimit, packed: cfg.packedGeglu)
                let o = opfBatchedExpertLinear(h, w2Sel, b2Sel).asType(.float32)
                let slotW = weightChunk[0..., slot ..< (slot + 1)].asType(.float32)
                accum = accum + o * slotW
            }
            outputs.append(accum.asType(x.dtype))
        }
        let yFlat = concatenated(outputs, axis: 0)
        let y = yFlat.reshaped(batchShape + [yFlat.shape.last!])
        return x + y
    }
}

// MARK: - Block + Transformer

final class OPFBlock {
    let attn: OPFAttention
    let mlp: OPFMLP

    init(cfg: OPFModelConfig, weights: [String: MLXArray], layerIdx: Int) throws {
        self.attn = try OPFAttention(cfg: cfg, weights: weights, prefix: "block.\(layerIdx).attn")
        self.mlp = try OPFMLP(cfg: cfg, weights: weights, prefix: "block.\(layerIdx).mlp")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return mlp(attn(x))
    }
}

enum OPFError: Error, CustomStringConvertible {
    case missingTensor(String)
    case shapeMismatch(String)

    var description: String {
        switch self {
        case .missingTensor(let key): return "missing tensor: \(key)"
        case .shapeMismatch(let msg): return "shape mismatch: \(msg)"
        }
    }
}

final class OPFTransformer {
    let cfg: OPFModelConfig
    let embedding: MLXArray
    let blocks: [OPFBlock]
    let normScale: MLXArray
    let unembedding: MLXArray

    init(cfg: OPFModelConfig, weights: [String: MLXArray]) throws {
        self.cfg = cfg
        guard let emb = weights["embedding.weight"] else { throw OPFError.missingTensor("embedding.weight") }
        guard let n = weights["norm.scale"] else { throw OPFError.missingTensor("norm.scale") }
        guard let u = weights["unembedding.weight"] else { throw OPFError.missingTensor("unembedding.weight") }
        self.embedding = emb
        self.normScale = n
        self.unembedding = u
        var built: [OPFBlock] = []
        built.reserveCapacity(cfg.numHiddenLayers)
        for i in 0 ..< cfg.numHiddenLayers {
            built.append(try OPFBlock(cfg: cfg, weights: weights, layerIdx: i))
        }
        self.blocks = built
    }

    func callAsFunction(_ tokenIds: MLXArray) -> MLXArray {
        precondition(tokenIds.ndim == 2, "Transformer expects [batch, tokens] int token ids")
        // embedding[token_ids] — fancy indexing along axis 0
        var x = take(embedding, tokenIds, axis: 0)
        for block in blocks {
            x = block(x)
            eval(x)
        }
        x = opfRMSNorm(x, normScale)
        return opfLinear(x, unembedding, nil)
    }
}
