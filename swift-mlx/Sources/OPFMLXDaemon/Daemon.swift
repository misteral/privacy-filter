import Foundation
import Darwin
import MLX

// MARK: - Daemon args

struct DaemonArgs {
    var checkpoint: String = "~/.opf/privacy_filter"
    var socketPath: String = "/tmp/opf-mlx-swift.sock"
    var context: Int = 128
    var moeChunkSize: Int = 1
}

func parseDaemonArgs(_ argv: [String]) -> DaemonArgs {
    var args = DaemonArgs()
    var i = 0
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--checkpoint":
            guard i + 1 < argv.count else { die("--checkpoint requires a value") }
            args.checkpoint = argv[i + 1]
            i += 2
        case "--socket":
            guard i + 1 < argv.count else { die("--socket requires a value") }
            args.socketPath = argv[i + 1]
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
    return args
}

// MARK: - Logging helpers

@inline(__always)
private func logStdout(_ message: String) {
    print(message)
    fflush(stdout)
}

@inline(__always)
private func logStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// MARK: - Socket I/O

private func writeAll(_ fd: Int32, _ data: Data) -> Bool {
    let total = data.count
    if total == 0 { return true }
    return data.withUnsafeBytes { rawBuf -> Bool in
        guard let base = rawBuf.baseAddress else { return true }
        var sent = 0
        while sent < total {
            let n = write(fd, base.advanced(by: sent), total - sent)
            if n < 0 {
                if errno == EINTR { continue }
                return false
            }
            if n == 0 { return false }
            sent += n
        }
        return true
    }
}

private struct LineReader {
    let fd: Int32
    private var buffer = Data()
    private var eof = false

    init(fd: Int32) {
        self.fd = fd
    }

    mutating func nextLine() -> Data? {
        while true {
            if let nl = buffer.firstIndex(of: 0x0A) {
                let line = Data(buffer[buffer.startIndex ..< nl])
                buffer = Data(buffer[(nl + 1) ..< buffer.endIndex])
                return line
            }
            if eof {
                if buffer.isEmpty { return nil }
                let leftover = buffer
                buffer = Data()
                return leftover
            }
            var tmp = [UInt8](repeating: 0, count: 8192)
            let n = tmp.withUnsafeMutableBufferPointer { ptr -> Int in
                read(fd, ptr.baseAddress, ptr.count)
            }
            if n < 0 {
                if errno == EINTR { continue }
                logStderr("read failed: errno=\(errno)")
                return nil
            }
            if n == 0 {
                eof = true
                continue
            }
            buffer.append(contentsOf: tmp.prefix(n))
        }
    }
}

private func bindUnixSocket(path: String) -> Int32 {
    if path.utf8.count >= 104 {
        die("socket path too long (\(path.utf8.count) >= 104 bytes): \(path)")
    }
    // Remove stale socket; ignore failure if it doesn't exist.
    unlink(path)

    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    if fd < 0 { die("socket() failed: errno=\(errno)") }

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = Array(path.utf8)
    withUnsafeMutablePointer(to: &addr.sun_path) { tuplePtr in
        tuplePtr.withMemoryRebound(to: CChar.self, capacity: 104) { cPtr in
            for idx in 0 ..< pathBytes.count {
                cPtr[idx] = CChar(bitPattern: pathBytes[idx])
            }
            cPtr[pathBytes.count] = 0
        }
    }

    let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sptr in
            bind(fd, sptr, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    if bindResult < 0 {
        die("bind(\(path)) failed: errno=\(errno)")
    }

    if listen(fd, 16) < 0 {
        die("listen() failed: errno=\(errno)")
    }
    return fd
}

// MARK: - JSON helpers

private func encodeResponse(_ dict: [String: Any]) -> Data {
    do {
        var data = try JSONSerialization.data(withJSONObject: dict, options: [])
        data.append(0x0A)
        return data
    } catch {
        let raw = "{\"error\":\"failed to encode response\"}\n"
        return Data(raw.utf8)
    }
}

private func successResponse(requestId: String?, labels: [Int32], latencyMs: Double) -> Data {
    var dict: [String: Any] = [
        "labels": labels.map { Int($0) },
        "latency_ms": latencyMs,
    ]
    if let id = requestId { dict["request_id"] = id }
    return encodeResponse(dict)
}

private func errorResponse(requestId: String?, message: String) -> Data {
    var dict: [String: Any] = ["error": message]
    if let id = requestId { dict["request_id"] = id }
    return encodeResponse(dict)
}

private func processRequest(
    line: Data,
    model: OPFTransformer,
    cfg: OPFModelConfig,
    contextLimit: Int
) -> Data {
    let obj: [String: Any]
    do {
        guard let parsed = try JSONSerialization.jsonObject(with: line) as? [String: Any] else {
            return errorResponse(requestId: nil, message: "request must be a JSON object")
        }
        obj = parsed
    } catch {
        return errorResponse(requestId: nil, message: "invalid JSON: \(error.localizedDescription)")
    }

    let requestId = obj["request_id"] as? String

    guard let rawTokens = obj["tokens"] as? [Any] else {
        return errorResponse(requestId: requestId, message: "missing or non-array 'tokens'")
    }
    if rawTokens.isEmpty {
        return errorResponse(requestId: requestId, message: "'tokens' must be non-empty")
    }
    if rawTokens.count > contextLimit {
        return errorResponse(
            requestId: requestId,
            message: "tokens length \(rawTokens.count) exceeds --context \(contextLimit)"
        )
    }

    var tokens: [Int32] = []
    tokens.reserveCapacity(rawTokens.count)
    for (idx, value) in rawTokens.enumerated() {
        guard let num = value as? NSNumber else {
            return errorResponse(requestId: requestId, message: "tokens[\(idx)] is not an integer")
        }
        let intValue = num.intValue
        if intValue < 0 || intValue >= cfg.vocabSize {
            return errorResponse(
                requestId: requestId,
                message: "tokens[\(idx)]=\(intValue) out of range [0, \(cfg.vocabSize))"
            )
        }
        tokens.append(Int32(intValue))
    }

    let start = Date()
    let tokenArray = MLXArray(tokens, [1, tokens.count])
    let logits = model(tokenArray).asType(.float32)
    let labelsArr = argMax(logits, axis: -1).asType(.int32)
    eval(labelsArr)
    let elapsedMs = Date().timeIntervalSince(start) * 1000.0

    let labels: [Int32] = labelsArr.asArray(Int32.self)
    return successResponse(requestId: requestId, labels: labels, latencyMs: elapsedMs)
}

// MARK: - Entry

func runServeTokens(_ argv: [String]) {
    let args = parseDaemonArgs(argv)
    let checkpoint = expandTilde(args.checkpoint)
    let checkpointURL = URL(fileURLWithPath: checkpoint, isDirectory: true)
    let configURL = checkpointURL.appendingPathComponent("config.json")
    let safetensorsURL = checkpointURL.appendingPathComponent("model.safetensors")

    let fm = FileManager.default
    guard fm.fileExists(atPath: configURL.path) else { die("config.json not found at \(configURL.path)") }
    guard fm.fileExists(atPath: safetensorsURL.path) else { die("model.safetensors not found at \(safetensorsURL.path)") }

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

    logStdout("opf-mlx-daemon serve-tokens starting")
    logStdout("checkpoint: \(checkpointURL.path)")
    logStdout("socket: \(args.socketPath)")
    logStdout("context: \(args.context)")
    logStdout("moe_chunk_size: \(args.moeChunkSize)")

    let loadStart = Date()
    let weights: [String: MLXArray]
    do {
        weights = try MLX.loadArrays(url: safetensorsURL)
    } catch {
        die("failed to load model.safetensors: \(error)")
    }

    let model: OPFTransformer
    do {
        model = try OPFTransformer(cfg: cfg, weights: weights)
    } catch let e as OPFError {
        die("model build failed: \(e.description)")
    } catch {
        die("model build failed: \(error)")
    }
    let loadElapsed = Date().timeIntervalSince(loadStart)
    logStdout(String(format: "load_seconds: %.2f", loadElapsed))

    // Avoid SIGPIPE killing the daemon when a client closes mid-write.
    signal(SIGPIPE, SIG_IGN)

    let listenFd = bindUnixSocket(path: args.socketPath)
    logStdout("listening on \(args.socketPath)")
    logStdout("ready")

    while true {
        var clientAddr = sockaddr_un()
        var clientLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        let cfd = withUnsafeMutablePointer(to: &clientAddr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sptr in
                accept(listenFd, sptr, &clientLen)
            }
        }
        if cfd < 0 {
            if errno == EINTR { continue }
            logStderr("accept failed: errno=\(errno)")
            continue
        }

        var reader = LineReader(fd: cfd)
        while let line = reader.nextLine() {
            if line.isEmpty { continue }
            let response = processRequest(
                line: line,
                model: model,
                cfg: cfg,
                contextLimit: args.context
            )
            if !writeAll(cfd, response) {
                logStderr("client write failed; closing connection")
                break
            }
        }
        close(cfd)
    }
}
