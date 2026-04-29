import Foundation
import Darwin

struct ServeArgs {
    var checkpoint: String = "~/.opf/privacy_filter"
    var socketPath: String = "/tmp/opf-mlx-swift-text.sock"
    var context: Int = 128
    var moeChunkSize: Int = 2
    var decodeMode: String = "viterbi"
    var tokenizerPath: String? = nil
    var noDownload: Bool = false
}

func parseServeArgs(_ argv: [String]) -> ServeArgs {
    var args = ServeArgs()
    var i = 0
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--checkpoint":
            guard i + 1 < argv.count else { die("--checkpoint requires a value") }
            args.checkpoint = argv[i + 1]; i += 2
        case "--socket":
            guard i + 1 < argv.count else { die("--socket requires a value") }
            args.socketPath = argv[i + 1]; i += 2
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

private func textServeLog(_ message: String) {
    print(message)
    fflush(stdout)
}

private func textServeLogErr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func writeAllSock(_ fd: Int32, _ data: Data) -> Bool {
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

private struct TextLineReader {
    let fd: Int32
    private var buffer = Data()
    private var eof = false

    init(fd: Int32) { self.fd = fd }

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
            var tmp = [UInt8](repeating: 0, count: 65536)
            let n = tmp.withUnsafeMutableBufferPointer { ptr -> Int in
                read(fd, ptr.baseAddress, ptr.count)
            }
            if n < 0 {
                if errno == EINTR { continue }
                textServeLogErr("read failed: errno=\(errno)")
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

private func textBindUnixSocket(path: String) -> Int32 {
    if path.utf8.count >= 104 {
        die("socket path too long (\(path.utf8.count) >= 104 bytes): \(path)")
    }
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
    if bindResult < 0 { die("bind(\(path)) failed: errno=\(errno)") }
    if listen(fd, 16) < 0 { die("listen() failed: errno=\(errno)") }
    return fd
}

private func encodeTextResponse(_ dict: [String: Any]) -> Data {
    do {
        var data = try JSONSerialization.data(withJSONObject: dict, options: [])
        data.append(0x0A)
        return data
    } catch {
        return Data("{\"error\":\"failed to encode response\"}\n".utf8)
    }
}

private func processTextRequest(line: Data, pipeline: InferencePipeline) -> Data {
    let obj: [String: Any]
    do {
        guard let parsed = try JSONSerialization.jsonObject(with: line) as? [String: Any] else {
            return encodeTextResponse(["error": "request must be a JSON object"])
        }
        obj = parsed
    } catch {
        return encodeTextResponse(["error": "invalid JSON: \(error.localizedDescription)"])
    }

    let requestId = obj["request_id"] as? String

    guard let text = obj["text"] as? String else {
        var dict: [String: Any] = ["error": "missing or non-string 'text'"]
        if let id = requestId { dict["request_id"] = id }
        return encodeTextResponse(dict)
    }
    let modeOverride = obj["decode_mode"] as? String

    let start = Date()
    let result = pipeline.run(text: text, decodeModeOverride: modeOverride)
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
    var dict: [String: Any] = [
        "text": result.tokenization.decodedMismatch ? result.tokenization.decodedText : text,
        "detected_spans": spansArr,
        "redacted_text": result.redactedText,
        "latency_ms": elapsedMs,
        "token_count": result.tokenization.tokenIds.count,
    ]
    if let id = requestId { dict["request_id"] = id }
    if let mode = modeOverride { dict["decode_mode"] = mode } else { dict["decode_mode"] = pipeline.decodeMode }
    return encodeTextResponse(dict)
}

func runServe(_ argv: [String]) {
    let args = parseServeArgs(argv)
    let checkpoint = expandTilde(args.checkpoint)

    textServeLog("opf-mlx-daemon serve starting")
    textServeLog("checkpoint: \(checkpoint)")
    textServeLog("socket: \(args.socketPath)")
    textServeLog("context: \(args.context)")
    textServeLog("moe_chunk_size: \(args.moeChunkSize)")
    textServeLog("decode_mode: \(args.decodeMode)")

    let loadStart = Date()
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
    let loadElapsed = Date().timeIntervalSince(loadStart)
    textServeLog(String(format: "load_seconds: %.2f", loadElapsed))

    signal(SIGPIPE, SIG_IGN)

    let listenFd = textBindUnixSocket(path: args.socketPath)
    textServeLog("listening on \(args.socketPath)")
    textServeLog("ready")

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
            textServeLogErr("accept failed: errno=\(errno)")
            continue
        }
        var reader = TextLineReader(fd: cfd)
        while let line = reader.nextLine() {
            if line.isEmpty { continue }
            let response = processTextRequest(line: line, pipeline: pipeline)
            if !writeAllSock(cfd, response) {
                textServeLogErr("client write failed; closing connection")
                break
            }
        }
        close(cfd)
    }
}
