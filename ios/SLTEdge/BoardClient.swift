// The server leg. The phone sends keypoints, not video: pixel coordinates in the crop plus the box,
// crop and source size the server needs for its own square-norm (demo_server._square_norm_fn), so
// the coordinate frame the frozen encoder was trained on is still computed where it always was.
// Endpoint /translate_kp is the keypoint twin of /translate; it enters the pipeline after the pose
// loop and before to_model_inputs.
import Foundation
import Network

/// Route check independent of demo_server: a TCP connect to the board's SSH port (22 always
/// listens on jetson-lpcv-03). "ready" or "refused" both mean the tunnel routes to the board;
/// a timeout means it does not -- usually the phone's WireGuard AllowedIPs lacking 192.168.1.0/24.
func tcpReach(host: String, port: UInt16, timeout: TimeInterval = 6) -> String {
    let conn = NWConnection(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!, using: .tcp)
    let sem = DispatchSemaphore(value: 0)
    var out = "timeout after \(Int(timeout))s"
    let t0 = CFAbsoluteTimeGetCurrent()
    conn.stateUpdateHandler = { st in
        switch st {
        case .ready: out = String(format: "ready (%.0f ms)", (CFAbsoluteTimeGetCurrent() - t0) * 1e3); sem.signal()
        case .failed(let e): out = "failed: \(e)"; sem.signal()
        case .waiting(let e): out = "waiting: \(e)"
        default: break
        }
    }
    conn.start(queue: DispatchQueue(label: "sltedge.reach"))
    _ = sem.wait(timeout: .now() + timeout)
    conn.cancel()
    return out
}

final class BoardClient {
    var baseURL: URL
    var beams = 4
    var maxNewTokens = 64

    init(baseURL: URL) { self.baseURL = baseURL }

    /// JSON body. Written by hand rather than JSONSerialization so that a NaN -- which must not reach
    /// here, the recorder drops those frames -- would be rejected loudly rather than silently nulled.
    static func payload(keypoints: [[SIMD2<Float>]], scores: [[Float]], bboxNorm: [Float],
                        sourceWH: (Int, Int), crop: Crop, srcFPS: Double, keepFPS: Double,
                        captured: Int, dropped: Int) -> Data {
        var s = "{\"format\":\"sltedge-kp/1\",\"keypoints\":["
        s.reserveCapacity(keypoints.count * 133 * 24)
        for (ti, frame) in keypoints.enumerated() {
            if ti > 0 { s += "," }
            s += "["
            for (ki, k) in frame.enumerated() {
                precondition(k.x.isFinite && k.y.isFinite, "non-finite keypoint reached the payload")
                if ki > 0 { s += "," }
                s += "[\(fmt(k.x)),\(fmt(k.y))]"
            }
            s += "]"
        }
        s += "],\"scores\":["
        for (ti, frame) in scores.enumerated() {
            if ti > 0 { s += "," }
            s += "[" + frame.map { precondition($0.isFinite); return fmt($0, 5) }.joined(separator: ",") + "]"
        }
        s += "],\"bbox_norm\":[\(bboxNorm.map { fmt($0, 6) }.joined(separator: ","))]"
        s += ",\"source_wh\":[\(sourceWH.0),\(sourceWH.1)],\"crop_xywh\":[\(crop.xywh.map(String.init).joined(separator: ","))]"
        s += ",\"src_fps\":\(fmt(Float(srcFPS), 3)),\"keep_fps\":\(fmt(Float(keepFPS), 3))"
        s += ",\"frames_captured\":\(captured),\"frames_dropped_nonfinite\":\(dropped)}"
        return s.data(using: .utf8)!
    }

    private static func fmt(_ v: Float, _ d: Int = 3) -> String { String(format: "%.\(d)f", v) }

    func translate(_ body: Data) throws -> [String: Any] {
        var comps = URLComponents(url: baseURL.appendingPathComponent("translate_kp"), resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "beams", value: String(beams)),
                            URLQueryItem(name: "max_new_tokens", value: String(maxNewTokens))]
        var req = URLRequest(url: comps.url!, timeoutInterval: 120)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        let sem = DispatchSemaphore(value: 0)
        var result: Result<[String: Any], Error> = .failure(NSError(domain: "SLTEdge", code: 4))
        URLSession.shared.dataTask(with: req) { data, resp, err in
            defer { sem.signal() }
            if let err = err { result = .failure(err); return }
            guard let data = data, let http = resp as? HTTPURLResponse else { return }
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                result = .failure(NSError(domain: "SLTEdge", code: http.statusCode,
                                          userInfo: [NSLocalizedDescriptionKey: "non-JSON reply \(http.statusCode)"]))
                return
            }
            if http.statusCode != 200 {
                result = .failure(NSError(domain: "SLTEdge", code: http.statusCode,
                                          userInfo: [NSLocalizedDescriptionKey: (obj["error"] as? String) ?? "HTTP \(http.statusCode)"]))
                return
            }
            result = .success(obj)
        }.resume()
        sem.wait()
        return try result.get()
    }

    func health() -> String {
        var req = URLRequest(url: baseURL.appendingPathComponent("health"), timeoutInterval: 5)
        req.httpMethod = "GET"
        let sem = DispatchSemaphore(value: 0)
        var out = "unreachable"
        URLSession.shared.dataTask(with: req) { data, _, err in
            defer { sem.signal() }
            if let err = err { out = err.localizedDescription; return }
            if let d = data, let s = String(data: d, encoding: .utf8) { out = s }
        }.resume()
        sem.wait()
        return out
    }
}
