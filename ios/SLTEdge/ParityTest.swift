// On-device parity test: the 299 frames behind results/rtmw_trt_fp32.json, through the phone's own
// preprocess -> ORT -> decode, dumped in the 03_infer_frames.py layout so
// task1_rtmpose/07_kpt_agreement.py scores them against the board FP32 engine unchanged.
//
// This is the gate for the pose stage on the phone. The Mac run of the same ONNX agreed with the
// board to 0.01 px (results/mobile/); whether the A14's CoreML path does is this file's question.
// Also the first honest A14 latency number: ms per frame on the phone, warm, not the M4 Pro's.
import Foundation
import UIKit

struct ParitySummary {
    var nFrames = 0, nNonFinite = 0, warmup = 10
    var runMs: [Double] = [], prepMs: [Double] = [], decodeMs: [Double] = []
    var wallS = 0.0, loadS = 0.0
    var compute = ""
    var outputPath = ""
    var error: String?
}

enum ParityTest {
    static func run(pose: RTMWPose, pp: Preproc, progress: @escaping (Int, Int) -> Void) -> ParitySummary {
        var s = ParitySummary()
        s.compute = pose.compute.rawValue; s.loadS = pose.loadSeconds
        guard let dir = Bundle.main.url(forResource: "test_frames", withExtension: nil),
              let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else {
            s.error = "test_frames not in bundle"; return s
        }
        let frames = names.filter { $0.lowercased().hasSuffix(".jpg") }.sorted()
        var lines: [String] = []
        lines.reserveCapacity(frames.count)
        let t0 = CFAbsoluteTimeGetCurrent()
        for (i, name) in frames.enumerated() {
            guard let img = UIImage(contentsOfFile: dir.appendingPathComponent(name).path)?.cgImage else {
                s.error = "could not decode \(name)"; return s
            }
            let a = CFAbsoluteTimeGetCurrent()
            let f = prepare(cgImage: img, pp: pp)
            let b = CFAbsoluteTimeGetCurrent()
            let r: PoseResult, ms: Double
            do { (r, ms) = try pose.infer(f) } catch { s.error = "\(name): \(error.localizedDescription)"; return s }
            let c = CFAbsoluteTimeGetCurrent()
            if i >= s.warmup { s.prepMs.append((b - a) * 1e3); s.runMs.append(ms); s.decodeMs.append((c - b) * 1e3 - ms) }
            if !r.isFinite { s.nNonFinite += 1 }
            lines.append(jsonFrame(name: name, r: r))
            progress(i + 1, frames.count)
        }
        s.wallS = CFAbsoluteTimeGetCurrent() - t0
        s.nFrames = frames.count

        func stat(_ v: [Double]) -> String {
            guard !v.isEmpty else { return "null" }
            let a = v.sorted()
            let mean = a.reduce(0, +) / Double(a.count)
            return String(format: "{\"mean\":%.3f,\"median\":%.3f,\"p95\":%.3f,\"min\":%.3f,\"max\":%.3f}",
                          mean, a[a.count / 2], a[Int(0.95 * Double(a.count - 1))], a.first!, a.last!)
        }
        let dev = UIDevice.current
        var summary = "{\"n_frames\":\(s.nFrames),\"n_timed\":\(s.runMs.count),\"warmup\":\(s.warmup)"
        summary += String(format: ",\"wall_s\":%.3f,\"n_nonfinite\":%d,\"load_s\":%.2f", s.wallS, s.nNonFinite, s.loadS)
        summary += ",\"provider\":\"\(pose.compute == .noCoreML ? "cpu" : "coreml")\",\"compute_units\":\"\(s.compute)\""
        summary += ",\"device\":\"\(modelIdentifier())\",\"system\":\"\(dev.systemName) \(dev.systemVersion)\""
        summary += ",\"thermal\":\"\(ProcessInfo.processInfo.thermalState.rawValue)\""
        summary += ",\"preprocess_ms\":\(stat(s.prepMs)),\"run_ms\":\(stat(s.runMs)),\"postprocess_ms\":\(stat(s.decodeMs))}"
        let doc = "{\"summary\":\(summary),\"results\":[\(lines.joined(separator: ","))]}"

        let out = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("parity_\(s.compute.lowercased()).json")
        do { try doc.write(to: out, atomically: true, encoding: .utf8); s.outputPath = out.path }
        catch { s.error = "write: \(error.localizedDescription)" }
        return s
    }

    /// One results[] entry. NaN is written as the bare token NaN, which Python's json.load accepts and
    /// 07_kpt_agreement's isfinite masks catch; a null would have become None and crashed np.asarray.
    private static func jsonFrame(name: String, r: PoseResult) -> String {
        func f(_ v: Float, _ d: Int) -> String { v.isFinite ? String(format: "%.\(d)f", v) : "NaN" }
        let k = r.keypoints.map { "[\(f($0.x, 4)),\(f($0.y, 4))]" }.joined(separator: ",")
        let sc = r.scores.map { f($0, 5) }.joined(separator: ",")
        return "{\"frame\":\"\(name)\",\"keypoints\":[\(k)],\"scores\":[\(sc)]}"
    }

    static func modelIdentifier() -> String {
        var sys = utsname(); uname(&sys)
        return withUnsafePointer(to: &sys.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) } }
    }
}
