// Per-sentence record, one JSON line each, in Documents/metrics.jsonl. Pulled off the phone with
// `xcrun devicectl device copy from`. This is the phone's half of the C9 results protocol: every
// number here is a measurement from this run, with the thermal state and battery level logged the
// way DVFS state is logged on the board.
import Foundation

enum Metrics {
    static var url: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("metrics.jsonl")
    }

    static func record(_ o: SentenceOutcome, compute: String) {
        func stat(_ v: [Double]) -> [String: Double] {
            guard !v.isEmpty else { return [:] }
            let a = v.sorted()
            return ["mean": a.reduce(0, +) / Double(a.count), "p50": a[a.count / 2],
                    "p95": a[Int(0.95 * Double(a.count - 1))], "min": a.first!, "max": a.last!]
        }
        var d: [String: Any] = [
            "ts": ISO8601DateFormatter().string(from: Date()),
            "compute": compute,
            "frames_captured": o.framesCaptured,
            "frames_after_thinning": o.framesAfterThinning,
            "frames_dropped_nonfinite": o.framesDroppedNonFinite,
            "src_fps": o.srcFPS, "keep_fps": o.keepFPS,
            "pose_run_ms": stat(o.poseRunMs),
            "pose_stage_ms": o.poseStageMs,
            "thermal": o.thermal,
            "battery_before": o.batteryBefore, "battery_after": o.batteryAfter,
            "bbox_norm": o.bboxNorm,
        ]
        if let c = o.crop { d["crop_xywh"] = c.xywh }
        if let t = o.lmTimingMs { d["lm_timing_ms"] = t; d["tokens"] = o.tokens }
        if let t = o.text { d["text"] = t }
        if let e = o.error { d["error"] = e }
        guard let line = try? JSONSerialization.data(withJSONObject: d) else { return }
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(line); h.write("\n".data(using: .utf8)!); try? h.close()
        } else {
            try? (line + "\n".data(using: .utf8)!).write(to: url)
        }
    }
}
