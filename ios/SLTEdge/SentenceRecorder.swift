// Hold-to-record. The button press is the sentence boundary (WORKSPLIT P12 was dropped for exactly
// this), the box is frozen at the press so one box serves the whole clip as on the board, frames are
// warped on arrival and kept as uint8 (147 KB each), and on release the clip is thinned to the
// operating rate from the camera's OWN timestamps, run through the pose model, guarded, and sent.
//
// Order of operations on release mirrors demo_server.Pipeline.translate: thin -> pose per frame ->
// drop non-finite -> hand the keypoints on. The one difference is who runs the pose stage.
import Foundation
import CoreMedia
import CoreVideo
import UIKit

struct SentenceOutcome {
    var text: String?
    var framesCaptured = 0
    var framesAfterThinning = 0
    var framesDroppedNonFinite = 0
    var srcFPS: Double = 0
    var keepFPS: Double = 0
    var poseRunMs: [Double] = []      // session.run per kept frame
    var poseStageMs: Double = 0       // whole pose loop incl. normalise + decode
    var payloadBytes = 0
    var networkMs: Double = 0
    var serverTiming: [String: Any]?
    var thermal: String = ""
    var batteryBefore: Float = -1, batteryAfter: Float = -1
    var error: String?
    var crop: Crop?
    var bboxNorm: [Float] = []
}

/// common/subsample_pkl.py:keep_idx. Python's round() and np.round are half-to-even.
func keepIdx(n: Int, keepFPS: Double, srcFPS: Double) -> [Int] {
    if keepFPS >= srcFPS { return Array(0..<n) }
    let m = max(1, Int((Double(n) * keepFPS / srcFPS).rounded(.toNearestOrEven)))
    var set = Set<Int>()
    for i in 0..<m {
        let v = m == 1 ? 0.0 : Double(n - 1) * Double(i) / Double(m - 1)
        set.insert(Int(v.rounded(.toNearestOrEven)))
    }
    return set.sorted()
}

final class SentenceRecorder {
    private(set) var isRecording = false
    private var frames: [PreparedFrame] = []
    private var stamps: [Double] = []
    private var crop: Crop?
    private var bboxNorm: [Float] = []
    private var sourceWH: (Int, Int) = (0, 0)
    private let lock = NSLock()
    let maxFrames = 900   // ~37 s at 24 fps, demo_server's own cap

    func start(bboxNorm: [Float], frameW: Int, frameH: Int, pp: Preproc) throws {
        let c = try cropXYWH(bboxNorm: bboxNorm, W: frameW, H: frameH)
        lock.lock(); defer { lock.unlock() }
        frames.removeAll(keepingCapacity: true); stamps.removeAll(keepingCapacity: true)
        crop = c; self.bboxNorm = bboxNorm; sourceWH = (frameW, frameH)
        isRecording = true
    }

    /// Called on the camera queue for every frame while recording.
    func ingest(_ pb: CVPixelBuffer, at t: CMTime, pp: Preproc) {
        lock.lock(); defer { lock.unlock() }
        guard isRecording, let c = crop, frames.count < maxFrames else { return }
        frames.append(prepare(pixelBuffer: pb, crop: c, pp: pp))
        stamps.append(CMTimeGetSeconds(t))
    }

    /// Stops, thins, runs the pose model, and returns everything the board needs plus the phone's
    /// own measurements. Runs on the caller's queue; the pose loop is sequential like the board's.
    func finish(pose: RTMWPose, keepFPS: Double, client: BoardClient?, pp: Preproc) -> SentenceOutcome {
        lock.lock()
        isRecording = false
        let fr = frames, ts = stamps, c = crop, bb = bboxNorm, wh = sourceWH
        frames = []; stamps = []
        lock.unlock()

        var o = SentenceOutcome()
        o.crop = c; o.bboxNorm = bb; o.keepFPS = keepFPS
        o.framesCaptured = fr.count
        o.thermal = thermalName()
        UIDevice.current.isBatteryMonitoringEnabled = true
        o.batteryBefore = UIDevice.current.batteryLevel
        guard fr.count >= 2, let crop = c else { o.error = "too few frames (\(fr.count))"; return o }

        // The clip's own rate, from its timestamps -- never an assumed 29.97 (RESULTS.md 2.9B).
        o.srcFPS = Double(fr.count - 1) / (ts.last! - ts.first!)
        let idx = keepIdx(n: fr.count, keepFPS: keepFPS, srcFPS: o.srcFPS)
        o.framesAfterThinning = idx.count

        var kps: [[SIMD2<Float>]] = [], scs: [[Float]] = []
        let t0 = CFAbsoluteTimeGetCurrent()
        for i in idx {
            do {
                let (r, ms) = try pose.infer(fr[i])
                o.poseRunMs.append(ms)
                if !r.isFinite { o.framesDroppedNonFinite += 1; continue }   // L28: never translate these
                kps.append(r.keypoints); scs.append(r.scores)
            } catch { o.error = "pose: \(error.localizedDescription)"; return o }
        }
        o.poseStageMs = (CFAbsoluteTimeGetCurrent() - t0) * 1e3
        if kps.isEmpty { o.error = "every frame failed the finite check -- no person in the box?"; return o }

        let payload = BoardClient.payload(keypoints: kps, scores: scs, bboxNorm: bb, sourceWH: wh,
                                          crop: crop, srcFPS: o.srcFPS, keepFPS: keepFPS,
                                          captured: fr.count, dropped: o.framesDroppedNonFinite)
        o.payloadBytes = payload.count
        if let client = client {
            let t1 = CFAbsoluteTimeGetCurrent()
            do {
                let reply = try client.translate(payload)
                o.networkMs = (CFAbsoluteTimeGetCurrent() - t1) * 1e3
                o.text = reply["text"] as? String
                o.serverTiming = reply["timing_ms"] as? [String: Any]
            } catch { o.error = "board: \(error.localizedDescription)" }
        }
        o.batteryAfter = UIDevice.current.batteryLevel
        return o
    }

    private func thermalName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"; case .fair: return "fair"
        case .serious: return "serious"; case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
