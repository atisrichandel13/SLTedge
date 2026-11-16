// Glue between the UI and the pipeline pieces. AppModel is main-actor state for the view; FramePump
// is the camera-queue side (detector, recorder) and never touches the UI except to post a box.
import Foundation
import SwiftUI
import CoreMedia
import AVFoundation

/// Timestamped log, to stdout (devicectl --console) and to the screen, so the app's state is
/// visible from the Mac and from the phone without guessing.
final class LogStore: ObservableObject {
    static let shared = LogStore()
    @Published var lines: [String] = []
    private let t0 = CFAbsoluteTimeGetCurrent()
    func add(_ s: String) {
        let line = String(format: "%7.1fs  %@", CFAbsoluteTimeGetCurrent() - t0, s)
        print("[sltedge] \(line)")
        DispatchQueue.main.async { self.lines.append(line); if self.lines.count > 40 { self.lines.removeFirst() } }
    }
}
func log(_ s: String) { LogStore.shared.add(s) }

/// Runs on the camera queue. Warping a frame on the main actor at 30 fps would stall the UI and
/// drop frames, so the recorder is fed here and only the detector's box is posted back.
final class FramePump {
    let recorder = SentenceRecorder()
    private let detector = PersonBox()
    let pp: Preproc
    private var counter = 0
    var onBox: (([Float]?) -> Void)?

    init(pp: Preproc) { self.pp = pp }

    func handle(_ pb: CVPixelBuffer, _ t: CMTime) {
        if recorder.isRecording {
            recorder.ingest(pb, at: t, pp: pp)
            return
        }
        counter += 1
        if counter % 6 == 0 {     // ~5 Hz detector while idle; frozen at the press
            let box = detector.detect(pb)
            onBox?(box)
        }
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var status = "starting"
    @Published var translation = ""
    @Published var lastOutcome: SentenceOutcome?
    @Published var liveBox: [Float]? = nil          // normalised, top-left origin
    @Published var isRecording = false
    @Published var busy = false
    @Published var compute: Compute = .all { didSet { reloadPose() } }
    @Published var boardHost: String = UserDefaults.standard.string(forKey: "boardHost") ?? "192.168.1.73:8765" {
        didSet { UserDefaults.standard.set(boardHost, forKey: "boardHost") }
    }
    @Published var keepFPS: Double = 24
    @Published var parity: ParitySummary?
    @Published var parityProgress = ""

    let camera = CameraController()
    private var pump: FramePump!
    private var pose: RTMWPose?
    private let work = DispatchQueue(label: "sltedge.work", qos: .userInitiated)

    init() {
        let pp: Preproc
        do {
            pp = try Preproc.loadFromBundle()
            try camera.configure()
        } catch { status = "setup failed: \(error.localizedDescription)"; log("setup FAILED: \(error)"); return }
        log("setup: preproc \(pp.width)x\(pp.height), camera configured (\(CameraController.position == .back ? "back" : "front"))")
        pump = FramePump(pp: pp)
        pump.onBox = { [weak self] box in DispatchQueue.main.async { self?.liveBox = box } }
        let pump = self.pump!
        camera.onFrame = { pb, t in pump.handle(pb, t) }
        reloadPose()
        AVCaptureDevice.requestAccess(for: .video) { ok in
            DispatchQueue.main.async {
                if ok { self.camera.start() } else { self.status = "camera permission denied" }
            }
        }
    }

    private func reloadPose() {
        guard let pump = pump else { return }
        let c = compute, pp = pump.pp
        busy = true; status = "loading pose model (\(c.rawValue))"
        log("pose: loading ONNX via CoreML EP, compute=\(c.rawValue); first time includes the CoreML compile")
        work.async {
            do {
                let p = try RTMWPose(compute: c, pp: pp)
                log(String(format: "pose: session created in %.1fs; warming up", p.loadSeconds))
                // Warm up as the board does, so the first sentence does not pay first-call cost.
                let blank = PreparedFrame(chw: [UInt8](repeating: 0, count: 3 * pp.width * pp.height),
                                          center: .zero, scale: SIMD2<Float>(1, 1), crop: Crop(x: 0, y: 0, w: 2, h: 2))
                var ms: [Double] = []
                for _ in 0..<5 { ms.append(try p.infer(blank).1) }
                log("pose: warm-up run ms " + ms.map { String(format: "%.1f", $0) }.joined(separator: " "))
                DispatchQueue.main.async {
                    self.pose = p; self.busy = false
                    self.status = String(format: "ready (%@, load %.1fs)", c.rawValue, p.loadSeconds)
                }
            } catch {
                log("pose: LOAD FAILED: \(error)")
                DispatchQueue.main.async { self.busy = false; self.status = "pose load failed: \(error.localizedDescription)" }
            }
        }
    }

    func pressBegan() {
        guard !busy, pose != nil, !isRecording, let pump = pump else { return }
        guard let box = liveBox else { status = "no person box -- stand in frame"; return }
        let (w, h) = camera.frameSize
        do {
            try pump.recorder.start(bboxNorm: box, frameW: w, frameH: h, pp: pump.pp)
            isRecording = true; status = "recording"; translation = ""
        } catch { status = "box: \(error.localizedDescription)" }
    }

    func pressEnded() {
        guard isRecording, let pose = pose, let pump = pump else { return }
        isRecording = false; busy = true; status = "pose on phone..."
        let client = boardHost.isEmpty ? nil : URL(string: "http://\(boardHost)").map(BoardClient.init)
        let keep = keepFPS, comp = compute.rawValue
        work.async {
            let o = pump.recorder.finish(pose: pose, keepFPS: keep, client: client, pp: pump.pp)
            Metrics.record(o, compute: comp)
            DispatchQueue.main.async {
                self.lastOutcome = o; self.busy = false
                self.translation = o.text ?? ""
                if let e = o.error { self.status = e }
                else {
                    let p50 = o.poseRunMs.sorted()[o.poseRunMs.count / 2]
                    self.status = String(format: "%d/%d frames, pose %.0f ms (%.1f ms/frame p50), %@ KB, net %.0f ms, %@",
                                         o.framesAfterThinning, o.framesCaptured, o.poseStageMs, p50,
                                         String(o.payloadBytes / 1024), o.networkMs, o.thermal)
                }
            }
        }
    }

    func runParity() {
        guard let pump = pump else { log("parity: tapped but setup failed earlier"); return }
        guard !busy else { log("parity: tapped while busy (\(status)) -- ignored"); return }
        guard let pose = pose else { log("parity: tapped but no pose model is loaded"); return }
        busy = true; parity = nil; status = "parity test running"
        log("parity: starting 299 frames on \(pose.compute.rawValue)")
        let pp = pump.pp
        work.async {
            let s = ParityTest.run(pose: pose, pp: pp) { i, n in
                if i % 50 == 0 || i == n { log("parity: \(i)/\(n)") }
                DispatchQueue.main.async { self.parityProgress = "\(i)/\(n)" }
            }
            DispatchQueue.main.async {
                self.parity = s; self.busy = false
                if let e = s.error { self.status = "parity: \(e)"; log("parity: FAILED: \(e)") }
                else {
                    let a = s.runMs.sorted()
                    self.status = String(format: "parity done: %d frames, run p50 %.1f ms, nonfinite %d -> %@",
                                         s.nFrames, a[a.count / 2], s.nNonFinite, (s.outputPath as NSString).lastPathComponent)
                    log(self.status)
                }
            }
        }
    }

    func checkBoard() {
        guard let url = URL(string: "http://\(boardHost)"), let host = url.host else { return }
        let c = BoardClient(baseURL: url)
        status = "checking board..."
        log("reach: TCP \(host):22 (route check, independent of the server)")
        work.async {
            let r = tcpReach(host: host, port: 22)
            log("reach: \(host):22 -> \(r)")
            let h = c.health()
            log("health: \(url.absoluteString)/health -> \(h.prefix(200))")
            DispatchQueue.main.async {
                self.status = "route \(r.hasPrefix("ready") || r.contains("refused") ? "OK" : "NO (\(r))"); server: " + h.prefix(80)
            }
        }
    }
}
