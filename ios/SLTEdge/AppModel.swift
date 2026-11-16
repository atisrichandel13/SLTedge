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
    @Published var compute: Compute = .all { didSet { if !applyingArgs { reloadPose() } } }
    @Published var variant: PoseVariant = .fp32 { didSet { if !applyingArgs { reloadPose() } } }
    @Published var beams = 4
    private var applyingArgs = false
    /// Launch arguments, so a measurement can be started from the Mac with devicectl and read from the
    /// console without anyone tapping: -autoParity  -poseVariant fp32|fp16head  -compute ALL|CPUAndGPU|...
    private let autoParity = ProcessInfo.processInfo.arguments.contains("-autoParity")
    @Published var translatorState = "translator: not loaded"
    @Published var keepFPS: Double = 24
    @Published var parity: ParitySummary?
    @Published var parityProgress = ""

    let camera = CameraController()
    private var pump: FramePump!
    private var pose: RTMWPose?
    private var translator: Translator?
    private let work = DispatchQueue(label: "sltedge.work", qos: .userInitiated)

    init() {
        let pp: Preproc
        do {
            pp = try Preproc.loadFromBundle()
            try camera.configure()
        } catch { status = "setup failed: \(error.localizedDescription)"; log("setup FAILED: \(error)"); return }
        log("setup: preproc \(pp.width)x\(pp.height), camera configured (\(CameraController.position == .back ? "back" : "front"))")
        applyingArgs = true
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-poseVariant"), i + 1 < args.count {
            variant = PoseVariant.allCases.first { $0.short == args[i + 1] } ?? .fp32 }
        if let i = args.firstIndex(of: "-compute"), i + 1 < args.count {
            compute = Compute(rawValue: args[i + 1]) ?? .all }
        applyingArgs = false
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
        let c = compute, v = variant, pp = pump.pp
        busy = true; status = "loading pose model (\(v.label), \(c.rawValue))"
        log("pose: loading \(v.rawValue) via CoreML EP, compute=\(c.rawValue); first time includes the CoreML compile")
        work.async {
            do {
                let p = try RTMWPose(compute: c, variant: v, pp: pp)
                log(String(format: "pose: session created in %.1fs; warming up", p.loadSeconds))
                let blank = PreparedFrame(chw: [UInt8](repeating: 0, count: 3 * pp.width * pp.height),
                                          center: .zero, scale: SIMD2<Float>(1, 1), crop: Crop(x: 0, y: 0, w: 2, h: 2))
                var ms: [Double] = []
                for _ in 0..<5 { ms.append(try p.infer(blank).1) }
                log("pose: warm-up run ms " + ms.map { String(format: "%.1f", $0) }.joined(separator: " "))
                DispatchQueue.main.async {
                    self.pose = p; self.busy = false
                    self.status = String(format: "ready (%@, %@, load %.1fs)", v.label, c.rawValue, p.loadSeconds)
                    self.loadTranslatorIfPresent()
                }
            } catch {
                log("pose: LOAD FAILED: \(error)")
                DispatchQueue.main.async { self.busy = false; self.status = "pose load failed: \(error.localizedDescription)" }
            }
        }
    }

    /// The translator graphs are pushed to Documents/models by devicectl; load them once if present.
    func loadTranslatorIfPresent() {
        guard translator == nil else { if autoParity { runParity() }; return }
        guard Translator.available else {
            translatorState = "translator: graphs not in Documents/models -- push them with devicectl"
            log(translatorState); return
        }
        translatorState = "translator: loading 1.46 GB of graphs..."
        log("translator: loading front/encoder/decoder_init/decoder_step from Documents/models")
        work.async {
            do {
                let t = try Translator()
                log(String(format: "translator: loaded in %.1fs", t.loadSeconds))
                DispatchQueue.main.async {
                    self.translator = t; self.translatorState = String(format: "translator: ready (%.1fs load)", t.loadSeconds)
                    if self.autoParity { self.runParity() }
                }
            } catch {
                log("translator: LOAD FAILED: \(error)")
                DispatchQueue.main.async { self.translatorState = "translator: load failed: \(error.localizedDescription)" }
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
        let tr = translator
        if tr == nil { log("sentence: the translator is not loaded; nothing will translate") }
        let keep = keepFPS, comp = "\(variant.short)/\(compute.rawValue)", nb = beams
        work.async {
            let o = pump.recorder.finish(pose: pose, keepFPS: keep, translator: tr, beams: nb, pp: pump.pp)
            Metrics.record(o, compute: comp)
            DispatchQueue.main.async {
                self.lastOutcome = o; self.busy = false
                self.translation = o.text ?? ""
                if let e = o.error { self.status = e; log("sentence: \(e)") }
                else {
                    let p50 = o.poseRunMs.sorted()[o.poseRunMs.count / 2]
                    var s = String(format: "%d/%d frames, pose %.0f ms (%.1f ms/frame p50)", o.framesAfterThinning, o.framesCaptured, o.poseStageMs, p50)
                    if let lm = o.lmTimingMs {
                        s += String(format: ", LM %.0f ms (front %.0f, enc %.0f, dec %.0f / %.0f tok)", lm["total"] ?? 0, lm["front"] ?? 0, lm["encoder"] ?? 0, lm["decoder"] ?? 0, lm["decoder_steps"] ?? 0)
                    }
                    s += ", \(o.thermal)"
                    self.status = s; log("sentence: \(s) -> \(o.text ?? "")")
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
        let pp = pump.pp, tr = translator
        // The camera and the person detector compete with the pose model for the same silicon; a
        // latency number is only clean with them stopped.
        camera.stop()
        log("parity: camera stopped for the measurement")
        work.async {
            var s = ParityTest.run(pose: pose, pp: pp) { i, n in
                if i % 50 == 0 || i == n { log("parity: \(i)/\(n)") }
                DispatchQueue.main.async { self.parityProgress = "\(i)/\(n)" }
            }
            if s.error == nil, let tr = tr, let (text, tm, T) = ParityTest.translateClip(results: s.results, translator: tr) {
                let ok = text == ParityTest.expectedBeam4
                log(String(format: "parity translate: T=%d, front %.0f ms, encoder %.0f ms, decoder %.0f ms (%.0f steps) -> %@", T, tm["front"] ?? 0, tm["encoder"] ?? 0, tm["decoder"] ?? 0, tm["decoder_steps"] ?? 0, text))
                log("parity translate: \(ok ? "MATCHES the Mac ONNX/PyTorch beam-4 sentence" : "DIFFERS from the Mac ONNX/PyTorch beam-4 sentence")")
                s.error = nil
            }
            s.results = []
            DispatchQueue.main.async {
                self.parity = s; self.busy = false
                self.camera.start()
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
}
