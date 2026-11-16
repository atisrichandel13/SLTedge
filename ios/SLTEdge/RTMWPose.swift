// The pose stage on the phone: ONNX Runtime + CoreML execution provider over the board's own ONNX.
// Decode side is a port of rtmpose_utils.py get_simcc_maximum + postprocess, including the rule
// that a non-finite confidence leaves NaN in the coordinates (the L28 guard).
import Foundation
import OnnxRuntimeBindings

enum Compute: String, CaseIterable, Identifiable {
    // ORT 1.24 validates "ALL" (coreml_options.cc), although its own header comment says "All".
    case all = "ALL", cpuGPU = "CPUAndGPU", cpuNE = "CPUAndNeuralEngine", cpuOnly = "CPUOnly", noCoreML = "ORT CPU"
    var id: String { rawValue }
}

/// Which RTMW graph runs (task1_rtmpose/05d_make_mobile_onnx.py). Both have the batch axis fixed to 1.
///   fp32      exact: 99.99 % of Uni-Sign's keypoints within 1 px of the board FP32 engine on the Mac
///   fp16head  FP16 backbone, FP32 SimCC head, for the Neural Engine: 62.8 % within 1 px, 96.9 % within
///             5 px on the Mac -- below the board FP16 engine (82.7 %) and NOT yet gated on BLEU-4
enum PoseVariant: String, CaseIterable, Identifiable {
    case fp32 = "rtmw-l-m_256x192_b1", fp16head = "rtmw-l-m_256x192_b1_fp16head32"
    var id: String { rawValue }
    var label: String { self == .fp32 ? "FP32" : "FP16 (unvalidated)" }
    var short: String { self == .fp32 ? "fp32" : "fp16head" }
}

struct PoseResult {
    let keypoints: [SIMD2<Float>]   // crop pixels; NaN where the estimator failed
    let scores: [Float]
    var isFinite: Bool {
        for (k, s) in zip(keypoints, scores) where !(k.x.isFinite && k.y.isFinite && s.isFinite) { return false }
        return true
    }
}

final class RTMWPose {
    let pp: Preproc
    let compute: Compute
    let variant: PoseVariant
    private let env: ORTEnv
    private let session: ORTSession
    private let inputName: String
    let loadSeconds: Double

    init(compute: Compute, variant: PoseVariant, pp: Preproc) throws {
        self.pp = pp
        self.compute = compute
        self.variant = variant
        let t0 = CFAbsoluteTimeGetCurrent()
        guard let path = Bundle.main.path(forResource: variant.rawValue, ofType: "onnx") else {
            throw NSError(domain: "SLTEdge", code: 2, userInfo: [NSLocalizedDescriptionKey: "\(variant.rawValue).onnx missing from bundle"])
        }
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(2)
        if compute != .noCoreML {
            // Same keys as the Mac run (05c_infer_frames_ort.py). The flag is a permission, not an
            // instruction: ORT logs how many nodes CoreML actually took. With the dynamic batch axis
            // the phone got 344 of 397 nodes in 6 partitions; the graph now has batch fixed to 1.
            try opts.appendCoreMLExecutionProvider(withOptionsV2: [
                "MLComputeUnits": compute.rawValue,
                "ModelFormat": "MLProgram",
                "RequireStaticInputShapes": "1",
                "EnableOnSubgraphs": "0",
            ])
        }
        session = try ORTSession(env: env, modelPath: path, sessionOptions: opts)
        inputName = try session.inputNames().first ?? "input"
        loadSeconds = CFAbsoluteTimeGetCurrent() - t0
    }

    /// Runs one prepared frame. Returns keypoints in crop pixels, scores, and the pure session.run
    /// time in ms (preprocess and decode are timed by the caller, as on the board).
    func infer(_ f: PreparedFrame) throws -> (PoseResult, Double) {
        let x = normalize(f.chw, pp: pp)
        let data = NSMutableData(bytes: x, length: x.count * MemoryLayout<Float>.size)
        let shape: [NSNumber] = [1, 3, NSNumber(value: pp.height), NSNumber(value: pp.width)]
        let input = try ORTValue(tensorData: data, elementType: .float, shape: shape)
        let t0 = CFAbsoluteTimeGetCurrent()
        let outs = try session.run(withInputs: [inputName: input],
                                   outputNames: ["simcc_x", "simcc_y"], runOptions: nil)
        let runMs = (CFAbsoluteTimeGetCurrent() - t0) * 1e3
        let sx = try floats(outs["simcc_x"]!), sy = try floats(outs["simcc_y"]!)
        return (Self.postprocess(sx: sx, sy: sy, center: f.center, scale: f.scale, pp: pp), runMs)
    }

    private func floats(_ v: ORTValue) throws -> [Float] {
        let d = try v.tensorData() as Data
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// rtmpose_utils.postprocess: argmax per axis, score = min(max_x, max_y), locs = -1 where the
    /// score <= 0, back through the affine into crop pixels, NaN where the score is not finite.
    static func postprocess(sx: [Float], sy: [Float], center: SIMD2<Float>, scale: SIMD2<Float>,
                            pp: Preproc) -> PoseResult {
        let K = pp.num_keypoints
        let wx = sx.count / K, wy = sy.count / K
        var kpts = [SIMD2<Float>](repeating: .zero, count: K)
        var vals = [Float](repeating: 0, count: K)
        let inW = Float(pp.width), inH = Float(pp.height), split = pp.simcc_split_ratio
        for k in 0..<K {
            var mx: Float = -.greatestFiniteMagnitude, ix = 0
            var my: Float = -.greatestFiniteMagnitude, iy = 0
            var nan = false
            for i in 0..<wx { let v = sx[k * wx + i]; if v.isNaN { nan = true }; if v > mx { mx = v; ix = i } }
            for i in 0..<wy { let v = sy[k * wy + i]; if v.isNaN { nan = true }; if v > my { my = v; iy = i } }
            // np.argmax on an all-NaN row is 0 and np.amax is NaN; mirror that so the guard below is
            // the thing that fires, as in the Python.
            if nan && (mx == -.greatestFiniteMagnitude || my == -.greatestFiniteMagnitude) {
                vals[k] = .nan; kpts[k] = SIMD2<Float>(.nan, .nan); continue
            }
            let v = min(mx, my)
            vals[k] = v
            var lx = Float(ix), ly = Float(iy)
            if v <= 0 { lx = -1; ly = -1 }
            let px = lx / split, py = ly / split
            kpts[k] = SIMD2<Float>(px / inW * scale.x + center.x - 0.5 * scale.x,
                                   py / inH * scale.y + center.y - 0.5 * scale.y)
            if !v.isFinite { kpts[k] = SIMD2<Float>(.nan, .nan) }
        }
        return PoseResult(keypoints: kpts, scores: vals)
    }
}
