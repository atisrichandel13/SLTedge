// The translator on the phone: front.onnx (ST-GCN + prefix) -> encoder.onnx -> decoder_init.onnx
// once -> decoder_step.onnx per token, greedy, KV cache passed through as ORTValues without
// copying. The same four graphs ran on the Mac token-identically to PyTorch
// (task3_mt5_onnx/02_export_manual.py, 04_export_front.py). FP32 on the CPU execution provider:
// mT5 in FP16 saturates (RESULTS.md L8), which rules out the Neural Engine for this half.
//
// The graphs live in Documents/models/ (1.46 GB), pushed with `devicectl device copy to`, not in
// the bundle, so a rebuild does not re-send them.
import Foundation
import OnnxRuntimeBindings

final class Translator {
    private let env: ORTEnv
    private let front: ORTSession, enc: ORTSession, dinit: ORTSession, dstep: ORTSession
    private let initOut: [String], stepOut: [String]
    private let vocab: [String]
    let loadSeconds: Double
    static let eos = 1, start = 0, maxNewTokens = 64

    static var modelsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("models")
    }
    static var available: Bool {
        ["front", "encoder", "decoder_init", "decoder_step"].allSatisfy {
            FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent("\($0).onnx").path) }
    }

    init() throws {
        let t0 = CFAbsoluteTimeGetCurrent()
        let e = try ORTEnv(loggingLevel: .warning)
        env = e
        front = try Self.session(e, "front"); enc = try Self.session(e, "encoder")
        dinit = try Self.session(e, "decoder_init"); dstep = try Self.session(e, "decoder_step")
        initOut = try dinit.outputNames(); stepOut = try dstep.outputNames()
        guard let vurl = Bundle.main.url(forResource: "pruned_vocab", withExtension: "json"),
              let v = try JSONSerialization.jsonObject(with: Data(contentsOf: vurl)) as? [String: Any],
              let pieces = v["pieces"] as? [String] else {
            throw NSError(domain: "SLTEdge", code: 5, userInfo: [NSLocalizedDescriptionKey: "pruned_vocab.json missing"])
        }
        vocab = pieces
        loadSeconds = CFAbsoluteTimeGetCurrent() - t0
    }

    private static func session(_ env: ORTEnv, _ name: String) throws -> ORTSession {
        let o = try ORTSessionOptions()
        try o.setIntraOpNumThreads(4)
        try o.setGraphOptimizationLevel(.all)
        return try ORTSession(env: env, modelPath: modelsDir.appendingPathComponent("\(name).onnx").path, sessionOptions: o)
    }

    private func f32(_ a: [Float], _ shape: [Int]) throws -> ORTValue {
        try ORTValue(tensorData: NSMutableData(bytes: a, length: a.count * 4), elementType: .float,
                     shape: shape.map { NSNumber(value: $0) })
    }
    private func i64(_ a: [Int64], _ shape: [Int]) throws -> ORTValue {
        try ORTValue(tensorData: NSMutableData(bytes: a, length: a.count * 8), elementType: .int64,
                     shape: shape.map { NSNumber(value: $0) })
    }
    private func floats(_ v: ORTValue) throws -> [Float] {
        (try v.tensorData() as Data).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    struct Output { let text: String; let tokens: [Int]; let timingMs: [String: Double]; let encoderLen: Int }

    func translate(_ t: UniSignTensors) throws -> Output {
        var tm: [String: Double] = [:]
        var t0 = CFAbsoluteTimeGetCurrent()
        let emb = try front.run(withInputs: ["body": f32(t.body, [1, t.T, 9, 3]), "left": f32(t.left, [1, t.T, 21, 3]),
                                             "right": f32(t.right, [1, t.T, 21, 3]), "face_all": f32(t.face, [1, t.T, 18, 3])],
                                outputNames: ["inputs_embeds"], runOptions: nil)["inputs_embeds"]!
        tm["front"] = (CFAbsoluteTimeGetCurrent() - t0) * 1e3
        let L = Int(truncating: try emb.tensorTypeAndShapeInfo().shape[1])
        let mask = try i64([Int64](repeating: 1, count: L), [1, L])

        t0 = CFAbsoluteTimeGetCurrent()
        let h = try enc.run(withInputs: ["inputs_embeds": emb, "attention_mask": mask],
                            outputNames: ["last_hidden_state"], runOptions: nil)["last_hidden_state"]!
        tm["encoder"] = (CFAbsoluteTimeGetCurrent() - t0) * 1e3

        t0 = CFAbsoluteTimeGetCurrent()
        var out = try dinit.run(withInputs: ["input_ids": i64([Int64(Self.start)], [1, 1]),
                                             "encoder_hidden_states": h, "encoder_attention_mask": mask],
                                outputNames: Set(initOut), runOptions: nil)
        var past: [String: ORTValue] = [:]
        for (n, v) in out where n.hasPrefix("present.") { past["past_key_values." + n.dropFirst("present.".count)] = v }
        var logits = try floats(out["logits"]!)
        var tokens: [Int] = []
        var steps = 0
        for _ in 0..<Self.maxNewTokens {
            let V = vocab.count
            let last = Array(logits[(logits.count - V)...])
            var best = 0; var bv = -Float.greatestFiniteMagnitude
            for (i, v) in last.enumerated() where v > bv { bv = v; best = i }
            tokens.append(best)
            if best == Self.eos { break }
            var feed = past
            feed["input_ids"] = try i64([Int64(best)], [1, 1]); feed["encoder_attention_mask"] = mask
            out = try dstep.run(withInputs: feed, outputNames: Set(stepOut), runOptions: nil)
            for (n, v) in out where n.hasPrefix("present.") { past["past_key_values." + n.dropFirst("present.".count)] = v }
            logits = try floats(out["logits"]!)
            steps += 1
        }
        tm["decoder"] = (CFAbsoluteTimeGetCurrent() - t0) * 1e3
        tm["decoder_steps"] = Double(steps)
        tm["total"] = tm["front"]! + tm["encoder"]! + tm["decoder"]!
        return Output(text: detokenize(tokens), tokens: tokens, timingMs: tm, encoderLen: L)
    }

    /// SentencePiece decode for the kept pieces: drop pad/eos/unk, concatenate, "\u{2581}" -> space.
    func detokenize(_ ids: [Int]) -> String {
        ids.filter { $0 > 2 && $0 < vocab.count }.map { vocab[$0] }.joined()
            .replacingOccurrences(of: "\u{2581}", with: " ").trimmingCharacters(in: .whitespaces)
    }
}
