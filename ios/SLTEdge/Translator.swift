// The translator on the phone: front.onnx (ST-GCN + prefix) -> encoder.onnx -> decoder_init.onnx
// once -> decoder_step.onnx per token. Beam 4 by default -- the configuration every reported number
// uses (greedy measured 20.66 vs beam-4 22.85 BLEU-4 on the 976-clip test split); greedy kept for
// comparison. The same four graphs ran on the Mac token-identically to PyTorch
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

    func translate(_ t: UniSignTensors, beams: Int = 4) throws -> Output {
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
        let (tokens, steps) = beams <= 1 ? try greedy(h: h, mask: mask) : try beamSearch(h: h, L: L, mask: mask, beams: beams)
        tm["decoder"] = (CFAbsoluteTimeGetCurrent() - t0) * 1e3
        tm["decoder_steps"] = Double(steps)
        tm["beams"] = Double(beams)
        tm["total"] = tm["front"]! + tm["encoder"]! + tm["decoder"]!
        return Output(text: detokenize(tokens), tokens: tokens, timingMs: tm, encoderLen: L)
    }

    private func greedy(h: ORTValue, mask: ORTValue) throws -> ([Int], Int) {
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
        return (tokens, steps)
    }

    // ------------------------------------------------------------------ beam search
    // A port of task3_mt5_onnx/05_beam_onnx.py, which reproduces transformers 4.57 _beam_search with
    // the mT5 defaults (length_penalty 1.0, early_stopping False) and is token-identical to
    // PoseOnlyUniSign.translate(num_beams=4) on 30/30 test clips. Same variable names, same order.

    private static let NEG: Float = -1.0e9

    private func shape(_ v: ORTValue) throws -> [Int] { try v.tensorTypeAndShapeInfo().shape.map { $0.intValue } }

    /// [1, ...] -> [n, ...] by repeating the single row.
    private func repeatBatch(_ v: ORTValue, _ n: Int) throws -> ORTValue {
        var s = try shape(v); let d = try v.tensorData() as Data
        var out = Data(capacity: d.count * n)
        for _ in 0..<n { out.append(d) }
        s[0] = n
        return try ORTValue(tensorData: NSMutableData(data: out), elementType: .float, shape: s.map { NSNumber(value: $0) })
    }

    /// [B, ...] -> [idx.count, ...], row i of the output = row idx[i] of the input.
    private func gatherBatch(_ v: ORTValue, _ idx: [Int]) throws -> ORTValue {
        let s = try shape(v); let d = try v.tensorData() as Data
        let row = d.count / s[0]
        var out = Data(capacity: row * idx.count)
        d.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: UInt8.self).baseAddress!
            for i in idx { out.append(base + i * row, count: row) }
        }
        var s2 = s; s2[0] = idx.count
        return try ORTValue(tensorData: NSMutableData(data: out), elementType: .float, shape: s2.map { NSNumber(value: $0) })
    }

    private static func logSoftmax(_ x: ArraySlice<Float>) -> [Float] {
        let m = x.max()!
        var s: Float = 0
        for v in x { s += expf(v - m) }
        let lse = m + logf(s)
        return x.map { $0 - lse }
    }

    /// Indices of `vals` sorted by value descending; ties keep the lower index first (np.argsort stable on -x).
    private static func argsortDesc(_ vals: [Float]) -> [Int] {
        vals.indices.sorted { vals[$0] > vals[$1] || (vals[$0] == vals[$1] && $0 < $1) }
    }

    private func beamSearch(h: ORTValue, L: Int, mask: ORTValue, beams: Int) throws -> ([Int], Int) {
        let maxLength = Self.maxNewTokens + 1
        let K = 2 * beams                                  // max(2, 1 + n_eos) * beams, one EOS id
        let V = vocab.count
        let o = try dinit.run(withInputs: ["input_ids": i64([Int64(Self.start)], [1, 1]),
                                           "encoder_hidden_states": h, "encoder_attention_mask": mask],
                              outputNames: Set(initOut), runOptions: nil)
        var past: [String: ORTValue] = [:]
        for (n, v) in o where n.hasPrefix("present.") {
            past["past_key_values." + n.dropFirst("present.".count)] = try repeatBatch(v, beams)
        }
        let maskb = try i64([Int64](repeating: 1, count: L * beams), [beams, L])
        let l0 = try floats(o["logits"]!)
        var logits = [Float](); logits.reserveCapacity(beams * V)
        for _ in 0..<beams { logits.append(contentsOf: l0[(l0.count - V)...]) }   // [beams, V]

        var runSeq = [[Int]](repeating: [], count: beams)
        var runScore = [Float](repeating: Self.NEG, count: beams); runScore[0] = 0
        var finSeq = [[Int]](repeating: [], count: beams)
        var finScore = [Float](repeating: Self.NEG, count: beams)
        var finDone = [Bool](repeating: false, count: beams)
        var heurUnsat = true
        var curLen = 1
        var steps = 0
        while true {
            // top-K over beams x vocab of (running score + log_softmax); ties -> lower flat index first
            var topV = [Float](), topI = [Int]()
            for b in 0..<beams {
                let lp = Self.logSoftmax(logits[(b * V)..<((b + 1) * V)])
                let rs = runScore[b]
                for t in 0..<V {
                    let v = lp[t] + rs
                    if topV.count < K || v > topV[K - 1] {
                        var pos = topV.count
                        while pos > 0 && v > topV[pos - 1] { pos -= 1 }
                        topV.insert(v, at: pos); topI.insert(b * V + t, at: pos)
                        if topV.count > K { topV.removeLast(); topI.removeLast() }
                    }
                }
            }
            let topBeam = topI.map { $0 / V }, topTok = topI.map { $0 % V }
            let topSeq = zip(topBeam, topTok).map { runSeq[$0] + [$1] }
            let hits = topTok.map { $0 == Self.eos || curLen + 1 >= maxLength }

            // e. next running beams: best `beams` candidates that did not finish
            let rLp = zip(topV, hits).map { $0 + ($1 ? Self.NEG : 0) }
            let nxt = Array(Self.argsortDesc(rLp).prefix(beams))
            let newRunSeq = nxt.map { topSeq[$0] }, newRunScore = nxt.map { rLp[$0] }, parent = nxt.map { topBeam[$0] }

            // f. finished beams: an EOS candidate enters only if its rank < beams
            var cand = [Float](repeating: 0, count: K)
            for i in 0..<K {
                let just = hits[i] && i < beams
                var c = topV[i] / Float(curLen)              // (cur_len + 1 - decoder_prompt_len) ** 1.0
                if !heurUnsat { c += Self.NEG }
                if !just { c += Self.NEG }
                cand[i] = c
            }
            let mSeq = finSeq + topSeq, mScore = finScore + cand
            let mDone = finDone + (0..<K).map { hits[$0] && $0 < beams }
            let keep = Array(Self.argsortDesc(mScore).prefix(beams))
            finSeq = keep.map { mSeq[$0] }; finScore = keep.map { mScore[$0] }; finDone = keep.map { mDone[$0] }

            runSeq = newRunSeq; runScore = newRunScore
            curLen += 1
            let worstFin = finScore.min()!
            let bestPossible = runScore[0] / Float(curLen - 1)
            heurUnsat = heurUnsat && (0..<beams).contains { bestPossible > (finDone[$0] ? worstFin : Self.NEG) }
            if !heurUnsat || hits.allSatisfy({ $0 }) { break }

            // reorder the decoder self-attention cache to the surviving parents; encoder K/V rows are identical
            for (n, v) in past where n.contains(".decoder.") { past[n] = try gatherBatch(v, parent) }
            var feed = past
            feed["input_ids"] = try i64(runSeq.map { Int64($0.last!) }, [beams, 1])
            feed["encoder_attention_mask"] = maskb
            let r = try dstep.run(withInputs: feed, outputNames: Set(stepOut), runOptions: nil)
            for (n, v) in r where n.hasPrefix("present.") { past["past_key_values." + n.dropFirst("present.".count)] = v }
            let lg = try floats(r["logits"]!)                // [beams, 1, V]
            logits = lg
            steps += 1
        }
        return (finSeq[0], steps)
    }

    /// SentencePiece decode for the kept pieces: drop pad/eos/unk, concatenate, "\u{2581}" -> space.
    func detokenize(_ ids: [Int]) -> String {
        ids.filter { $0 > 2 && $0 < vocab.count }.map { vocab[$0] }.joined()
            .replacingOccurrences(of: "\u{2581}", with: " ").trimmingCharacters(in: .whitespaces)
    }
}
