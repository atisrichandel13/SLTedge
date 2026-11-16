// Keypoints -> the four Uni-Sign input tensors. Ports, in the order the board applies them:
//   demo_server._square_norm_fn  (openasl_square_crop geometry; note the TRUNCATING int conversion,
//                                 which is the authors' and differs from crop_xywh's rounding)
//   pose_to_unisign.subsample    (uniform thinning to max_length = 256)
//   pose_to_unisign.load_part_kp + crop_scale  (verbatim from Uni-Sign datasets.py)
// Checked on the device by the parity test: the board clip's keypoints must yield the same token
// sequence the Python ONNX chain and PyTorch produce.
import Foundation

struct UniSignTensors {
    let T: Int
    let body: [Float]      // [T, 9, 3]
    let left: [Float]      // [T, 21, 3]
    let right: [Float]     // [T, 21, 3]
    let face: [Float]      // [T, 18, 3]
}

enum UniSign {
    static let BODY = [0, 3, 4, 5, 6, 7, 8, 9, 10]
    static let LHAND = Array(91..<112)
    static let RHAND = Array(112..<133)
    static let FACE: [Int] = Array(stride(from: 23, to: 40, by: 2)) + Array(83..<91) + [53]
    static let THR: Float = 0.3
    static let maxLength = 256

    struct Square { let sx: Int, sy: Int, side: Int }

    /// openasl_square_crop(bbox_norm, W, H) -> square_xywh. int() truncation, as in the authors' code.
    static func square(bboxNorm b: [Float], W: Int, H: Int) throws -> Square {
        let x0 = Int(b[0] * Float(W)), x1 = Int(b[2] * Float(W))
        let y0 = Int(b[1] * Float(H)), y1 = Int(b[3] * Float(H))
        let dw = x1 - x0, dh = y1 - y0
        if dw < 16 || dh < 16 { throw PreprocessError.degenerateCrop("square: \(x0),\(y0),\(x1),\(y1) in \(W)x\(H)") }
        let side = max(dw, dh)
        return Square(sx: x0 - (side - dw) / 2, sy: y0 - (side - dh) / 2, side: side)
    }

    /// Crop pixels -> the authors' square frame (§2.5d), the coordinate frame the encoder was trained on.
    static func squareNorm(_ k: [SIMD2<Float>], crop: Crop, sq: Square) -> [SIMD2<Float>] {
        let side = Float(sq.side), cx = Float(crop.x), cy = Float(crop.y), sx = Float(sq.sx), sy = Float(sq.sy)
        return k.map { SIMD2<Float>((cx + $0.x - sx) / side, (cy + $0.y - sy) / side) }
    }

    /// pose_to_unisign.subsample: uniform indices when T exceeds max_length (np.round is half-to-even).
    static func subsampleIdx(_ T: Int) -> [Int] {
        if T <= maxLength { return Array(0..<T) }
        return (0..<maxLength).map { Int((Double(T - 1) * Double($0) / Double(maxLength - 1)).rounded(.toNearestOrEven)) }
    }

    /// load_part_kp over normalised keypoints [T][133] and scores [T][133].
    static func tensors(kpts: [[SIMD2<Float>]], scores: [[Float]]) -> UniSignTensors {
        let T = kpts.count
        // body first: crop_scale gives the scale every other part is divided by
        var body = [Float](repeating: 0, count: T * 9 * 3)
        for t in 0..<T { for (j, i) in BODY.enumerated() {
            let o = (t * 9 + j) * 3; body[o] = kpts[t][i].x; body[o + 1] = kpts[t][i].y; body[o + 2] = scores[t][i] } }
        let scale = cropScale(&body, T: T, V: 9)

        func part(_ idx: [Int], root: Int) -> [Float] {
            let V = idx.count
            var r = [Float](repeating: 0, count: T * V * 3)
            if scale == 0 { return r }     // `if scale == 0: result = np.zeros(result.shape)`
            for t in 0..<T {
                let rt = kpts[t][idx[root]]
                for (j, i) in idx.enumerated() {
                    let o = (t * V + j) * 3
                    var x = (kpts[t][i].x - rt.x) / scale, y = (kpts[t][i].y - rt.y) / scale
                    let c = scores[t][i]
                    x = min(max(x, -1), 1); y = min(max(y, -1), 1)     // np.clip(result, -1, 1)
                    if c <= THR { r[o] = 0; r[o + 1] = 0; r[o + 2] = 0 } else { r[o] = x; r[o + 1] = y; r[o + 2] = min(c, 1) }
                }
            }
            return r
        }
        return UniSignTensors(T: T, body: body, left: part(LHAND, root: 0), right: part(RHAND, root: 0),
                              face: part(FACE, root: FACE.count - 1))
    }

    /// Uni-Sign crop_scale on a [T,V,3] buffer, in place. Returns the scale (0 when fewer than 4
    /// confident joints, in which case the buffer is zeroed).
    private static func cropScale(_ m: inout [Float], T: Int, V: Int) -> Float {
        var xmin = Float.greatestFiniteMagnitude, xmax = -Float.greatestFiniteMagnitude
        var ymin = Float.greatestFiniteMagnitude, ymax = -Float.greatestFiniteMagnitude
        var n = 0
        for i in 0..<(T * V) where m[i * 3 + 2] > THR {
            n += 1
            xmin = min(xmin, m[i * 3]); xmax = max(xmax, m[i * 3]); ymin = min(ymin, m[i * 3 + 1]); ymax = max(ymax, m[i * 3 + 1])
        }
        if n < 4 { for i in m.indices { m[i] = 0 }; return 0 }
        let scale = max(xmax - xmin, ymax - ymin)
        if scale == 0 { for i in m.indices { m[i] = 0 }; return 0 }
        let xs = (xmin + xmax - scale) / 2, ys = (ymin + ymax - scale) / 2
        for i in 0..<(T * V) {
            let o = i * 3
            var x = ((m[o] - xs) / scale - 0.5) * 2, y = ((m[o + 1] - ys) / scale - 0.5) * 2
            x = min(max(x, -1), 1); y = min(max(y, -1), 1)
            if m[o + 2] <= THR { m[o] = 0; m[o + 1] = 0; m[o + 2] = 0 } else { m[o] = x; m[o + 1] = y; m[o + 2] = min(m[o + 2], 1) }
        }
        return scale
    }

    /// Everything from crop-pixel keypoints to tensors, as the board does it after its pose loop.
    static func build(kpts: [[SIMD2<Float>]], scores: [[Float]], bboxNorm: [Float], crop: Crop,
                      sourceWH: (Int, Int)) throws -> UniSignTensors {
        let sq = try square(bboxNorm: bboxNorm, W: sourceWH.0, H: sourceWH.1)
        let idx = subsampleIdx(kpts.count)
        let k = idx.map { squareNorm(kpts[$0], crop: crop, sq: sq) }
        let s = idx.map { scores[$0] }
        return tensors(kpts: k, scores: s)
    }
}
