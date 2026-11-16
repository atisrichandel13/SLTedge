// Port of task1_rtmpose/rtmpose_utils.py (preprocess side) and data/openasl_fetch.py:crop_xywh.
//
// The contract this file keeps, in the order the server applies it (jetson/demo_server.py):
//   1. crop_xywh: the normalised box -> integer crop clamped to the frame, width/height rounded
//      DOWN to even. No square, no pad, no resize (data/clips*/index.json: crop_style=native).
//   2. preprocess(crop, bbox=None): the WHOLE crop is the box -> centre/scale with 1.25 padding ->
//      aspect fixed to the model input -> one warpAffine to 192x256, bilinear, zero border ->
//      BGR->RGB -> (x - mean) / std -> CHW.
// Anything outside the crop is border (0), not the frame pixel that happens to be there, because the
// server warps the already-cropped image. The parity test against results/rtmw_trt_fp32.json is
// what proves this port, not this comment.
import Foundation
import CoreVideo
import CoreGraphics

struct Preproc: Decodable {
    let input_size: [Int]          // [width, height] = [192, 256]
    let simcc_split_ratio: Float
    let mean: [Float]
    let std: [Float]
    let bgr_to_rgb: Bool
    let bbox_padding: Float?
    let num_keypoints: Int

    var width: Int { input_size[0] }
    var height: Int { input_size[1] }
    var padding: Float { bbox_padding ?? 1.25 }

    static func loadFromBundle() throws -> Preproc {
        guard let url = Bundle.main.url(forResource: "preproc", withExtension: "json") else {
            throw NSError(domain: "SLTEdge", code: 1, userInfo: [NSLocalizedDescriptionKey: "preproc.json missing from bundle"])
        }
        return try JSONDecoder().decode(Preproc.self, from: Data(contentsOf: url))
    }
}

struct Crop: Equatable {
    let x: Int, y: Int, w: Int, h: Int
    var xywh: [Int] { [x, y, w, h] }
}

enum PreprocessError: Error { case degenerateCrop(String) }

/// data/openasl_fetch.py:crop_xywh, verbatim semantics. The box may fall outside 0..1 (OpenASL's do).
func cropXYWH(bboxNorm: [Float], W: Int, H: Int) throws -> Crop {
    var x0 = Int((bboxNorm[0] * Float(W)).rounded(.toNearestOrEven))
    var x1 = Int((bboxNorm[2] * Float(W)).rounded(.toNearestOrEven))
    var y0 = Int((bboxNorm[1] * Float(H)).rounded(.toNearestOrEven))
    var y1 = Int((bboxNorm[3] * Float(H)).rounded(.toNearestOrEven))
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if x1 - x0 < 16 || y1 - y0 < 16 {
        throw PreprocessError.degenerateCrop("\(x0),\(y0),\(x1),\(y1) in \(W)x\(H)")
    }
    return Crop(x: x0, y: y0, w: (x1 - x0) & ~1, h: (y1 - y0) & ~1)
}

/// bbox_xyxy2cs + fix_aspect_ratio for the whole-crop box. Returns (center, scale) in crop pixels.
func centerScale(crop: Crop, pp: Preproc) -> (SIMD2<Float>, SIMD2<Float>) {
    let w = Float(crop.w), h = Float(crop.h)
    let center = SIMD2<Float>(w * 0.5, h * 0.5)
    var scale = SIMD2<Float>(w, h) * pp.padding
    let aspect = Float(pp.width) / Float(pp.height)
    if scale.x > scale.y * aspect {
        scale = SIMD2<Float>(scale.x, scale.x / aspect)
    } else {
        scale = SIMD2<Float>(scale.y * aspect, scale.y)
    }
    return (center, scale)
}

/// get_warp_matrix with rot = 0, shift = 0, solved in closed form: cv2.getAffineTransform on those
/// three point pairs is a uniform scale s = dst_w / src_w about the box centre, so
/// M = [[s, 0, tx], [0, s, ty]] with tx = dst_w/2 - s*cx, ty = dst_h/2 - s*cy.
struct Affine {
    let s: Float, tx: Float, ty: Float
    init(center: SIMD2<Float>, scale: SIMD2<Float>, outW: Int, outH: Int) {
        s = Float(outW) / scale.x
        tx = Float(outW) * 0.5 - s * center.x
        ty = Float(outH) * 0.5 - s * center.y
    }
}

/// cv2.warpAffine(crop, M, (w, h), INTER_LINEAR) with BORDER_CONSTANT 0, reading the crop window out
/// of a BGRA frame. Output is uint8, CHW, RGB when bgr_to_rgb (the channel order the model expects),
/// i.e. the array `crop[..., ::-1]` in the Python before the float conversion.
func warpCrop(base: UnsafePointer<UInt8>, stride: Int, crop: Crop,
              affine a: Affine, pp: Preproc) -> [UInt8] {
    let outW = pp.width, outH = pp.height, plane = outW * outH
    var out = [UInt8](repeating: 0, count: 3 * plane)
    let cw = crop.w, ch = crop.h, cx = crop.x, cy = crop.y
    let invS = 1.0 / a.s
    out.withUnsafeMutableBufferPointer { o in
        for y in 0..<outH {
            let ys = (Float(y) - a.ty) * invS
            let y0 = Int(ys.rounded(.down)); let fy = ys - Float(y0)
            let y0in = y0 >= 0 && y0 < ch, y1in = y0 + 1 >= 0 && y0 + 1 < ch
            let row0 = (cy + y0) * stride, row1 = (cy + y0 + 1) * stride
            for x in 0..<outW {
                let xs = (Float(x) - a.tx) * invS
                let x0 = Int(xs.rounded(.down)); let fx = xs - Float(x0)
                let x0in = x0 >= 0 && x0 < cw, x1in = x0 + 1 >= 0 && x0 + 1 < cw
                let c00 = (cx + x0) * 4, c01 = (cx + x0 + 1) * 4
                let w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy
                for c in 0..<3 {                      // c indexes B, G, R in the BGRA source
                    var v: Float = 0
                    if y0in {
                        if x0in { v += w00 * Float(base[row0 + c00 + c]) }
                        if x1in { v += w01 * Float(base[row0 + c01 + c]) }
                    }
                    if y1in {
                        if x0in { v += w10 * Float(base[row1 + c00 + c]) }
                        if x1in { v += w11 * Float(base[row1 + c01 + c]) }
                    }
                    let ch_ = pp.bgr_to_rgb ? (2 - c) : c   // output channel 0 = R
                    o[ch_ * plane + y * outW + x] = UInt8(clamping: Int((v + 0.5).rounded(.down)))
                }
            }
        }
    }
    return out
}

/// (x - mean) / std per channel, CHW float32, the tensor the ONNX input takes.
func normalize(_ chw: [UInt8], pp: Preproc) -> [Float] {
    let plane = pp.width * pp.height
    var x = [Float](repeating: 0, count: 3 * plane)
    for c in 0..<3 {
        let m = pp.mean[c], inv = 1 / pp.std[c]
        let off = c * plane
        for i in 0..<plane { x[off + i] = (Float(chw[off + i]) - m) * inv }
    }
    return x
}

/// One frame, end to end: BGRA pixel buffer + normalised box -> model tensor and the affine context
/// postprocess needs to map keypoints back into crop pixels.
struct PreparedFrame {
    let chw: [UInt8]                 // the uint8 warp, kept so recording does not hold float tensors
    let center: SIMD2<Float>
    let scale: SIMD2<Float>
    let crop: Crop
}

func prepare(pixelBuffer pb: CVPixelBuffer, crop: Crop, pp: Preproc) -> PreparedFrame {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let (center, scale) = centerScale(crop: crop, pp: pp)
    let a = Affine(center: center, scale: scale, outW: pp.width, outH: pp.height)
    let chw = warpCrop(base: base, stride: stride, crop: crop, affine: a, pp: pp)
    return PreparedFrame(chw: chw, center: center, scale: scale, crop: crop)
}

/// A decoded still (the parity frames) rendered into the same BGRA layout the camera delivers, so
/// one warp routine serves both and the parity test exercises the camera path's code.
func prepare(cgImage: CGImage, pp: Preproc) -> PreparedFrame {
    let W = cgImage.width, H = cgImage.height
    let stride = W * 4
    var buf = [UInt8](repeating: 0, count: stride * H)
    buf.withUnsafeMutableBytes { raw in
        let ctx = CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                            bytesPerRow: stride, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)!   // BGRA in memory
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: W, height: H))
    }
    let crop = Crop(x: 0, y: 0, w: W, h: H)   // 03_infer_frames.py:11 -- the whole frame is the box
    let (center, scale) = centerScale(crop: crop, pp: pp)
    let a = Affine(center: center, scale: scale, outW: pp.width, outH: pp.height)
    let chw = buf.withUnsafeBufferPointer { warpCrop(base: $0.baseAddress!, stride: stride, crop: crop, affine: a, pp: pp) }
    return PreparedFrame(chw: chw, center: center, scale: scale, crop: crop)
}
