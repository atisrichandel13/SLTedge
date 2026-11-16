// The person box, built the way OpenASL's boxes are built relative to the signer -- not iOS Vision's
// human rectangle.
//
// OpenASL's bbox-v1.0.json boxes are what every number in this project was measured with. Expressed in
// units of the signer's shoulder width, over 30 test clips (shoulders from RTMW keypoints), they are
// very consistent:
//     width  1.82 x shoulder   [p10 1.71, p90 1.98], centred on the shoulder midpoint (offset -0.01)
//     top    1.46 x shoulder ABOVE the shoulder line  [1.32, 1.54]
//     bottom 1.61 x shoulder BELOW it                 [1.47, 1.77]
// i.e. about 3.0 shoulder-widths tall, usually more than the frame, and clamped by crop_xywh.
// VNDetectHumanRectanglesRequest(upperBodyOnly) gave boxes 0.31-0.41 of the frame tall on the phone --
// a torso slice that cut the hands out of the crop whenever they moved, which is most of signing.
//
// So the box comes from VNDetectHumanBodyPoseRequest's two shoulder joints. Normalised [x0, y0, x1, y1],
// top-left origin, allowed outside 0..1 exactly as OpenASL's are.
import Vision
import CoreVideo

final class PersonBox {
    static let widthPerShoulder: Float = 1.82
    static let abovePerShoulder: Float = 1.46
    static let belowPerShoulder: Float = 1.61
    static let minJointConfidence: Float = 0.3

    private let request = VNDetectHumanBodyPoseRequest()

    /// Box for the person with the widest shoulders (the nearest signer), or nil if no one's shoulders
    /// are both visible.
    func detect(_ pb: CVPixelBuffer) -> [Float]? {
        let W = Float(CVPixelBufferGetWidth(pb)), H = Float(CVPixelBufferGetHeight(pb))
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        guard (try? handler.perform([request])) != nil, let people = request.results else { return nil }
        var best: (sh: Float, mid: SIMD2<Float>)?
        for p in people {
            guard let l = try? p.recognizedPoint(.leftShoulder), let r = try? p.recognizedPoint(.rightShoulder),
                  l.confidence >= Self.minJointConfidence, r.confidence >= Self.minJointConfidence else { continue }
            // Vision: normalised, origin bottom-left. To pixels, top-left, so distances are isotropic.
            let lp = SIMD2<Float>(Float(l.location.x) * W, (1 - Float(l.location.y)) * H)
            let rp = SIMD2<Float>(Float(r.location.x) * W, (1 - Float(r.location.y)) * H)
            let d = rp - lp
            let sh = (d.x * d.x + d.y * d.y).squareRoot()
            if sh > (best?.sh ?? 0) { best = (sh, (lp + rp) * 0.5) }
        }
        guard let b = best, b.sh > 8 else { return nil }
        let half = 0.5 * Self.widthPerShoulder * b.sh
        return [(b.mid.x - half) / W, (b.mid.y - Self.abovePerShoulder * b.sh) / H,
                (b.mid.x + half) / W, (b.mid.y + Self.belowPerShoulder * b.sh) / H]
    }
}
