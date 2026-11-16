// The person box. There is no detector anywhere in this project (task1_rtmpose/03_infer_frames.py:11);
// every box the board has ever consumed came from OpenASL's bbox-v1.0.json. On the phone the box
// comes from iOS Vision, and the board receives it the way it receives OpenASL's: normalised
// [x0, y0, x1, y1], top-left origin, allowed to fall outside 0..1 (crop_xywh clamps).
//
// upperBodyOnly matches the corpus framing better than a full-body box: OpenASL signers are framed
// chest-up, and a full-body box on a seated signer would put the crop centre below the hands.
import Vision
import CoreVideo

final class PersonBox {
    private let request: VNDetectHumanRectanglesRequest

    init() {
        request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = true
    }

    /// Largest detected box as normalised [x0, y0, x1, y1] with a top-left origin, or nil.
    func detect(_ pb: CVPixelBuffer) -> [Float]? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        guard (try? handler.perform([request])) != nil,
              let best = request.results?.max(by: { $0.boundingBox.width * $0.boundingBox.height
                                                   < $1.boundingBox.width * $1.boundingBox.height })
        else { return nil }
        let b = best.boundingBox       // Vision: origin bottom-left, normalised
        return [Float(b.minX), Float(1 - b.maxY), Float(b.maxX), Float(1 - b.minY)]
    }
}
