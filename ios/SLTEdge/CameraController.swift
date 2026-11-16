// Back camera, 1280x720 (the corpus height cap), BGRA frames in portrait, NOT mirrored.
//
// The data output is left unmirrored on purpose: OpenASL was recorded by a camera facing the signer,
// so left and right hands in the frame are the signer's right and left, and Uni-Sign keeps separate
// left-hand and right-hand groups (common/pose_to_unisign.py). A mirrored buffer would swap them.
// The back camera matches that geometry directly: a second person holds the phone facing the signer,
// which is also the counter/desk deployment of the proposal. With the front camera only the preview
// would be mirrored; the buffers never are.
import AVFoundation
import UIKit

final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    static let position: AVCaptureDevice.Position = .back
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "sltedge.camera", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?
    private(set) var frameSize: (w: Int, h: Int) = (0, 0)
    private(set) var nominalFPS: Double = 0

    func configure() throws {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        guard let cam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: Self.position) else {
            throw NSError(domain: "SLTEdge", code: 3, userInfo: [NSLocalizedDescriptionKey: "no camera at the requested position"])
        }
        let input = try AVCaptureDeviceInput(device: cam)
        if session.canAddInput(input) { session.addInput(input) }
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video) {
            if conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }   // portrait buffers
            conn.isVideoMirrored = false
        }
        session.commitConfiguration()
        let fmt = cam.activeFormat.videoSupportedFrameRateRanges.first
        nominalFPS = fmt?.maxFrameRate ?? 0
    }

    func start() { queue.async { if !self.session.isRunning { self.session.startRunning() } } }
    func stop() { queue.async { if self.session.isRunning { self.session.stopRunning() } } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if frameSize.w == 0 { frameSize = (CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb)) }
        onFrame?(pb, CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
    }
}
