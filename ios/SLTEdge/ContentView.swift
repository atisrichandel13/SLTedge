import SwiftUI
import AVFoundation

struct ContentView: View {
    @EnvironmentObject var m: AppModel
    @ObservedObject var logStore = LogStore.shared

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                CameraPreview(session: m.camera.session)
                GeometryReader { g in
                    if let b = m.liveBox {
                        // The box is in buffer coordinates. The preview layer mirrors the front
                        // camera only, so flip x in that case to draw it where the signer sees it.
                        let mirrored = CameraController.position == .front
                        let x0 = CGFloat(mirrored ? 1 - b[2] : b[0]), x1 = CGFloat(mirrored ? 1 - b[0] : b[2])
                        Rectangle()
                            .stroke(m.isRecording ? Color.red : Color.green, lineWidth: 2)
                            .frame(width: (x1 - x0) * g.size.width, height: CGFloat(b[3] - b[1]) * g.size.height)
                            .position(x: (x0 + x1) / 2 * g.size.width, y: CGFloat(b[1] + b[3]) / 2 * g.size.height)
                    }
                }
            }
            .aspectRatio(9 / 16, contentMode: .fit)
            .frame(maxHeight: 300)
            .clipped()

            Text(m.translation.isEmpty ? " " : m.translation)
                .font(.title3).multilineTextAlignment(.center).padding(.horizontal)
                .frame(minHeight: 40)

            Text(m.status).font(.footnote).bold()
                .multilineTextAlignment(.center).padding(.horizontal).frame(minHeight: 32)

            ScrollView {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(logStore.lines.suffix(8).enumerated()), id: \.offset) { _, l in
                        Text(l).font(.system(size: 10, design: .monospaced)).lineLimit(2)
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 110).padding(.horizontal, 8)
            .background(Color.secondary.opacity(0.08))

            Circle()
                .fill(m.isRecording ? Color.red : (m.busy ? Color.gray : Color.blue))
                .frame(width: 84, height: 84)
                .overlay(Text(m.isRecording ? "signing" : "hold").foregroundColor(.white).bold())
                .gesture(DragGesture(minimumDistance: 0)
                    .onChanged { _ in if !m.isRecording { m.pressBegan() } }
                    .onEnded { _ in m.pressEnded() })

            HStack {
                Text(m.translatorState).font(.caption2).foregroundColor(.secondary).lineLimit(1)
                Spacer()
                Button("load") { m.loadTranslatorIfPresent() }.font(.caption).disabled(m.busy)
            }.padding(.horizontal)

            HStack {
                Picker("pose", selection: $m.variant) {
                    ForEach(PoseVariant.allCases) { Text($0.label).tag($0) }
                }.pickerStyle(.menu).disabled(m.busy)
                Picker("compute", selection: $m.compute) {
                    ForEach(Compute.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.menu).disabled(m.busy)
                Picker("beams", selection: $m.beams) {
                    Text("beam 4").tag(4); Text("greedy").tag(1)
                }.pickerStyle(.menu).disabled(m.busy)
                Spacer()
                Button(m.parityProgress.isEmpty || m.parity != nil ? "parity test" : m.parityProgress) { m.runParity() }
                    .disabled(m.busy)
            }.padding(.horizontal)
        }
        .padding(.vertical)
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        if let c = v.previewLayer.connection, c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
