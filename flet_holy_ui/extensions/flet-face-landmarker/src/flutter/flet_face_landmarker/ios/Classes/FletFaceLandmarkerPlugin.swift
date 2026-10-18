import Flutter
import Foundation
import MediaPipeTasksVision
import UIKit

/// The same offline 478-point MediaPipe model and MethodChannel as Android.
public final class FletFaceLandmarkerPlugin: NSObject, FlutterPlugin {
    private let worker = DispatchQueue(label: "dev.minifaceiq.face-landmarker")

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "dev.minifaceiq/face_landmarker",
            binaryMessenger: registrar.messenger()
        )
        registrar.addMethodCallDelegate(FletFaceLandmarkerPlugin(), channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        if call.method == "convertToJpeg" {
            guard let args = call.arguments as? [String: Any],
                  let image = args["image"] as? FlutterStandardTypedData else {
                result(FlutterError(code: "NO_IMAGE", message: "Missing photo bytes", details: nil))
                return
            }
            worker.async {
                do {
                    let jpeg = try Self.convertToJpeg(image.data)
                    DispatchQueue.main.async { result(FlutterStandardTypedData(bytes: jpeg)) }
                } catch {
                    DispatchQueue.main.async {
                        result(FlutterError(
                            code: "HEIC_DECODE", message: error.localizedDescription, details: nil
                        ))
                    }
                }
            }
            return
        }
        guard call.method == "detect" else {
            result(FlutterMethodNotImplemented)
            return
        }
        guard let args = call.arguments as? [String: Any],
              let image = args["image"] as? FlutterStandardTypedData,
              let model = args["model"] as? FlutterStandardTypedData else {
            result(FlutterError(code: "MISSING_DATA", message: "Missing image/model bytes", details: nil))
            return
        }

        // Decode and inference run off the UI thread. A new task is used per
        // request, as on Android; temp model bytes never leave the device.
        worker.async {
            do {
                let coords = try Self.detect(image: image.data, model: model.data)
                DispatchQueue.main.async { result(coords) }
            } catch {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "MEDIAPIPE",
                        message: "iOS face detection: \(error.localizedDescription)",
                        details: nil
                    ))
                }
            }
        }
    }

    /// UIImage supports the iPhone camera's HEIC/HEIF image data.
    private static func uprightImage(_ bytes: Data) throws -> UIImage {
        guard let source = UIImage(data: bytes) else {
            throw NSError(domain: "MiniFaceIQ", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid image data"])
        }
        // A normalized RGBA image avoids mirrored EXIF orientation surprises.
        let format = UIGraphicsImageRendererFormat()
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: source.size, format: format)
        return renderer.image { _ in
            source.draw(in: CGRect(origin: .zero, size: source.size))
        }
    }

    private static func convertToJpeg(_ imageBytes: Data) throws -> Data {
        let upright = try uprightImage(imageBytes)
        guard let jpeg = upright.jpegData(compressionQuality: 0.9) else {
            throw NSError(domain: "MiniFaceIQ", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "JPEG encoding failed"])
        }
        return jpeg
    }

    private static func detect(image imageBytes: Data, model modelBytes: Data) throws -> [[Double]] {
        let input = try MPImage(uiImage: uprightImage(imageBytes))

        // MediaPipe's iOS BaseOptions accepts a local path, not model bytes.
        let modelURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mini-faceiq-\(UUID().uuidString).task")
        try modelBytes.write(to: modelURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: modelURL) }

        let options = FaceLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelURL.path
        options.runningMode = .image
        options.numFaces = 1
        options.minFaceDetectionConfidence = 0.5
        options.minFacePresenceConfidence = 0.5
        let landmarker = try FaceLandmarker(options: options)
        let faces = try landmarker.detect(image: input).faceLandmarks
        return faces.first?.map { [Double($0.x), Double($0.y)] } ?? []
    }
}
