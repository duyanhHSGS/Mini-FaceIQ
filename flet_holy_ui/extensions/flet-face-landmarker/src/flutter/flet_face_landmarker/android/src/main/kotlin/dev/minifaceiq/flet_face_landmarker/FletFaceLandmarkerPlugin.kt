package dev.minifaceiq.flet_face_landmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.nio.ByteBuffer
import java.util.concurrent.Executors

/** Original MediaPipe Tasks face_landmarker.task, running fully offline on Android. */
class FletFaceLandmarkerPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context
    private val mainThread = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        context = binding.applicationContext
        channel = MethodChannel(binding.binaryMessenger, "dev.minifaceiq/face_landmarker")
        channel.setMethodCallHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "detect") {
            result.notImplemented()
            return
        }
        val imageBytes = call.argument<ByteArray>("image")
        val modelBytes = call.argument<ByteArray>("model")
        if (imageBytes == null) {
            result.error("NO_IMAGE", "No photo bytes were received", null)
            return
        }
        worker.execute {
            // One native graph per analysis: initialize -> detect -> close.
            var stage = "initialize model"
            var landmarker: FaceLandmarker? = null
            val startedAt = SystemClock.elapsedRealtime()
            try {
                val model = modelBytes
                    ?: throw IllegalStateException("The bundled face model was not received")
                // Keep a direct buffer strongly referenced until the native graph closes.
                val buffer = ByteBuffer.allocateDirect(model.size)
                buffer.put(model)
                buffer.rewind()
                Log.i("MINIFACE_MEDIAPIPE", "Initializing face model: ${model.size} bytes")
                val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(buffer).build())
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumFaces(1)
                    .setMinFaceDetectionConfidence(0.5f)
                    .setMinFacePresenceConfidence(0.5f)
                    .build()
                val detector = FaceLandmarker.createFromOptions(context, options)
                landmarker = detector
                val initializedAt = SystemClock.elapsedRealtime()
                stage = "decode image"
                val decoded = BitmapFactory.decodeByteArray(
                    imageBytes, 0, imageBytes.size,
                    BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
                ) ?: throw IllegalArgumentException("Invalid image bytes")
                val bitmap = if (decoded.config == Bitmap.Config.ARGB_8888) decoded else {
                    decoded.copy(Bitmap.Config.ARGB_8888, false)
                        ?: throw IllegalArgumentException("Unable to convert photo to ARGB_8888")
                }
                stage = "build MPImage"
                val image = BitmapImageBuilder(bitmap).build()
                stage = "detect face"
                val detected = detector.detect(image).faceLandmarks()
                val coords: List<List<Double>> = if (detected.isEmpty()) emptyList() else {
                    detected[0].map { listOf(it.x().toDouble(), it.y().toDouble()) }
                }
                val detectedAt = SystemClock.elapsedRealtime()
                stage = "release model"
                detector.close()
                landmarker = null
                Log.i(
                    "MINIFACE_MEDIAPIPE",
                    "Front model released; initialize=${initializedAt - startedAt}ms, " +
                        "decode+detect=${detectedAt - initializedAt}ms, " +
                        "total=${SystemClock.elapsedRealtime() - startedAt}ms"
                )
                // Preserve buffer until after close; do not keep it between analyses.
                buffer.rewind()
                mainThread.post { result.success(coords) }
            } catch (e: Exception) {
                Log.e("MINIFACE_MEDIAPIPE", "Face Landmarker failed at: $stage", e)
                val reason = "$stage: ${e.javaClass.simpleName}: ${e.message ?: "unknown native error"}"
                mainThread.post { result.error("MEDIAPIPE", reason, null) }
            } finally {
                try {
                    landmarker?.close()
                } catch (closeError: Exception) {
                    Log.e("MINIFACE_MEDIAPIPE", "Failed to release model", closeError)
                }
            }
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        // Queued requests release their own graph in finally.
        worker.shutdown()
    }
}
