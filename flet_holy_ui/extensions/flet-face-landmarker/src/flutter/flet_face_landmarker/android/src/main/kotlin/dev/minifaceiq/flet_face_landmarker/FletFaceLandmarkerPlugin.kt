package dev.minifaceiq.flet_face_landmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
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
    private var landmarker: FaceLandmarker? = null
    // Retain direct model buffer while graph remains open.
    private var modelBuffer: ByteBuffer? = null

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
            var stage = "initialize model"
            try {
                if (landmarker == null) {
                    if (modelBytes == null) {
                        throw IllegalStateException("The bundled face model was not received")
                    }
                    // MediaPipe requires a DIRECT ByteBuffer, not ByteBuffer.wrap().
                    val buffer = ByteBuffer.allocateDirect(modelBytes.size)
                    buffer.put(modelBytes)
                    buffer.rewind()
                    modelBuffer = buffer
                    Log.i("MINIFACE_MEDIAPIPE", "Initializing face model: ${modelBytes.size} bytes")
                    val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(buffer).build())
                        .setRunningMode(RunningMode.IMAGE)
                        .setNumFaces(1)
                        .setMinFaceDetectionConfidence(0.5f)
                        .setMinFacePresenceConfidence(0.5f)
                        .build()
                    landmarker = FaceLandmarker.createFromOptions(context, options)
                }
                stage = "decode image"
                val decoded = BitmapFactory.decodeByteArray(
                    imageBytes, 0, imageBytes.size,
                    BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
                ) ?: throw IllegalArgumentException("Invalid image bytes")
                // The MediaPipe Java API accepts ARGB_8888, not every BitmapFactory format.
                val bitmap = if (decoded.config == Bitmap.Config.ARGB_8888) decoded else {
                    decoded.copy(Bitmap.Config.ARGB_8888, false)
                        ?: throw IllegalArgumentException("Unable to convert photo to ARGB_8888")
                }
                stage = "build MPImage"
                val image = BitmapImageBuilder(bitmap).build()
                stage = "detect face"
                val detected = landmarker!!.detect(image).faceLandmarks()
                val coords: List<List<Double>> = if (detected.isEmpty()) emptyList() else {
                    detected[0].map { listOf(it.x().toDouble(), it.y().toDouble()) }
                }
                mainThread.post { result.success(coords) }
            } catch (e: Exception) {
                // Log the full stack trace, not just the truncated Flet status message.
                Log.e("MINIFACE_MEDIAPIPE", "Face Landmarker failed at: $stage", e)
                val reason = "$stage: ${e.javaClass.simpleName}: ${e.message ?: "unknown native error"}"
                mainThread.post { result.error("MEDIAPIPE", reason, null) }
            }
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        worker.execute { landmarker?.close(); landmarker = null; modelBuffer = null }
        worker.shutdown()
    }
}
