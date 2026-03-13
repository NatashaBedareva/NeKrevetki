package com.example.myapplication

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import java.util.concurrent.Executors

interface CameraManagerListener {
    fun onPoseDetected(pose: Pose?, imageWidth: Int, imageHeight: Int)
}

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val listener: CameraManagerListener
) {

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val poseDetector: PoseDetector

    init {
        val options = AccuratePoseDetectorOptions.Builder()
            .setDetectorMode(AccuratePoseDetectorOptions.STREAM_MODE)
            .build()
        poseDetector = PoseDetection.getClient(options)
    }

    @ExperimentalGetImage
    fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(context))
    }

    @androidx.camera.core.ExperimentalGetImage
    private fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        if (mediaImage != null) {
            val image = InputImage.fromMediaImage(mediaImage, rotationDegrees)

            poseDetector.process(image)
                .addOnSuccessListener { pose ->
                    listener.onPoseDetected(pose, image.width, image.height)
                }
                .addOnFailureListener { e ->
                    e.printStackTrace()
                    listener.onPoseDetected(null, 0, 0)
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        } else {
            imageProxy.close()
        }
    }

    fun shutdown() {
        cameraExecutor.shutdown()
        poseDetector.close()
    }
}