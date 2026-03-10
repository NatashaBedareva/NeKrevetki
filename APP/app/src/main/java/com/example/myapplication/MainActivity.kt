package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.pose.Pose

class MainActivity : AppCompatActivity(), CameraManagerListener {

    private lateinit var previewView: PreviewView
    private lateinit var poseOverlayView: PoseOverlayView
    private lateinit var textView: TextView
    private lateinit var cameraManager: CameraManager

    private val requestCodePermissions = 10
    @androidx.camera.core.ExperimentalGetImage
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        poseOverlayView = findViewById(R.id.poseOverlayView)
        textView = findViewById(R.id.textView)

        // Инициализируем CameraManager
        cameraManager = CameraManager(this, this, previewView, this)

        // Запрашиваем разрешение на камеру
        if (allPermissionsGranted()) {
            cameraManager.startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), requestCodePermissions
            )
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED
    @androidx.camera.core.ExperimentalGetImage
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == requestCodePermissions) {
            if (allPermissionsGranted()) {
                cameraManager.startCamera()
            } else {
                Toast.makeText(this, "Без разрешения камера не работает", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onPoseDetected(pose: Pose?, imageWidth: Int, imageHeight: Int) {
        runOnUiThread {
            poseOverlayView.setPose(pose, imageWidth, imageHeight)

            val landmarksCount = pose?.allPoseLandmarks?.size ?: 0
            val text = if (landmarksCount > 0)
                "Обнаружено точек: $landmarksCount"
            else
                "Поза не обнаружена"
            textView.text = text
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager.shutdown()
    }
}