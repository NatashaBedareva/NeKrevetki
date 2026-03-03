package com.example.myapplication

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark

class PoseOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var pose: Pose? = null
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0

    // Краски для рисования
    private val landmarkPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        strokeWidth = 8f
    }

    private val linePaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    // Связи между ключевыми точками (скелет)
    private val poseConnections = listOf(
        // Туловище
        Pair(PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
        Pair(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
        Pair(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
        Pair(PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),

        // Левая рука
        Pair(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
        Pair(PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),

        // Правая рука
        Pair(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
        Pair(PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),

        // Левая нога
        Pair(PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
        Pair(PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),

        // Правая нога
        Pair(PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
        Pair(PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),

        // Лицо (шея)
        Pair(PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_EAR),
        Pair(PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_EAR),
        Pair(PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR)
    )

    fun setPose(pose: Pose?, imageWidth: Int, imageHeight: Int) {
        this.pose = pose
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate() // Перерисовываем view
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        pose?.let { pose ->
            // Рисуем линии (скелет)
            drawPoseLines(canvas, pose)

            // Рисуем точки (суставы)
            drawPoseLandmarks(canvas, pose)
        }
    }

    private fun drawPoseLines(canvas: Canvas, pose: Pose) {
        for (connection in poseConnections) {
            val startLandmark = pose.getPoseLandmark(connection.first)
            val endLandmark = pose.getPoseLandmark(connection.second)

            if (startLandmark != null && endLandmark != null &&
                startLandmark.inFrameLikelihood > 0.5 && endLandmark.inFrameLikelihood > 0.5) {

                val startPoint = transformCoordinates(startLandmark.position)
                val endPoint = transformCoordinates(endLandmark.position)

                canvas.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y, linePaint)
            }
        }
    }

    private fun drawPoseLandmarks(canvas: Canvas, pose: Pose) {
        for (landmark in pose.allPoseLandmarks) {
            if (landmark.inFrameLikelihood > 0.5) { // Рисуем только достоверные точки
                val point = transformCoordinates(landmark.position)
                canvas.drawCircle(point.x, point.y, 8f, landmarkPaint)
            }
        }
    }

    // Трансформируем координаты из системы координат изображения в систему координат экрана
    private fun transformCoordinates(position: PointF): PointF {
        if (imageWidth == 0 || imageHeight == 0 || width == 0 || height == 0) {
            return position
        }

        // Вычисляем масштаб и смещение для центрирования изображения
        val viewAspectRatio = width.toFloat() / height
        val imageAspectRatio = imageWidth.toFloat() / imageHeight

        var scaleX = 1f
        var scaleY = 1f
        var offsetX = 0f
        var offsetY = 0f

        if (viewAspectRatio > imageAspectRatio) {
            // Изображение уже, чем view - подгоняем по высоте
            scaleY = height.toFloat() / imageHeight
            scaleX = scaleY
            offsetX = (width - imageWidth * scaleX) / 2
        } else {
            // Изображение шире, чем view - подгоняем по ширине
            scaleX = width.toFloat() / imageWidth
            scaleY = scaleX
            offsetY = (height - imageHeight * scaleY) / 2
        }

        return PointF(
            position.x * scaleX + offsetX,
            position.y * scaleY + offsetY
        )
    }
}