import cv2
import time
import numpy as np
from PIL import Image, ImageEnhance
import os

from video_capture import advanced_preprocessing
from detect_point import YOLOPoseDetector


if __name__ == '__main__':

    cv2.namedWindow("result")

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
   
    prev_time = time.time()
    frame_count = 0
    fps_res = 0
    
    # Ограничение FPS
    target_fps = 15  # Желаемый FPS
    frame_time = 1.0 / target_fps
    last_process_time = time.time()
    
    # Создаем детектор один раз перед циклом
    detector = YOLOPoseDetector(model_path='yolov8n-pose.pt', confidence_threshold=0.5)

    while True:
        flag, img = cap.read()
        fps_res = 0
        
        if not flag:
            print("Не удалось получить кадр с камеры")
            break
            
        try:
            current_time = time.time()
            
            # Ограничение FPS - пропускаем кадры если обрабатываем слишком быстро
            if current_time - last_process_time < frame_time:
                # Пропускаем обработку, показываем оригинальный кадр
                cv2.putText(img, f"FPS: {fps_res:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('result', img)
                
                ch = cv2.waitKey(1)
                if ch == 27:
                    break
                continue
            
            last_process_time = current_time
            
            frame_count += 1
            if current_time - prev_time >= 1.0:
               fps_res = frame_count / (current_time - prev_time)
               frame_count = 0
               prev_time = current_time
               
            img = cv2.flip(img, 1) 
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            processed_pil = advanced_preprocessing(pil_img)
            
            processed_img = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)

            # Обработка изображения детектором
            detector.process_image(processed_img, draw_boxes=False)

            if fps_res > 0:
                cv2.putText(processed_img, f"FPS: {fps_res:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('result', processed_img)
            
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            cap.release()
            raise
 
        ch = cv2.waitKey(1)  
        if ch == 27: 
            break
        
    cap.release()
    cv2.destroyAllWindows()