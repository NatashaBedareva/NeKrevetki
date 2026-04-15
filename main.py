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
    process_interval = 1.0
    last_process_time = time.time()
    last_processing_image = None
    slouch_start_time = 0
    slouch_detected_count =0;
    slouch_detected = False
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
            display_img = cv2.flip(img, 1)
            # Ограничение FPS - пропускаем кадры если обрабатываем слишком быстро
            if current_time - last_process_time >= process_interval:
            
                last_process_time = current_time
            
            
               
                img = cv2.flip(img, 1) 
            
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                processed_pil = advanced_preprocessing(pil_img)
            
                processed_img = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)

                # Обработка изображения детектором
                detector.process_image(processed_img, draw_boxes=False)

                all_points_eye, all_points_shoulders =detector.get_keypoints_by_indices(processed_img, indices=[3,4,5,6])
                signal_person = detector.analyze_slouch(all_points_eye, all_points_shoulders)
                slouch_detected_count =0
                if len(signal_person) != 0:
   
                    if not slouch_detected:
       
                        slouch_start_time = time.time()
                        slouch_detected = True
                        slouch_detected_count = 0
       
                    else:
                        cur = time.time()
                        elapsed = cur - slouch_start_time
                        slouch_detected_count = 0
                        if  cur - slouch_start_time >= 7:
                            detector.signal(signal_person)
                            slouch_start_time = 0
                            slouch_detected = False
                else:
                
                    slouch_detected_count += 1
                    if slouch_detected_count == 3:
                            slouch_start_time = 0
                            slouch_detected = False


                
                    
                

                
                last_processing_image= processed_img;
            
                cv2.imshow('result', processed_img)
            else :
                if (last_processing_image is not None):
                    cv2.imshow('result', last_processing_image)
            
            
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            cap.release()
            raise
 
        ch = cv2.waitKey(1)  
        if ch == 27: 
            break
        
    cap.release()
    cv2.destroyAllWindows()
