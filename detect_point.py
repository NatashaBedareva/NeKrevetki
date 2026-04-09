from ultralytics import YOLO
import cv2
import numpy as np
import os
import winsound
import logging

class YOLOPoseDetector:
    """
    Класс для обнаружения поз человека с использованием YOLOv8-Pose
    """
    
    # Словарь цветов для различных элементов
    COLORS = {
        'white': (255, 255, 255),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255)
    }
    
    # Пары ключевых точек для рисования скелета
    SKELETON_PAIRS = {
        'arms': [(5, 7), (7, 9), (6, 8), (8, 10)],  # Руки
        'legs': [(11, 13), (13, 15), (12, 14), (14, 16)],  # Ноги
        'body': [(5, 11), (6, 12)]  # Тело
    }
    
    def __init__(self, model_path='yolov8n-pose.pt', confidence_threshold=0.5):
        """
        Инициализация детектора поз
        
        Args:
            model_path (str): Путь к файлу модели YOLO
            confidence_threshold (float): Порог уверенности для отображения точек
        """
        # Отключаем вывод YOLO
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        os.environ['YOLO_VERBOSE'] = 'False'
        
        # Загружаем модель с отключенным выводом
        self.model = YOLO(model_path, verbose=False)
        self.confidence_threshold = confidence_threshold
        
        # Для хранения состояния наклона
        self.eye_tilt_detected = False
        self.shoulder_tilt_detected = False
    
    def get_keypoints_by_indices(self, image, indices=[3,4,5,6]):
        """
        Получение координат указанных ключевых точек
        
        Параметры:
            image: изображение
            indices: список индексов точек
        
        Возвращает:
            list: список словарей с координатами для каждого человека
        """
        # Отключаем вывод для предсказания
        results = self.model(image, verbose=False)[0]
        points_eye = []
        points_shoulders = []
        all_points_eye = []
        all_points_shoulders = []
        
        if (hasattr(results, 'keypoints') and results.keypoints is not None and
            hasattr(results, 'boxes') and results.boxes is not None):
            
            keypoints = results.keypoints.data.cpu().numpy()
            confs = results.keypoints.conf.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            
            if len(keypoints) > 0 and len(boxes) > 0:
                # Находим самого крупного человека
                max_area = 0
                best_idx = 0
                
                for i, box in enumerate(boxes):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > max_area:
                        max_area = area
                        best_idx = i
                
                # Берем только самого крупного человека
                kp = keypoints[best_idx]
                conf_arr = confs[best_idx]
                
                person_eye = {'person_id': best_idx, 'left_eye': None, 'right_eye': None}
                person_shoulders = {'person_id': best_idx, 'left_shoulder': None, 'right_shoulder': None}
                
                for idx in indices:
                    if idx < len(kp) and conf_arr[idx] > self.confidence_threshold:
                        x, y = int(kp[idx][0]), int(kp[idx][1])

                        if idx == 3:  # левый глаз
                            person_eye['left_eye'] = (x, y, conf_arr[idx])
                        elif idx == 4:  # правый глаз
                            person_eye['right_eye'] = (x, y, conf_arr[idx])
                        elif idx == 5:  # левое плечо
                            person_shoulders['left_shoulder'] = (x, y, conf_arr[idx])
                        elif idx == 6:  # правое плечо
                            person_shoulders['right_shoulder'] = (x, y, conf_arr[idx])
                        
                all_points_eye.append(person_eye)
                all_points_shoulders.append(person_shoulders)
                
            return all_points_eye, all_points_shoulders
        
        return points_eye, points_shoulders
    
    def draw_eye_line(self, image, person_eye, color):
        """
        Рисование линии между глазами
        
        Args:
            image: изображение
            person_eye: словарь с координатами глаз
            color: цвет линии
        """
        if (person_eye.get('left_eye') is not None and 
            person_eye.get('right_eye') is not None):
            
            x1, y1 = person_eye['left_eye'][0], person_eye['left_eye'][1]
            x2, y2 = person_eye['right_eye'][0], person_eye['right_eye'][1]
            
            # Рисуем линию между глазами
            cv2.line(image, (x1, y1), (x2, y2), color, 3)
            
            # Добавляем подпись с разницей высот
            diff = abs(y1 - y2)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(image, f"{diff:.0f}px", (mid_x - 30, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_shoulder_line(self, image, person_shoulders, color):
        """
        Рисование линии между плечами
        
        Args:
            image: изображение
            person_shoulders: словарь с координатами плеч
            color: цвет линии
        """
        if (person_shoulders.get('left_shoulder') is not None and 
            person_shoulders.get('right_shoulder') is not None):
            
            x1, y1 = person_shoulders['left_shoulder'][0], person_shoulders['left_shoulder'][1]
            x2, y2 = person_shoulders['right_shoulder'][0], person_shoulders['right_shoulder'][1]
            
            # Рисуем линию между плечами
            cv2.line(image, (x1, y1), (x2, y2), color, 3)
            
            # Добавляем подпись с разницей высот
            diff = abs(y1 - y2)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(image, f"{diff:.0f}px", (mid_x - 30, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def analyze_slouch(self, all_points_eye, all_points_shoulders):
        """
        Анализ наклона и возврат списка нарушений
        """
        len_all_points_eye = len(all_points_eye)
        len_all_points_shoulders = len(all_points_shoulders)
        signal_person = []
        
        # Сбрасываем флаги
        self.eye_tilt_detected = False
        self.shoulder_tilt_detected = False
        
        # Анализ глаз
        for idx in range(len_all_points_eye):
            if (all_points_eye[idx].get('left_eye') is not None and 
                all_points_eye[idx].get('right_eye') is not None):
                
                y1 = all_points_eye[idx].get('left_eye')[1]
                y2 = all_points_eye[idx].get('right_eye')[1]
                
                if abs(y1 - y2) > 40:
                    signal_person.append((idx, 'eye'))
                    self.eye_tilt_detected = True
        
        # Анализ плеч
        for idx in range(len_all_points_shoulders):
            if (all_points_shoulders[idx].get('left_shoulder') is not None and 
                all_points_shoulders[idx].get('right_shoulder') is not None):
                
                y1 = all_points_shoulders[idx].get('left_shoulder')[1]
                y2 = all_points_shoulders[idx].get('right_shoulder')[1]
                
                if abs(y1 - y2) > 40:
                    signal_person.append((idx, 'shoulder'))
                    self.shoulder_tilt_detected = True
        
        return signal_person
    
    def draw_tilt_lines(self, image, all_points_eye, all_points_shoulders):
        """
        Рисование линий между глазами и плечами с учетом наклона
        """
        # Рисуем линии для глаз
        for person_eye in all_points_eye:
            if self.eye_tilt_detected:
                color = self.COLORS['red']  # Красный при наклоне
            else:
                color = self.COLORS['green']  # Зеленый в норме
            self.draw_eye_line(image, person_eye, color)
        
        # Рисуем линии для плеч
        for person_shoulders in all_points_shoulders:
            if self.shoulder_tilt_detected:
                color = self.COLORS['red']  # Красный при наклоне
            else:
                color = self.COLORS['green']  # Зеленый в норме
            self.draw_shoulder_line(image, person_shoulders, color)
    
    def signal(self, signal_person):
        """
        Звуковой сигнал при обнаружении наклона
        """
        if len(signal_person) > 0:
            # Разные звуки для разных типов наклона
            has_eye = any(tilt_type == 'eye' for _, tilt_type in signal_person)
            has_shoulder = any(tilt_type == 'shoulder' for _, tilt_type in signal_person)
            
            if has_eye and has_shoulder:
                frequency = 400
                duration = 800
            elif has_eye:
                frequency = 500
                duration = 500
            elif has_shoulder:
                frequency = 300
                duration = 700
            else:
                frequency = 250
                duration = 1000
            
            winsound.Beep(frequency, duration)
        return

    def draw_skeleton(self, image, keypoints, confs, pairs, color):
        """
        Рисование скелета по заданным парам ключевых точек
        """
        for (start, end) in pairs:
            if (confs[start] > self.confidence_threshold and 
                confs[end] > self.confidence_threshold):
                x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                
                if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)
    
    def draw_keypoints(self, image, keypoints, confs):
        """
        Рисование ключевых точек с их номерами
        """
        for j, (point, point_conf) in enumerate(zip(keypoints, confs)):
            if point_conf > self.confidence_threshold:
                x, y = int(point[0]), int(point[1])
                if (x, y) != (0, 0):
                    # Выделяем глаза и плечи другим цветом
                    if j in [3, 4]:  # Глаза
                        color = self.COLORS['yellow']
                        radius = 6
                    elif j in [5, 6]:  # Плечи
                        color = self.COLORS['yellow']
                        radius = 6
                    else:
                        color = self.COLORS['blue']
                        radius = 5
                    
                    cv2.circle(image, (x, y), radius, color, -1)
                    cv2.putText(image, str(j), (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['blue'], 2)
    
    def draw_bounding_box(self, image, box, class_name):
        """
        Рисование ограничивающей рамки с названием класса
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), self.COLORS['white'], 2)
        cv2.putText(image, class_name, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 2)
    
    def process_image(self, image, draw_boxes=False, save_result=False, show_result=False):
        """
        Обработка одного изображения
        """
        if image is None:
            print("Ошибка: изображение не загружено")
            return None
        
        # Обработка изображения с моделью
        results = self.model(image, verbose=False)[0]
        
        # Получаем точки для анализа
        all_points_eye, all_points_shoulders = self.get_keypoints_by_indices(image)
        
        # Анализируем наклон
        signal_person = self.analyze_slouch(all_points_eye, all_points_shoulders)
        
        # Рисуем линии между глазами и плечами
        self.draw_tilt_lines(image, all_points_eye, all_points_shoulders)
        
        # Проверка на наличие обнаруженных объектов
        if (hasattr(results, 'boxes') and hasattr(results.boxes, 'cls') 
            and len(results.boxes.cls) > 0):
            classes_names = results.names
            classes = results.boxes.cls.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
            
            # Обработка ключевых точек
            if results.keypoints is not None:
                keypoints = results.keypoints.data.cpu().numpy()
                confs = results.keypoints.conf.cpu().numpy()
                
                for i, (class_id, box, kp, conf) in enumerate(zip(classes, boxes, keypoints, confs)):
                    if draw_boxes:
                        class_name = classes_names[int(class_id)]
                        self.draw_bounding_box(image, box, class_name)
                    
                    # Визуализация ключевых точек
                    self.draw_keypoints(image, kp, conf)
                    
                    # Рисование скелета
                    self.draw_skeleton(image, kp, conf, self.SKELETON_PAIRS['arms'], self.COLORS['white'])
                    self.draw_skeleton(image, kp, conf, self.SKELETON_PAIRS['legs'], self.COLORS['red'])
                    self.draw_skeleton(image, kp, conf, self.SKELETON_PAIRS['body'], self.COLORS['blue'])
        
        # Добавляем статус на изображение
        if self.eye_tilt_detected:
            cv2.putText(image, "EYE TILT DETECTED!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['red'], 2)
        if self.shoulder_tilt_detected:
            cv2.putText(image, "SHOULDER TILT DETECTED!", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['red'], 2)
        
        # Сохранение результатов
        if save_result:
            output_path = "result_pose_detected.jpg"
            cv2.imwrite(output_path, image)
            print(f"Сохранено изображение с результатами: {output_path}")
        
        # Отображение результатов
        if show_result:
            cv2.imshow('YOLOv8-Pose Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return image


if __name__ == "__main__":
    # Отключаем вывод YOLO глобально
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    os.environ['YOLO_VERBOSE'] = 'False'
    
    detector = YOLOPoseDetector(model_path='yolov8n-pose.pt', confidence_threshold=0.5)
    
    image_path = 'image.png'
    detector.process_image(image_path, draw_boxes=False, save_result=True, show_result=True)