from ultralytics import YOLO
import cv2
import numpy as np
import os

class YOLOPoseDetector:
    """
    Класс для обнаружения поз человека с использованием YOLOv8-Pose
    """
    
    # Словарь цветов для различных элементов
    COLORS = {
        'white': (255, 255, 255),
        'red': (0, 0, 255),
        'blue': (255, 0, 0)
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
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def draw_skeleton(self, image, keypoints, confs, pairs, color):
        """
        Рисование скелета по заданным парам ключевых точек
        
        Args:
            image: Изображение для рисования
            keypoints: Массив ключевых точек
            confs: Массив уверенностей для точек
            pairs: Список пар точек для соединения
            color: Цвет линий
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
        
        Args:
            image: Изображение для рисования
            keypoints: Массив ключевых точек
            confs: Массив уверенностей для точек
        """
        for j, (point, point_conf) in enumerate(zip(keypoints, confs)):
            if point_conf > self.confidence_threshold:
                x, y = int(point[0]), int(point[1])
                if (x, y) != (0, 0):
                    cv2.circle(image, (x, y), 5, self.COLORS['blue'], -1)
                    cv2.putText(image, str(j), (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['blue'], 2)
    
    def draw_bounding_box(self, image, box, class_name):
        """
        Рисование ограничивающей рамки с названием класса
        
        Args:
            image: Изображение для рисования
            box: Координаты рамки [x1, y1, x2, y2]
            class_name: Название класса
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), self.COLORS['white'], 2)
        cv2.putText(image, class_name, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['white'], 2)
    
    def process_image(self, image, draw_boxes=False, save_result=False, show_result=False):
        """
        Обработка одного изображения
        
        Args:
            image (numpy.ndarray): Изображение в формате BGR
            draw_boxes (bool): Рисовать ли ограничивающие рамки
            save_result (bool): Сохранять ли результат
            show_result (bool): Показывать ли результат в окне
            
        Returns:
            numpy.ndarray: Обработанное изображение или None в случае ошибки
        """
        if image is None:
            print("Ошибка: изображение не загружено")
            return None
        
        # Обработка изображения с помощью модели
        results = self.model(image)[0]
        
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
 
    detector = YOLOPoseDetector(model_path='yolov8n-pose.pt', confidence_threshold=0.5)
    
    image_path = 'image.png'
    detector.process_image(image_path, draw_boxes=False, save_result=True, show_result=True)