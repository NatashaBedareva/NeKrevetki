import tkinter as tk
from tkinter import ttk
import cv2
import time
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import threading
import os

import sys


# Импорт ваших модулей
from video_capture import advanced_preprocessing
from detect_point import YOLOPoseDetector


def resource_path(relative_path):
    """Получить абсолютный путь к файлу. Работает в разработке и в скомпилированном .exe"""
    try:
        # PyInstaller создает временную папку и хранит путь в _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Запущено как обычный скрипт
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)




class VideoProcessorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Pose Detection")
        self.window.geometry("800x650")
        self.window.configure(bg='#2b2b2b')
        
        # Переменные состояния
        self.is_running = False
        self.is_calibrating = False
        self.cap = None
        self.detector = None
        self.video_thread = None
        
        # Параметры калибровки
        self.calibration_data = {
            'eye_level': 0,
            'shoulder_level': 0,
            'tilt_threshold': 40
        }
        
        # Параметры видео
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        self.process_interval = 1.0
        self.last_process_time = time.time()
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_res = 0
        self.last_processing_image = None
        
        # Параметры сутулости
        self.slouch_start_time = 0
        self.slouch_detected_count = 0
        self.slouch_detected = False
        
        # Кэш для изображений
        self.current_photo = None
        self.canvas_image_id = None
        
        # Создание интерфейса
        self.create_widgets()
        
        # Привязка закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        # Главный контейнер
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Фрейм для видео (по центру)
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas для отображения видео
        self.canvas = tk.Canvas(video_frame, bg='black', width=640, height=480)
        self.canvas.pack(pady=10)
        
        # Фрейм для кнопок управления
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10)
        
        # Кнопки Start и Stop
        self.start_btn = ttk.Button(control_frame, text="▶ START", 
                                   command=self.start_video, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="■ STOP", 
                                  command=self.stop_video, width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка Калибровка
        self.calibrate_btn = ttk.Button(control_frame, text="⚙ КАЛИБРОВКА", 
                                       command=self.calibrate, width=15)
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        # Фрейм для информации о калибровке
        calib_frame = ttk.Frame(main_frame)
        calib_frame.pack(fill=tk.X, pady=5)
        
        self.calibration_label = ttk.Label(calib_frame, 
                                          text="Калибровка: не выполнена", 
                                          font=('Arial', 9),
                                          foreground='gray')
        self.calibration_label.pack(side=tk.LEFT, padx=10)
        
        # Информационная панель
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        # Статус
        self.status_label = ttk.Label(info_frame, text="Статус: Остановлен", 
                                     font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # FPS
        self.fps_label = ttk.Label(info_frame, text="FPS: 0.0", 
                                  font=('Arial', 10))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
    
    def calibrate(self):
        """Функция-заглушка для калибровки"""
        if self.is_calibrating:
            return
        
        self.is_calibrating = True
        self.calibrate_btn.config(state=tk.DISABLED, text="⏳ КАЛИБРОВКА...")
        
        def calibration_process():
            time.sleep(2)
            
            self.calibration_data = {
                'eye_level': 100,
                'shoulder_level': 200,
                'tilt_threshold': 40,
                'calibration_time': time.strftime("%H:%M:%S")
            }
            
            self.window.after(0, self._finish_calibration)
        
        calib_thread = threading.Thread(target=calibration_process)
        calib_thread.daemon = True
        calib_thread.start()
        
        self.status_label.config(text="Статус: Калибровка...")
    
    def _finish_calibration(self):
        """Завершение калибровки и обновление интерфейса"""
        self.is_calibrating = False
        self.calibrate_btn.config(state=tk.NORMAL, text="⚙ КАЛИБРОВКА")
        
        calib_time = self.calibration_data.get('calibration_time', 'неизвестно')
        self.calibration_label.config(
            text=f"✓ Калибровка выполнена в {calib_time} | Порог: {self.calibration_data['tilt_threshold']}px",
            foreground='green'
        )
        
        if self.is_running:
            self.status_label.config(text="Статус: Работает")
        else:
            self.status_label.config(text="Статус: Остановлен")
        
        print(f"[КАЛИБРОВКА] Завершена. Данные: {self.calibration_data}")
    
    def start_video(self):
        """Запуск видео потока"""
        if self.is_running:
            return
            
        # Инициализация камеры с оптимизированными настройками
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if not self.cap.isOpened():
            self.status_label.config(text="Статус: Ошибка камеры")
            return
        
        # Инициализация детектора
        try:
            # Получаем правильный путь к файлу модели
            model_path = resource_path('yolov8n-pose.pt')
            self.detector = YOLOPoseDetector(
                model_path=model_path, 
                confidence_threshold=0.5
            )
        except Exception as e:
            self.status_label.config(text=f"Статус: Ошибка загрузки модели")
            print(f"Ошибка загрузки модели из {model_path}: {e}")
            if self.cap:
                self.cap.release()
            return
        
        # Сброс счетчиков времени
        self.last_process_time = time.time()
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_res = 0
        self.last_processing_image = None
        self.slouch_start_time = 0
        self.slouch_detected_count = 0
        self.slouch_detected = False
        
        # Запуск потока обработки
        self.is_running = True
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Обновление интерфейса
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.calibrate_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Статус: Работает")
    
    def stop_video(self):
        """Остановка видео потока"""
        self.is_running = False
        
        # Ожидание завершения потока
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        
        # Освобождение ресурсов
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.detector = None
        self.current_photo = None
        self.canvas_image_id = None
        
        # Очистка canvas
        self.canvas.delete("all")
        
        # Обновление интерфейса
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if not self.is_calibrating:
            self.calibrate_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Статус: Остановлен")
        self.fps_label.config(text="FPS: 0.0")
    
    def process_video(self):
        """Обработка видео в отдельном потоке"""
        # Предварительное создание постоянного черного фона для canvas
        self.window.after(0, self._create_canvas_background)
        
        while self.is_running:
            try:
                flag, img = self.cap.read()
                self.fps_res = 0
                
                if not flag:
                    print("Не удалось получить кадр с камеры")
                    break
                
                current_time = time.time()
                
                # Ограничение FPS
                if current_time - self.last_process_time >= self.process_interval:
                    self.last_process_time = current_time
                    
                    img = cv2.flip(img, 1)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    # Предобработка изображения
                    processed_pil = advanced_preprocessing(pil_img)
                    processed_img = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
                    
                    # Обработка изображения детектором
                    self.detector.process_image(processed_img, draw_boxes=False)
                    
                    all_points_eye, all_points_shoulders = self.detector.get_keypoints_by_indices(
                        processed_img, indices=[3, 4, 5, 6]
                    )
                    signal_person = self.detector.analyze_slouch(all_points_eye, all_points_shoulders)
                    
                    # Логика обнаружения сутулости
                    self.slouch_detected_count = 0
                    if len(signal_person) != 0:
                        if not self.slouch_detected:
                            self.slouch_start_time = time.time()
                            self.slouch_detected = True
                            self.slouch_detected_count = 0
                        else:
                            cur = time.time()
                            if cur - self.slouch_start_time >= 7:
                                self.detector.signal(signal_person)
                                self.slouch_start_time = 0
                                self.slouch_detected = False
                    else:
                        self.slouch_detected_count += 1
                        if self.slouch_detected_count == 3:
                            self.slouch_start_time = 0
                            self.slouch_detected = False
                    
                    self.last_processing_image = processed_img
                    
                    # Расчет FPS
                    self.frame_count += 1
                    if current_time - self.prev_time >= 1.0:
                        self.fps_res = self.frame_count / (current_time - self.prev_time)
                        self.frame_count = 0
                        self.prev_time = current_time
                        
                        # Обновление FPS в интерфейсе
                        if self.fps_res > 0:
                            self.window.after(0, lambda: self.fps_label.config(
                                text=f"FPS: {self.fps_res:.1f}"
                            ))
                    
                    # Добавление информации на изображение
                    if self.fps_res > 0:
                        cv2.putText(self.last_processing_image, f"FPS: {self.fps_res:.1f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                    
                    if self.calibration_data.get('calibration_time'):
                        cv2.putText(self.last_processing_image, "Calibrated", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (255, 255, 0), 1)
                    
                    if self.slouch_detected:
                        remaining_time = max(0, 7 - (time.time() - self.slouch_start_time))
                        cv2.putText(self.last_processing_image, f"Slouch detected! Time to alert: {remaining_time:.1f}s", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 255), 2)
                
                # Обновление отображения (всегда показываем последний кадр)
                if self.last_processing_image is not None:
                    self.update_canvas(self.last_processing_image)
                
                # Оптимальная задержка
                time.sleep(0.005)
                
            except Exception as e:
                print(f"Ошибка обработки: {e}")
                break
        
        # Если цикл завершился, останавливаем видео
        if self.is_running:
            self.window.after(0, self.stop_video)
    
    def _create_canvas_background(self):
        """Создание постоянного фона для canvas"""
        if not self.canvas_image_id:
            # Создаем черное изображение как фон
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                self.canvas_image_id = self.canvas.create_image(
                    canvas_w // 2, canvas_h // 2, 
                    anchor=tk.CENTER
                )
    
    def update_canvas(self, img):
        """Оптимизированное обновление изображения в canvas без мерцания"""
        try:
            # Конвертация BGR в RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Получение размеров canvas
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w <= 1 or canvas_h <= 1:
                # Canvas еще не инициализирован, используем размеры по умолчанию
                canvas_w, canvas_h = 640, 480
            
            # Изменение размера с сохранением пропорций
            h, w = img_rgb.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Используем INTER_NEAREST для скорости или INTER_LINEAR для качества
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Конвертация в PhotoImage
            pil_image = Image.fromarray(img_resized)
            new_photo = ImageTk.PhotoImage(image=pil_image)
            
            # Обновление изображения без очистки canvas
            def update_image():
                if self.canvas_image_id is None:
                    # Первое создание изображения
                    x = (canvas_w - new_w) // 2
                    y = (canvas_h - new_h) // 2
                    self.canvas_image_id = self.canvas.create_image(
                        x, y, image=new_photo, anchor=tk.NW
                    )
                else:
                    # Обновление существующего изображения
                    self.canvas.itemconfig(self.canvas_image_id, image=new_photo)
                    # Центрирование
                    x = (canvas_w - new_w) // 2
                    y = (canvas_h - new_h) // 2
                    self.canvas.coords(self.canvas_image_id, x, y)
                
                # Сохраняем ссылку на изображение
                self.current_photo = new_photo
            
            # Выполняем обновление в главном потоке
            self.window.after(0, update_image)
            
        except Exception as e:
            print(f"Ошибка обновления canvas: {e}")
    
    def on_closing(self):
        """Обработка закрытия окна"""
        self.stop_video()
        self.window.destroy()


def main():
    """Главная функция запуска приложения"""
    root = tk.Tk()
    app = VideoProcessorApp(root)
    
    # Центрирование окна на экране
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()