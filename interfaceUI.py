import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
import os

# Импорт ваших модулей
from video_capture import advanced_preprocessing
from detect_point import YOLOPoseDetector


class VideoProcessorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Video Pose Detection")
        self.window.geometry("800x650")  # Немного увеличил высоту для новой кнопки
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
        self.target_fps = 15
        self.frame_time = 1.0 / self.target_fps
        self.last_process_time = time.time()
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_res = 0
        
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
        """
        Функция-заглушка для калибровки
        В будущем здесь будет логика калибровки положения человека
        """
        if self.is_calibrating:
            return
        
        self.is_calibrating = True
        self.calibrate_btn.config(state=tk.DISABLED, text="⏳ КАЛИБРОВКА...")
        
        # Имитация процесса калибровки в отдельном потоке
        def calibration_process():
            time.sleep(2)  # Имитация процесса калибровки
            
            # Заглушка данных калибровки
            self.calibration_data = {
                'eye_level': 100,
                'shoulder_level': 200,
                'tilt_threshold': 40,
                'calibration_time': time.strftime("%H:%M:%S")
            }
            
            # Обновление интерфейса в главном потоке
            self.window.after(0, self._finish_calibration)
        
        # Запуск калибровки в отдельном потоке
        calib_thread = threading.Thread(target=calibration_process)
        calib_thread.daemon = True
        calib_thread.start()
        
        # Временное обновление статуса
        self.status_label.config(text="Статус: Калибровка...")
    
    def _finish_calibration(self):
        """Завершение калибровки и обновление интерфейса"""
        self.is_calibrating = False
        self.calibrate_btn.config(state=tk.NORMAL, text="⚙ КАЛИБРОВКА")
        
        # Обновление метки калибровки
        calib_time = self.calibration_data.get('calibration_time', 'неизвестно')
        self.calibration_label.config(
            text=f"✓ Калибровка выполнена в {calib_time} | Порог: {self.calibration_data['tilt_threshold']}px",
            foreground='green'
        )
        
        # Обновление статуса
        if self.is_running:
            self.status_label.config(text="Статус: Работает")
        else:
            self.status_label.config(text="Статус: Остановлен")
        
        print(f"[КАЛИБРОВКА] Завершена. Данные: {self.calibration_data}")
    
    def start_video(self):
        """Запуск видео потока"""
        if self.is_running:
            return
            
        # Инициализация камеры
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
            self.detector = YOLOPoseDetector(
                model_path='yolov8n-pose.pt', 
                confidence_threshold=0.5
            )
        except Exception as e:
            self.status_label.config(text=f"Статус: Ошибка загрузки модели")
            print(f"Ошибка загрузки модели: {e}")
            self.cap.release()
            return
        
        # Сброс счетчиков времени
        self.last_process_time = time.time()
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_res = 0
        
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
        while self.is_running:
            try:
                flag, img = self.cap.read()
                
                if not flag:
                    print("Не удалось получить кадр с камеры")
                    break
                
                current_time = time.time()
                
                # Ограничение FPS
                if current_time - self.last_process_time < self.frame_time:
                    # Отображение без обработки
                    img = cv2.flip(img, 1)
                    self.update_canvas(img)
                    continue
                
                self.last_process_time = current_time
                
                # Расчет FPS
                self.frame_count += 1
                if current_time - self.prev_time >= 1.0:
                    self.fps_res = self.frame_count / (current_time - self.prev_time)
                    self.frame_count = 0
                    self.prev_time = current_time
                
                # Обработка изображения
                img = cv2.flip(img, 1)
                
                # Ваша обработка
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                processed_pil = advanced_preprocessing(pil_img)
                processed_img = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
                
                # Детекция позы
                self.detector.process_image(processed_img, draw_boxes=False)
                
                all_points_eye, all_points_shoulders = self.detector.get_keypoints_by_indices(
                    processed_img, indices=[3, 4, 5, 6]
                )
                signal_person = self.detector.analyze_slouch(
                    all_points_eye, all_points_shoulders
                )
                
                if len(signal_person) != 0:
                    self.detector.signal(signal_person)
                
                # Добавление информации на изображение
                if self.fps_res > 0:
                    cv2.putText(processed_img, f"FPS: {self.fps_res:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                
                # Отображение статуса калибровки на видео
                if self.calibration_data.get('calibration_time'):
                    cv2.putText(processed_img, "Calibrated", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (255, 255, 0), 1)
                
                # Обновление отображения
                self.update_canvas(processed_img)
                
                # Обновление FPS в интерфейсе
                if self.fps_res > 0:
                    self.window.after(0, lambda: self.fps_label.config(
                        text=f"FPS: {self.fps_res:.1f}"
                    ))
                
            except Exception as e:
                print(f"Ошибка обработки: {e}")
                break
        
        # Если цикл завершился, останавливаем видео
        if self.is_running:
            self.window.after(0, self.stop_video)
    
    def update_canvas(self, img):
        """Обновление изображения в canvas"""
        try:
            # Конвертация BGR в RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Изменение размера под canvas
            h, w = img.shape[:2]
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w > 1 and canvas_h > 1:  # Если canvas уже имеет размер
                # Сохранение пропорций
                scale = min(canvas_w / w, canvas_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_resized = cv2.resize(img_rgb, (new_w, new_h))
            else:
                img_resized = img_rgb
            
            # Конвертация в PhotoImage
            pil_image = Image.fromarray(img_resized)
            self.photo = ImageTk.PhotoImage(image=pil_image)
            
            # Обновление canvas в главном потоке
            self.window.after(0, self._update_canvas_image)
            
        except Exception as e:
            print(f"Ошибка обновления canvas: {e}")
    
    def _update_canvas_image(self):
        """Вспомогательный метод для обновления canvas"""
        self.canvas.delete("all")
        # Центрирование изображения
        x = (self.canvas.winfo_width() - self.photo.width()) // 2
        y = (self.canvas.winfo_height() - self.photo.height()) // 2
        self.canvas.create_image(x, y, image=self.photo, anchor=tk.NW)
    
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