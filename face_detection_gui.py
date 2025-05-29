import sys
sys.path.append('..')
import logging
import logging.config
logging.config.fileConfig("../config/logging.conf")
logger = logging.getLogger('api')

import cv2
import numpy as np
import yaml
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
import face_detect
import face_crop
import face_alignment
import face_feature
import face_pipline

# Ładowanie konfiguracji modelu
with open('../config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

# Inicjalizacja modelu detekcji
model_path = '../models'
scene = 'non-mask'
model_category = 'face_detection'
model_name = model_conf[scene][model_category]
logger.info('Start to load the face detection model...')
try:
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    model, cfg = faceDetModelLoader.load_model()
    device = "cpu"
    faceDetModelHandler = FaceDetModelHandler(model, device, cfg)
except Exception as e:
    logger.error('Failed to load face detection model!')
    logger.error(e)
    sys.exit(-1)

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection GUI")
        self.root.geometry("800x600")

        # Zmienne
        self.cap = None
        self.camera_running = False
        self.video_running = False
        self.image = None
        self.frame_counter = 0  # Licznik klatek do rzadziej detekcji

        # UI Elements
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack(pady=10)

        self.start_button = tk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.load_image_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT, padx=10)

        self.load_video_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_video_button.pack(side=tk.LEFT, padx=10)

        self.quit_button = tk.Button(root, text="Quit", command=self.quit)
        self.quit_button.pack(side=tk.LEFT, padx=10)

        # Start pętli aktualizacji
        self.update_frame()

    def start_camera(self):
        if not self.camera_running and not self.video_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Cannot open camera!")
                return
            self.camera_running = True
            self.start_button.config(state=tk.DISABLED)
            self.load_video_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            logger.info("Camera started.")

    def load_video(self):
        if not self.camera_running and not self.video_running:
            file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
            if file_path:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    logger.error("Cannot open video file!")
                    return
                self.video_running = True
                self.start_button.config(state=tk.DISABLED)
                self.load_video_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                logger.info(f"Video loaded: {file_path}")

    def stop(self):
        if self.camera_running or self.video_running:
            if self.cap is not None:
                self.cap.release()
            self.camera_running = False
            self.video_running = False
            self.start_button.config(state=tk.NORMAL)
            self.load_video_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.canvas.delete("all")
            logger.info("Stopped camera or video.")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.stop()
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.process_and_display_image(self.image)
                logger.info(f"Image loaded: {file_path}")

    def process_and_display_image(self, frame, detect=True):
        # Skalowanie klatki przed detekcją (max 640x480)
        h, w = frame.shape[:2]
        scale = min(640 / w, 480 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Detekcja co 5 klatek (lub dla obrazu zawsze)
        if detect and (self.frame_counter % 1 == 0 or not (self.camera_running or self.video_running)):
            try:
                dets = faceDetModelHandler.inference_on_image(resized_frame)
                for i in range(dets.shape[0]):
                    bbox = dets[i, :4].astype(int)
                    score = dets[i, 4]
                    if score > 0.5:
                        cv2.rectangle(resized_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            except Exception as e:
                logger.error("Detection failed!")
                logger.error(e)

        # Skalowanie do canvasu
        canvas_w, canvas_h = 640, 480
        final_scale = min(canvas_w / new_w, canvas_h / new_h)
        final_w, final_h = int(new_w * final_scale), int(new_h * final_scale)
        final_frame = cv2.resize(resized_frame, (final_w, final_h), interpolation=cv2.INTER_AREA)

        # Wyświetlanie
        frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def update_frame(self):
        if self.camera_running or self.video_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_counter += 1
                self.process_and_display_image(frame, detect=True)
            elif self.video_running:  # Koniec filmu
                self.stop()
        self.root.after(33, self.update_frame)  # ~30 FPS

    def quit(self):
        self.stop()
        self.root.quit()
        logger.info("Application closed.")
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()