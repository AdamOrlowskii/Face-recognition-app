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
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
import face_detect
import face_crop
import face_alignment
import face_feature
import face_pipline
import face_merge
import os

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
        self.root.geometry("1200x800")

        # Zmienne
        self.cap = None
        self.camera_running = False
        self.video_running = False
        self.image = None
        self.frame_counter = 0  # Licznik klatek do rzadziej detekcji

        # UI Elements
        self.video_frame = tk.Frame(root)
        self.video_frame.pack()

        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack(side=tk.LEFT, padx=10)

        self.image_canvas = tk.Canvas(self.video_frame, width=320, height=240)
        self.image_canvas.pack(side=tk.RIGHT, padx=10)

        self.start_button = tk.Button(root, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.save_image_button = tk.Button(root, text="Save Image", command=self.save_current_image)
        self.save_image_button.pack(side=tk.LEFT, padx=10)


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
    
    def save_current_image(self):
        try:
            # Temporarily pause video/camera stream
            was_running = self.camera_running or self.video_running
            self.stop()

            if not self.canvas.image:
                logger.warning("No image to save!")
                messagebox.showwarning("Warning", "No image to save!")
                return

            # Ask user for save preference
            save_dialog = tk.Toplevel()
            save_dialog.title("Save Image")
            save_dialog.geometry("300x150")
            
            save_type = tk.StringVar(value="original")
            
            tk.Label(save_dialog, text="Select save option:").pack(pady=5)
            tk.Radiobutton(save_dialog, text="Save original image", variable=save_type, value="original").pack()
            tk.Radiobutton(save_dialog, text="Save cropped face", variable=save_type, value="cropped").pack()
            
            def on_save():
                try:
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".jpg",
                        filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
                    )
                    
                    if not file_path:
                        return

                    # Access the image that was displayed on the canvas
                    pil_image = ImageTk.getimage(self.canvas.image)
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                    if save_type.get() == "original":
                        # Save original image
                        cv2.imwrite(file_path, cv_image)
                        logger.info(f"Original image saved: {file_path}")
                    else:
                        # Save cropped face
                        temp_path = 'test_images/temp_for_crop.jpg'
                        cv2.imwrite(temp_path, cv_image)
                        
                        try:
                            # Run face detection
                            detect_result = face_detect.gui(temp_path)
                            
                            if os.path.exists('test_images/testowanie_res.txt'):
                                with open('test_images/testowanie_res.txt', 'r') as f:
                                    face_data = f.readlines()
                                
                                if len(face_data) > 0:
                                    # Run face alignment and get landmarks
                                    face_alignment.gui(temp_path, 'test_images/testowanie_res.txt')
                                    # Crop the face
                                    cropped_path = face_crop.gui(temp_path, 'test_images/testowanie_landmark_res0.txt')
                                    
                                    if os.path.exists(cropped_path):
                                        # Read the cropped image and save it to the user-selected path
                                        cropped_image = cv2.imread(cropped_path)
                                        cv2.imwrite(file_path, cropped_image)
                                        logger.info(f"Cropped face saved: {file_path}")
                                    else:
                                        raise Exception("Failed to crop face")
                                else:
                                    raise Exception("No face detected in image")
                            else:
                                raise Exception("Face detection failed")
                                
                        except Exception as e:
                            logger.error(f"Failed to save cropped face: {str(e)}")
                            messagebox.showerror("Error", f"Failed to save cropped face: {str(e)}")
                            # Fallback to saving original image
                            cv2.imwrite(file_path, cv_image)
                            logger.info(f"Fallback: Original image saved: {file_path}")
                        
                        finally:
                            # Cleanup temporary files
                            temp_files = [
                                'test_images/temp_for_crop.jpg',
                                'test_images/testowanie_res.txt',
                                'test_images/testowanie_landmark_res0.txt',
                                'test_images/testowanie_crop.jpg',
                                'test_images/testowanie_detect.jpg'
                            ]
                            for temp_file in temp_files:
                                if os.path.exists(temp_file):
                                    try:
                                        os.remove(temp_file)
                                    except:
                                        pass
                    
                    save_dialog.destroy()
                
                except Exception as e:
                    logger.error(f"Failed to save image: {str(e)}")
                    messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            
            tk.Button(save_dialog, text="Save", command=on_save).pack(pady=10)
            save_dialog.transient(self.root)
            save_dialog.grab_set()
            self.root.wait_window(save_dialog)

        except Exception as e:
            logger.error(f"Unexpected error in save_current_image: {str(e)}")
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
        
        finally:
            # Resume stream if it was running
            if was_running:
                if self.cap is not None and self.cap.isOpened():
                    if self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):
                        self.camera_running = True
                    else:
                        self.video_running = True
                self.start_button.config(state=tk.NORMAL)
                self.load_video_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.NORMAL)

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
            self.image = cv2.imread(file_path)
            if self.image is not None:
                try:
                    cv2.imwrite('test_images/reference.jpg', self.image)
                    dets = faceDetModelHandler.inference_on_image(self.image)
                    for i in range(dets.shape[0]):
                        bbox = dets[i, :4].astype(int)
                        score = dets[i, 4]
                        if score > 0.5:
                            cv2.rectangle(self.image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                except Exception as e:
                    logger.error("Detection on image failed!")
                    logger.error(e)

                self.display_image_on_canvas(self.image)
                logger.info(f"Image loaded: {file_path}")

    def process_and_display_image(self, frame, detect=True):
        # Skalowanie klatki przed detekcją (max 640x480)
        h, w = frame.shape[:2]
        scale = min(640 / w, 480 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Only run detection every 3 frames to reduce lag
        should_detect = detect and (self.frame_counter % 3 == 0 or not (self.camera_running or self.video_running))
        
        if should_detect:
            # 💾 Save current frame - only when we're actually going to process it
            cv2.imwrite('test_images/testowanie.jpg', resized_frame)

            try:
                # Run face detection first and get the result paths
                detect_result = face_detect.gui('test_images/testowanie.jpg')
                
                # Check if the detection result file exists and contains face data
                if os.path.exists('test_images/testowanie_res.txt'):
                    with open('test_images/testowanie_res.txt', 'r') as f:
                        face_data = f.readlines()
                    
                    if len(face_data) > 0:  # We have detected faces
                        try:
                            face_alignment.gui('test_images/testowanie.jpg', 'test_images/testowanie_res.txt')
                            face_crop.gui('test_images/testowanie.jpg', 'test_images/testowanie_landmark_res0.txt')
                            face_merge.gui('test_images/testowanie_crop.jpg', 'test_images/reference.jpg')
                            score = face_pipline.gui('test_images/testowanie_merged.jpg')

                            # Store the last valid score
                            self.last_score = score
                            self.last_score_frame = self.frame_counter

                            # 🎯 Add score on the image
                            cv2.putText(
                                resized_frame,
                                f"Score: {score:.2f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 0),
                                2,
                                cv2.LINE_AA
                            )
                        except Exception as e:
                            logger.error(f"Error in face processing pipeline: {str(e)}")
                            cv2.putText(
                                resized_frame,
                                "Processing error",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA
                            )
                    else:
                        # No faces detected in the frame
                        self.last_score = None
                        cv2.putText(
                            resized_frame,
                            "No face detected",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA
                        )
                else:
                    # Detection result file doesn't exist
                    self.last_score = None
                    cv2.putText(
                        resized_frame,
                        "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
            except Exception as e:
                logger.error(f"Error in face detection: {str(e)}")
                cv2.putText(
                    resized_frame,
                    "Detection error",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
        else:
            # For frames where we don't detect, show the last known score if it's recent enough
            if hasattr(self, 'last_score') and self.last_score is not None:
                if self.frame_counter - self.last_score_frame < 30:  # Show last score for ~1 second
                    cv2.putText(
                        resized_frame,
                        f"Score: {self.last_score:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

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

    
    def display_image_on_canvas(self, image):
        h, w = image.shape[:2]
        scale = min(320 / w, 240 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.image_canvas.image = img_tk


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