import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def gui(path_to_crop, path_to_image):
    root = tk.Tk()
    root.withdraw()  # Hide main window

    # Ask for two image files
    img1_path = path_to_crop
    if not img1_path:
        messagebox.showerror("Error", "First image not selected.")
        return

    img2_path = path_to_image
    if not img2_path:
        messagebox.showerror("Error", "Second image not selected.")
        return

    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        messagebox.showerror("Error", "One or both images could not be loaded.")
        return

    if img1.shape != img2.shape:
        messagebox.showerror("Error", "Images must be the same size!")
        return

    # Merge images horizontally
    merged = np.hstack((img1, img2))

    # Save output
    output_path = os.path.join(os.path.dirname(img1_path), "testowanie_merged.jpg")
    cv2.imwrite(output_path, merged)
    

    # Show the result
    return output_path
