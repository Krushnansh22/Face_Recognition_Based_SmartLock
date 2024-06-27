import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import pickle
import numpy as np
import DataCollection as DC
import Predict as PR

class SmartLockApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("SmartLock System")
        self.geometry("800x600")

        self.frames = {}
        for F in (RegisterPage, MainPage, VideoPage):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("RegisterPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class RegisterPage(ctk.CTkFrame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.label = ctk.CTkLabel(self, text="User Registration")
        self.label.pack(pady=10)

        self.name_entry = ctk.CTkEntry(self, placeholder_text="Enter your name")
        self.name_entry.pack(pady=10)

        self.register_button = ctk.CTkButton(self, text="Register", command=self.register_user)
        self.register_button.pack(pady=10)

    def register_user(self):
        name = self.name_entry.get()
        if name:
            self.controller.name = name
            self.controller.show_frame("MainPage")

class MainPage(ctk.CTkFrame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.add_faces_button = ctk.CTkButton(self, text="Add Faces", command=self.add_faces)
        self.add_faces_button.pack(pady=10)

        self.go_live_button = ctk.CTkButton(self, text="Go Live", command=self.go_live)
        self.go_live_button.pack(pady=10)

    def add_faces(self):
        self.controller.show_frame("VideoPage")
        self.controller.frames["VideoPage"].start_data_collection()

    def go_live(self):
        self.controller.show_frame("VideoPage")
        self.controller.frames["VideoPage"].start_prediction()

class VideoPage(ctk.CTkFrame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.video_label = ctk.CTkLabel(self)
        self.video_label.pack()

        self.back_button = ctk.CTkButton(self, text="Back", command=self.stop_video)
        self.back_button.pack(pady=10)

        self.cap = None
        self.running = False
        self.thread = None

    def start_video(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.start()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            time.sleep(0.01)

    def stop_video(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        self.controller.show_frame("MainPage")

    def start_data_collection(self):
        self.start_video()
        # Implement data collection logic
        DC.collect_and_train('Training_data','face_recognition_model.xml')

    def start_prediction(self):
        self.start_video()
        # Implement prediction logic
        PR.make_predictions('face_recognition_model.xml')

if __name__ == "__main__":
    app = SmartLockApp()
    app.mainloop()
