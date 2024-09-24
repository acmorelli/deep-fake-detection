import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk


forehead_landmarks = [107, 66, 69, 109, 10, 338, 299, 296, 336, 9]
left_cheek_landmarks = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
right_cheek_landmarks = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def get_avg_rgb(frame, landmarks):
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in range(len(landmarks))], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    avg_color_per_row = np.average(masked_frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

#GUI and video display
class HeartRateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Rate Estimation")
        self.root.geometry("800x600")

        #Video feed label
        self.video_label = Label(root)
        self.video_label.pack()

        #BPM display label
        self.bpm_label = Label(root, text="BPM: --", font=("Arial", 24))
        self.bpm_label.pack()

        #Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Default to 30 FPS if unknown

        #Initialize data collection for rPPG
        self.rgb_values = {'forehead': [], 'left_cheek': [], 'right_cheek': []}
        self.window_size = 300  # Sliding window size for rPPG (~10 seconds)
        self.lowcut = 0.65  # Lower cutoff for heart rate frequency (BPM)
        self.highcut = 4.0  # Upper cutoff for heart rate frequency (BPM)

        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        #Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #Extract the bounding box around the face using min and max coordinates
                h, w, _ = frame.shape
                face_xmin = int(min([lm.x * w for lm in face_landmarks.landmark]))
                face_ymin = int(min([lm.y * h for lm in face_landmarks.landmark]))
                face_xmax = int(max([lm.x * w for lm in face_landmarks.landmark]))
                face_ymax = int(max([lm.y * h for lm in face_landmarks.landmark]))

                #Draw the bounding box 
                cv2.rectangle(frame, (face_xmin, face_ymin), (face_xmax, face_ymax), (0, 255, 0), 2)

                #Collect RGB values for forehead, left cheek, and right cheek
                landmarks = face_landmarks.landmark
                self.rgb_values['forehead'].append(get_avg_rgb(frame, [landmarks[i] for i in forehead_landmarks]))
                self.rgb_values['left_cheek'].append(get_avg_rgb(frame, [landmarks[i] for i in left_cheek_landmarks]))
                self.rgb_values['right_cheek'].append(get_avg_rgb(frame, [landmarks[i] for i in right_cheek_landmarks]))

                #Limit the number of stored frames (real-time window)
                if len(self.rgb_values['forehead']) > self.window_size:
                    for key in self.rgb_values:
                        self.rgb_values[key] = self.rgb_values[key][-self.window_size:]

                #If enough frames are collected, process the signal
                if len(self.rgb_values['forehead']) == self.window_size:
                    # Convert collected RGB values to NumPy arrays
                    self.rgb_values['forehead'] = np.array(self.rgb_values['forehead'])
                    self.rgb_values['left_cheek'] = np.array(self.rgb_values['left_cheek'])
                    self.rgb_values['right_cheek'] = np.array(self.rgb_values['right_cheek'])

                    #Average the RGB values across the three ROIs
                    avg_rgb = np.mean([self.rgb_values['forehead'], self.rgb_values['left_cheek'], self.rgb_values['right_cheek']], axis=0)

                    #Extract R, G, B channels
                    R = avg_rgb[:, 0]
                    G = avg_rgb[:, 1]
                    B = avg_rgb[:, 2]

                    #Apply Butterworth bandpass filter
                    R_filtered = butter_bandpass_filter(R, self.lowcut, self.highcut, self.fps)
                    G_filtered = butter_bandpass_filter(G, self.lowcut, self.highcut, self.fps)
                    B_filtered = butter_bandpass_filter(B, self.lowcut, self.highcut, self.fps)

                    #Calculate the rPPG signal (averaged filtered RGB signals)
                    rPPG_signal = (R_filtered + G_filtered + B_filtered) / 3.0

                    #Calculate heart rate using power spectrum analysis (Welch method)
                    frequencies, power_spectrum = welch(rPPG_signal, fs=self.fps, nperseg=len(rPPG_signal)//2)
                    peak_freq = frequencies[np.argmax(power_spectrum)]
                    bpm = peak_freq * 60.0

                    #Update the BPM label
                    self.bpm_label.config(text=f"BPM: {int(bpm)}")

        #Convert the frame back to BGR (for OpenCV display) and display it
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(frame_bgr)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        #Continue updating the video feed
        self.root.after(10, self.update_video_feed)

    def on_closing(self):
        self.cap.release()
        self.root.quit()


#GUI 
root = tk.Tk()
app = HeartRateGUI(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()






