import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from itertools import zip_longest
from scipy.signal import butter, filtfilt

# Define landmark indices based on the paper by F. Haugg et al.
landmark_head = [107, 66, 69, 67, 109, 10, 338, 297, 299, 296, 336, 9]
landmark_lcheek = [118, 119, 120, 47, 126, 209, 49, 129, 203, 205, 50]
landmark_rcheek = [347, 348, 349, 277, 355, 429, 279, 358, 423, 425, 280]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Butterworth filter to smooth the signal
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

class HeartRateVideoAnalysisGUI:
    def __init__(self, root, video_list):
        self.root = root
        self.root.title("Heart Rate Estimation from Video")
        self.root.geometry("800x600")

        self.video_label = Label(root)
        self.video_label.pack()

        self.bpm_label = Label(root, text="BPM: --", font=("Arial", 32))
        self.bpm_label.pack(pady=20)

        
        self.cap = cv2.VideoCapture(0)  # Use 0 for the default camera
        if not self.cap.isOpened():
            print("Error: Camera could not be opened.")
            self.root.quit()  # Exit if camera can't be opened

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30 

        # Initialize data collection for BPM estimation
        self.avg_b = []
        self.avg_g = []
        self.avg_r = []
        self.grgb_signal = []  
        self.bpm = 0

        self.frame_count = 0  # Initialize frame counter

        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return 

        # Check if frame is empty
        if frame is None or frame.size == 0:
            self.root.quit()
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        # Extract the values of specific ROI with respect to landmark position points
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmark_coords_head = []
                landmark_coords_lcheek = []
                landmark_coords_rcheek = []

                for i, j, k in zip_longest(landmark_head, landmark_lcheek, landmark_rcheek, fillvalue=None):
                    if i is not None:
                        landmark_h = face_landmarks.landmark[i]
                        x = int(landmark_h.x * w)
                        y = int(landmark_h.y * h)
                        landmark_coords_head.append((x, y))
                    
                    if j is not None:
                        landmark_lc = face_landmarks.landmark[j]
                        x1 = int(landmark_lc.x * w)
                        y1 = int(landmark_lc.y * h)
                        landmark_coords_lcheek.append((x1, y1))
                    
                    if k is not None:
                        landmark_rc = face_landmarks.landmark[k]
                        x2 = int(landmark_rc.x * w)
                        y2 = int(landmark_rc.y * h)
                        landmark_coords_rcheek.append((x2, y2))
                
                #forehead 
                x_coords, y_coords = zip(*landmark_coords_head)
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
                forehead_roi = frame[y_min:y_max, x_min:x_max]
                forehead_red = np.mean(forehead_roi[:, :, 2])
                forehead_green = np.mean(forehead_roi[:, :, 1])
                forehead_blue = np.mean(forehead_roi[:, :, 0])

                #left cheek 
                x_coords, y_coords = zip(*landmark_coords_lcheek)
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
                lcheek_roi = frame[y_min:y_max, x_min:x_max]
                lcheek_red = np.mean(lcheek_roi[:, :, 2])
                lcheek_green = np.mean(lcheek_roi[:, :, 1])
                lcheek_blue = np.mean(lcheek_roi[:, :, 0])

                #right cheek 
                x_coords, y_coords = zip(*landmark_coords_rcheek)
                x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
                rcheek_roi = frame[y_min:y_max, x_min:x_max]
                rcheek_red = np.mean(rcheek_roi[:, :, 2])
                rcheek_green = np.mean(rcheek_roi[:, :, 1])
                rcheek_blue = np.mean(rcheek_roi[:, :, 0])

                #avg values
                self.avg_r.append(np.mean([forehead_red, lcheek_red, rcheek_red]))
                self.avg_g.append(np.mean([forehead_green, lcheek_green, rcheek_green]))
                self.avg_b.append(np.mean([forehead_blue, lcheek_blue, rcheek_blue]))

                # Calculate the GR and GB signals
                if len(self.avg_r) > 1:
                    gr_signal = [g/r for g, r in zip(self.avg_g, self.avg_r)]
                    gb_signal = [g/b for g, b in zip(self.avg_g, self.avg_b)]
                    self.grgb_signal = [gr + gb for gr, gb in zip(gr_signal, gb_signal)]

        # Calculate BPM based on the GRGB signal
        if len(self.grgb_signal) > self.fps:  # Ensure we have enough data (at least one second)
            # Apply a smoothing filter to the GRGB signal
            smoothed_signal = butter_lowpass_filter(self.grgb_signal[-int(self.fps * 5):], 2.5, self.fps)

            # Apply FFT on the smoothed signal
            signal_chunk = smoothed_signal

            #FFT 
            freqs = np.fft.rfftfreq(len(signal_chunk), d=1/self.fps)
            fft_magnitude = np.abs(np.fft.rfft(signal_chunk))

            # Find the frequency with the highest magnitude in the expected heart rate range (0.8 - 2.5 Hz)
            valid_freqs = (freqs >= 0.8) & (freqs <= 2.5)
            if np.any(valid_freqs):  # Ensure there are valid frequencies within the range
                peak_freq = freqs[valid_freqs][np.argmax(fft_magnitude[valid_freqs])]
                # Convert the peak frequency to BPM
                self.bpm = int(peak_freq * 60)

           
            self.bpm_label.config(text=f"BPM: {self.bpm}")

        # Convert the frame back to BGR and display it
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(frame_bgr)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.frame_count += 1  
        self.root.after(10, self.update_video_feed)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()  
        self.root.quit()

# Define a placeholder for get_videos_info function
def get_videos_info(folder):
    return ['1_video_input/video1.mp4', '1_video_input/video2.mp4']

root = tk.Tk()
video_list = get_videos_info('1_video_input/')
app = HeartRateVideoAnalysisGUI(root, video_list)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()

face_mesh.close()
cv2.destroyAllWindows()

