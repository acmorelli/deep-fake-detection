import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
import mediapipe as mp


video_path = r"C:\Users\revan\OneDrive\Desktop\COMPUTER VISION\DATASET\DeepfakeTIMIT\fadg0-original.mov"
cap = cv2.VideoCapture(video_path)

# ROI  for the forehead, left and right cheek
forehead_landmarks = [107, 66, 69, 109, 10, 338, 299, 296, 336, 9]
left_cheek_landmarks = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
right_cheek_landmarks = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# get the average RGB values from the ROIs
def get_avg_rgb(frame, landmarks):
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in range(len(landmarks))], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    avg_color_per_row = np.average(masked_frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color 

# apply Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Processing each frame
fps = cap.get(cv2.CAP_PROP_FPS)
rgb_values = {'forehead': [], 'left_cheek': [], 'right_cheek': []}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            rgb_values['forehead'].append(get_avg_rgb(frame, [landmarks[i] for i in forehead_landmarks]))
            rgb_values['left_cheek'].append(get_avg_rgb(frame, [landmarks[i] for i in left_cheek_landmarks]))
            rgb_values['right_cheek'].append(get_avg_rgb(frame, [landmarks[i] for i in right_cheek_landmarks]))

cap.release()

# Convert collected RGB values to NumPy arrays
rgb_values['forehead'] = np.array(rgb_values['forehead'])
rgb_values['left_cheek'] = np.array(rgb_values['left_cheek'])
rgb_values['right_cheek'] = np.array(rgb_values['right_cheek'])

# Average the RGB values across the three ROIs
avg_rgb = np.mean([rgb_values['forehead'], rgb_values['left_cheek'], rgb_values['right_cheek']], axis=0)

# Extract R, G, B channels
R = avg_rgb[:, 0]
G = avg_rgb[:, 1]
B = avg_rgb[:, 2]

# Apply Butterworth bandpass filter
lowcut = 0.65
highcut = 4.0
R_filtered = butter_bandpass_filter(R, lowcut, highcut, fps)
G_filtered = butter_bandpass_filter(G, lowcut, highcut, fps)
B_filtered = butter_bandpass_filter(B, lowcut, highcut, fps)

# filtered signals to create the rPPG signal
rPPG_signal = (R_filtered + G_filtered + B_filtered) / 3.0

# heart rate using power spectrum analysis
frequencies, power_spectrum = welch(rPPG_signal, fs=fps, nperseg=len(rPPG_signal)//2)
peak_freq = frequencies[np.argmax(power_spectrum)]
bpm = peak_freq * 60.0

print(f"Estimated BPM: {bpm}")

# Plotting the power spectrum
plt.plot(frequencies, power_spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectrum')
plt.title('Power Spectrum of rPPG Signal')
plt.show()





