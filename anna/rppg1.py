# %%

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import scipy.signal 
from scipy.signal import find_peaks
from multiprocessing import Pool
import butterworth_filter
from sklearn.preprocessing import MinMaxScaler
# %%
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define function to extract ROI using landmark indices
def extract_roi(image, landmarks, indices):
    points = np.array([(landmarks.landmark[idx].x * image.shape[1], landmarks.landmark[idx].y * image.shape[0]) for idx in indices], np.int32)
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    cropped = image[y:y+h, x:x+w].copy()

    # Create a mask for the ROI
    points = points - points.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, points, 255)

    # Apply the mask to the cropped ROI
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    return result

# %%
def apply_FIR_filter(
        signal,
        fps,
        filter_size=1.5,        # filter length = 2 seconds
        min_f=0.75,           # 0.75 Hz = 45 bpm minimum heart rate
        max_f=3.3            # 3.3 Hz = 198 bpm maximum heart rate
):
    FIR_filter = scipy.signal.firwin(
        numtaps=int(filter_size * fps),
        cutoff=[min_f * 2 / fps, max_f * 2 / fps],
        window='hamming',
        pass_zero=False)
    filtered_signal = np.convolve(
        signal,
        FIR_filter,
        mode='valid')
    return filtered_signal
# %%
def compute_average_color(image):
    # Calculate the mean of each channel
    mean_vals = cv2.mean(image)[:3]
    return np.array(mean_vals)

# %%
# %%
# Open the video file - example subject 3
video_path = '/Users/annaclara/Documents/TechLabAachen/Project/evm-sourcecode/vidOut.avi'

#evm_video=butterworth_filter.start(video_path, 5, 48, 0.75, 3.3, linearAttenuation=True, chromAttenuation=False)
# issue kernel dies / debugger exits with no logs
# alternative: run original code through and input the magnified video
# %%
# Landmark indices for forehead and cheeks (Google)
forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356]
left_cheek_indices = [234, 93, 132, 58, 172, 136, 150]
right_cheek_indices = [454, 323, 361, 288, 397, 365, 379]
rppg_signal = []
evm_video = cv2.VideoCapture(video_path)
fps = evm_video.get(cv2.CAP_PROP_FPS)

# %%# Loop through the magnified frames
while True:
    ret, frame = evm_video.read()

    # Check if the frame was read correctly
    if not ret:
        print("Reached end of video or error reading frame.")
        break

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract ROIs
            forehead = extract_roi(frame, face_landmarks, forehead_indices)
            left_cheek = extract_roi(frame, face_landmarks, left_cheek_indices)
            right_cheek = extract_roi(frame, face_landmarks, right_cheek_indices)

            forehead_avg = compute_average_color(forehead) # BGR
            left_cheek_avg = compute_average_color(left_cheek)
            right_cheek_avg = compute_average_color(right_cheek)

            # Compute rPPG
            B_arr = [forehead_avg[0], left_cheek_avg[0], right_cheek_avg[0]]
            G_arr = [forehead_avg[1], left_cheek_avg[1], right_cheek_avg[1]]
            R_arr = [forehead_avg[2], left_cheek_avg[2], right_cheek_avg[2]]
            B = np.mean(B_arr)
            G = np.mean(G_arr)
            R = np.mean(R_arr)
            rPPG = (G / R) + (G / B)
            rppg_signal.append(rPPG)

# Release the video capture object
evm_video.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()

# Plot the rPPG signal
plt.plot(rppg_signal)
plt.xlabel('Frame')
plt.ylabel('rPPG Signal')
plt.title('rPPG Signal Over Time')
plt.show()
# %%

rppg_signal_filtered = apply_FIR_filter(rppg_signal, fps)

# Plot the original and filtered rPPG signal
plt.figure(figsize=(12, 6))
plt.plot(rppg_signal, label='Original rPPG Signal', alpha=0.5)
plt.plot(rppg_signal_filtered, label='Filtered rPPG Signal', color='red')
plt.xlabel('Frame')
plt.ylabel('rPPG Signal')
plt.title('Original and Filtered rPPG Signal Over Time')
plt.legend()
plt.show()
# %%
# Transform the filtered signal to the frequency domain
N = len(rppg_signal_filtered)
T = 1 / fps
yf = fft(rppg_signal_filtered)
xf = fftfreq(N, T)[:N//2]

# Find the frequency with the most power
idx = np.argmax(np.abs(yf[:N//2]))
dominant_frequency = xf[idx]
bpm = dominant_frequency * 60

plt.subplot(3, 1, 2)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), label='Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Domain of rPPG Signal')
plt.legend()
# %% 

fft_result = np.fft.fft(rppg_signal_filtered)
amplitude_spectrum = np.abs(fft_result)
freqs = np.fft.fftfreq(len(rppg_signal_filtered))
plt.figure(figsize=(10, 6))
plt.plot(freqs, amplitude_spectrum)
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
# %%
# Plot the original, filtered rPPG signal and BPM over time
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(rppg_signal, label='Original rPPG Signal', alpha=0.5)
plt.plot(range(window_size - 1, len(rppg_signal)), rppg_signal_filtered, label='Filtered rPPG Signal', color='red')
plt.xlabel('Frame')
plt.ylabel('rPPG Signal')
plt.title('Original and Filtered rPPG Signal Over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot([bpm] * N, label=f'Estimated BPM: {bpm:.2f}', color='blue')
plt.xlabel('Frame')
plt.ylabel('BPM')
plt.title('BPM Over Time')
plt.legend()

plt.tight_layout()
plt.show()
# %%
def compute_fft(signal, fps):
    # Compute FFT
    fft_signal = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), d=1/fps)
    
    # Only keep positive frequencies
    pos_mask = fft_freqs >= 0
    fft_freqs = fft_freqs[pos_mask]
    fft_signal = fft_signal[pos_mask]
    
    return fft_freqs, np.abs(fft_signal)

def compute_bpm_from_fft(fft_freqs, fft_signal, fps):
    # Find peaks in the FFT spectrum
    peaks, _ = find_peaks(fft_signal, height=np.max(fft_signal) * 0.5)
    
    # Get the frequency corresponding to the highest peak
    dominant_freq = fft_freqs[peaks][np.argmax(fft_signal[peaks])]
    
    # Convert frequency to BPM
    bpm = dominant_freq * 60
    
    return bpm

def plot_bpm_over_time(bpm_over_time):
    plt.figure(figsize=(12, 6))
    plt.plot(bpm_over_time, label='BPM')
    plt.xlabel('Frame Index')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('Estimated Heart Rate Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
scale= MinMaxScaler()
rppg_signal_filtered_reshape=rppg_signal_filtered.reshape(-1,1)
rppg_signal_filtered_n =scale.fit_transform(rppg_signal_filtered_reshape)
fft_freqs, fft_signal = compute_fft(rppg_signal_filtered_n, fps)
fft_signal=fft_signal.reshape(-1)
plt.plot(fft_freqs, fft_signal)
bpm = compute_bpm_from_fft(fft_freqs, fft_signal, fps)
fft_signal.sort()
plt.plot(fft_freqs, fft_signal)

# %%
