import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

mp_face_mesh = mp.solutions.face_mesh

def bandpass_filter(signal, lowcut=0.75, highcut=3.5, fs=30, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def estimate_hr_from_signal(signal, fps=30):
    if len(signal) < fps:
        return -1
    signal = np.array(signal) - np.mean(signal)
    filtered = bandpass_filter(signal, fs=fps)
    fft_vals = np.abs(rfft(filtered))
    freqs = rfftfreq(len(filtered), 1 / fps)
    valid = (freqs >= 0.75) & (freqs <= 3.5)
    if not np.any(valid):
        return -1
    dominant_freq = freqs[valid][np.argmax(fft_vals[valid])]
    return dominant_freq * 60

def extract_forehead_mean(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        points = [lm[i] for i in [10, 338, 297]]
        coords = [(int(p.x * w), int(p.y * h)) for p in points]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(coords, np.int32), 255)
        green = np.mean(frame[:, :, 1][mask == 255])
        red = np.mean(frame[:, :, 2][mask == 255])
        return green, red
    return None, None

def resample_to_30fps(input_path, output_path, target_fps=30):
    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == target_fps:
        cap.release()
        return input_path

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    duration = len(frames) / original_fps
    new_count = int(duration * target_fps)
    indices = np.linspace(0, len(frames) - 1, new_count).astype(int)

    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))
    for i in indices:
        out.write(frames[i])
    out.release()
    return output_path

def extract_features(video_path, fps=30, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    green_values, red_values = [], []
    frame_count = 0
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames to process only every 3rd frame
            if frame_count % 3 != 0:
                frame_count += 1
                continue
                
            green, red = extract_forehead_mean(frame, face_mesh)
            if green is not None and red is not None:
                green_values.append(green)
                red_values.append(red)
                
            frame_count += 1
            
            # Early stopping if we have enough good frames
            if len(green_values) >= 50:
                break
                
    cap.release()

    if not green_values:
        raise ValueError("No valid frames with face landmarks")

    # Use only the first 50 frames if we have more
    if len(green_values) > 50:
        green_values = green_values[:50]
        red_values = red_values[:50]

    green_mean = np.mean(green_values)
    green_std = np.std(green_values)
    red_mean = np.mean(red_values)
    red_std = np.std(red_values)
    est_hr = estimate_hr_from_signal(green_values, fps)

    best_mean = green_mean if green_std < red_std else red_mean
    best_std = green_std if green_std < red_std else red_std

    return {
        'green_mean': green_mean,
        'green_std': green_std,
        'red_mean': red_mean,
        'red_std': red_std,
        'estimated_hr': est_hr,
        'best_mean': best_mean,
        'best_std': best_std
    }
