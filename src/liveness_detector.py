"""
Anti-spoofing and Liveness Detection Module.
Detects presentation attacks (printed photos, phone screen replays)
using texture analysis, Laplacian frequency variance, and color spectrum inspection.
(Directly addresses PRD Section 4 Scope & Section 13 Risks & Challenges).
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Any
import config

class LivenessDetector:
    """
    Lightweight, real-time passive liveness detector.
    Evaluates micro-texture sharpness and frequency distribution to reject flat/printed attacks.
    """
    def __init__(self, sharpness_threshold: float = config.LIVENESS_LAPLACIAN_THRESHOLD):
        self.sharpness_threshold = sharpness_threshold

    def evaluate_liveness(self, face_rgb: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluates whether a cropped face is from a live individual or a static photo/screen.
        Returns:
            (is_live: bool, liveness_score: float [0..1], diagnostics: dict)
        """
        if face_rgb is None or face_rgb.size == 0:
            return False, 0.0, {"error": "Empty face crop"}

        # 1. Texture Sharpness via Laplacian Variance
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalize variance score: typical live faces have laplacian variance > 60-100
        # Flat screen photos or printed papers often have low variance or blur
        sharpness_score = float(np.clip(laplacian_var / 150.0, 0.0, 1.0))

        # 2. High Frequency Energy via Fast Fourier Transform (FFT)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # High frequency content ratio
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        # Mask out center (low frequencies)
        radius = min(rows, cols) // 8
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)
        high_freq_energy = float(np.mean(magnitude_spectrum[mask == 1]))
        high_freq_score = float(np.clip(high_freq_energy / 80.0, 0.0, 1.0))

        # 3. Color distribution balance (Color space analysis)
        # Recaptured screens often show unnatural color saturation
        hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        sat_score = 1.0 if (20.0 <= sat_mean <= 190.0) else 0.5

        # Combined Liveness Score
        combined_score = (0.5 * sharpness_score) + (0.3 * high_freq_score) + (0.2 * sat_score)
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        # Decision
        is_live = (laplacian_var >= self.sharpness_threshold) and (combined_score >= 0.45)

        diagnostics = {
            "laplacian_variance": round(laplacian_var, 2),
            "sharpness_score": round(sharpness_score, 3),
            "high_freq_energy": round(high_freq_energy, 2),
            "combined_score": round(combined_score, 3),
            "is_live": is_live
        }

        return is_live, combined_score, diagnostics

