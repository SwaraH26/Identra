"""
Face detection and preprocessing module.
Handles face localization, margin expansion, cropping, and normalization.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import config

class FaceDetector:
    """
    Robust face detection using OpenCV Haar Cascades and image preprocessing.
    """
    def __init__(self, min_size: Tuple[int, int] = config.FACE_MIN_SIZE):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {cascade_path}")
        self.min_size = min_size

    def detect_faces(self, image_bgr: np.ndarray, 
                     scale_factor: float = 1.1, 
                     min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Detects all faces in an input BGR image.
        Returns a list of bounding boxes: [(x, y, w, h), ...]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization to normalize lighting (FR4 / Challenge in PRD)
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return [tuple(f) for f in faces]

    def extract_face(self, image_bgr: np.ndarray, 
                     bbox: Optional[Tuple[int, int, int, int]] = None,
                     padding_ratio: float = config.FACE_PADDING_RATIO,
                     target_size: Tuple[int, int] = config.FACE_INPUT_SIZE) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crops and standardizes the face region with margin.
        If bbox is None, automatically detects the largest face.
        Returns:
            (face_rgb_normalized, (x, y, w, h))
        """
        h_img, w_img = image_bgr.shape[:2]

        if bbox is None:
            faces = self.detect_faces(image_bgr)
            if not faces:
                # If no face is detected with standard settings, try more relaxed parameters
                faces = self.detect_faces(image_bgr, scale_factor=1.05, min_neighbors=3)
            
            if not faces:
                raise ValueError("No face detected in the provided image.")

            # Pick largest face by area
            bbox = max(faces, key=lambda b: b[2] * b[3])

        x, y, w, h = bbox

        # Add contextual margin
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        cropped_bgr = image_bgr[y1:y2, x1:x2]
        if cropped_bgr.size == 0:
            raise ValueError("Extracted face crop is empty.")

        # Resize to standardized deep learning input size
        resized_bgr = cv2.resize(cropped_bgr, target_size, interpolation=cv2.INTER_AREA)

        # Convert to RGB (expected by PyTorch Vision backbones)
        face_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        return face_rgb, bbox

    def draw_face_box(self, image_bgr: np.ndarray, 
                      bbox: Tuple[int, int, int, int], 
                      label: Optional[str] = None, 
                      score: Optional[float] = None,
                      is_accepted: Optional[bool] = None) -> np.ndarray:
        """
        Visual annotation utility to draw bounding boxes and informative tags.
        """
        img_out = image_bgr.copy()
        x, y, w, h = bbox

        # Color coding: Green for Accepted, Red for Rejected, Blue for Neutral/Detected
        if is_accepted is True:
            color = (40, 200, 40)  # Green
        elif is_accepted is False:
            color = (40, 40, 230)  # Red
        else:
            color = (255, 160, 30)  # Cyan/Blue

        # Draw rounded rectangle bounding box
        cv2.rectangle(img_out, (x, y), (x + w, y + h), color, 2)

        # Build label text
        text = label if label else "Face"
        if score is not None:
            text += f" ({score:.1%})"

        # Header background banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        banner_y1 = max(0, y - th - 8)
        banner_y2 = y
        cv2.rectangle(img_out, (x, banner_y1), (x + tw + 10, banner_y2), color, -1)
        cv2.putText(img_out, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return img_out

