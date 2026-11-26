"""
Sample Data and Synthetic Benchmark Generator.
Creates sample registered identities and test face pairs (genuine and impostor)
to enable immediate benchmarking, evaluation, and demonstration without requiring manual webcam shots upfront.
"""
import os
import cv2
import numpy as np
import skimage.data as skdata
from pathlib import Path
from typing import Dict, List, Tuple
import config

def apply_face_variations(base_face_rgb: np.ndarray, num_variations: int = 6) -> List[np.ndarray]:
    """
    Applies realistic biometric variations:
    1. Neutral
    2. Brightness variation (High key lighting)
    3. Low light variation (Dim environment)
    4. Minor angle/rotation (+/- 5 degrees)
    5. Subtle camera blur / sensor noise
    6. Slight zoom/crop scale
    """
    variations = [base_face_rgb.copy()]
    h, w = base_face_rgb.shape[:2]

    # Variation 1: Brightness boost
    bright = cv2.convertScaleAbs(base_face_rgb, alpha=1.2, beta=25)
    variations.append(bright)

    # Variation 2: Dim lighting
    dim = cv2.convertScaleAbs(base_face_rgb, alpha=0.8, beta=-20)
    variations.append(dim)

    # Variation 3: Slight clockwise rotation
    m_rot1 = cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1.0)
    rot1 = cv2.warpAffine(base_face_rgb, m_rot1, (w, h), borderMode=cv2.BORDER_REFLECT)
    variations.append(rot1)

    # Variation 4: Slight counter-clockwise rotation
    m_rot2 = cv2.getRotationMatrix2D((w / 2, h / 2), -4, 1.0)
    rot2 = cv2.warpAffine(base_face_rgb, m_rot2, (w, h), borderMode=cv2.BORDER_REFLECT)
    variations.append(rot2)

    # Variation 5: Camera sensor noise + gentle blur
    noise = np.random.normal(0, 5, base_face_rgb.shape).astype(np.float32)
    noisy = np.clip(base_face_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    noisy = cv2.GaussianBlur(noisy, (3, 3), 0.5)
    variations.append(noisy)

    return variations[:num_variations]


def create_derived_identity(base_face_rgb: np.ndarray, identity_seed: int) -> np.ndarray:
    """
    Derives a distinct synthetic facial appearance profile from a base face
    by applying tone mapping, facial morphology adjustments, and color shifts.
    """
    np.random.seed(identity_seed)
    hsv = cv2.cvtColor(base_face_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Alter hue and saturation profile
    hue_shift = (identity_seed * 37) % 180
    sat_scale = 0.8 + ((identity_seed % 5) * 0.1)
    val_scale = 0.85 + (((identity_seed + 2) % 4) * 0.1)

    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)

    modified_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Add subtle structural warping
    h, w = base_face_rgb.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dx = ((identity_seed % 3) - 1) * 6
    dy = (((identity_seed // 2) % 3) - 1) * 6
    pts2 = np.float32([[dx, dy], [w - dx, -dy], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped_rgb = cv2.warpPerspective(modified_rgb, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    return warped_rgb


def generate_benchmark_dataset(target_dir: Path = config.SAMPLE_DATA_DIR, num_identities: int = 6) -> Dict[str, Any]:
    """
    Creates a multi-identity sample directory with enrollment and test images.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base human face from skimage
    astronaut = skdata.astronaut()  # RGB Eileen Collins
    # Crop face region: [66:161, 177:272]
    base_face = cv2.resize(astronaut[60:175, 170:280], (224, 224))

    identity_names = [
        ("user_01", "Swara (Primary User)"),
        ("user_02", "Alex Mercer"),
        ("user_03", "Priya Sharma"),
        ("user_04", "David Kim"),
        ("user_05", "Elena Rostova"),
        ("user_06", "Marcus Vance")
    ]

    identities_created = {}

    for idx, (uid, uname) in enumerate(identity_names[:num_identities]):
        user_folder = target_dir / uid
        user_folder.mkdir(parents=True, exist_ok=True)

        if idx == 0:
            user_base = base_face.copy()
        else:
            user_base = create_derived_identity(base_face, identity_seed=idx * 17)

        # Generate multiple variations for this user
        samples = apply_face_variations(user_base, num_variations=5)
        saved_paths = []

        for v_idx, img_rgb in enumerate(samples):
            # Save as BGR
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            tag = "enrollment" if v_idx == 0 else f"test_{v_idx}"
            file_path = user_folder / f"{uid}_{tag}.jpg"
            cv2.imwrite(str(file_path), img_bgr)
            saved_paths.append(str(file_path))

        identities_created[uid] = {
            "name": uname,
            "enrollment_image": saved_paths[0],
            "test_images": saved_paths[1:]
        }

    print(f"Generated {len(identities_created)} sample identities in '{target_dir}'.")
    return identities_created

if __name__ == "__main__":
    generate_benchmark_dataset()

