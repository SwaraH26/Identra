"""
Core Face Recognition Engine.
Orchestrates face detection, feature extraction, database enrollment,
verification (1:1), identification (1:N), and decision logic.
"""
import json
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import config
from src.face_detector import FaceDetector
from src.feature_extractor import FaceFeatureExtractor
from src.liveness_detector import LivenessDetector

class FaceRecognitionEngine:
    """
    Complete Deep Learning Face Recognition Engine.
    """
    def __init__(self, backbone: str = config.DEFAULT_BACKBONE,
                 embedding_dim: int = config.EMBEDDING_DIM,
                 verification_threshold: float = config.DEFAULT_VERIFICATION_THRESHOLD,
                 enable_liveness: bool = config.ENABLE_LIVENESS_CHECK):
        
        self.detector = FaceDetector()
        self.extractor = FaceFeatureExtractor(backbone=backbone, embedding_dim=embedding_dim)
        self.liveness_detector = LivenessDetector() if enable_liveness else None
        
        self.threshold = verification_threshold
        self.database_file = config.USERS_FILE
        self.faces_dir = config.FACES_DIR
        
        # In-memory database: {user_id: {"name": str, "embedding": np.ndarray, ...}}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.load_database()

    def load_database(self) -> None:
        """Loads enrolled users and embeddings from the JSON file."""
        if not self.database_file.exists():
            self.users = {}
            return

        try:
            with open(self.database_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.users = {}
            for user_id, info in data.items():
                self.users[user_id] = {
                    "user_id": user_id,
                    "name": info.get("name", user_id),
                    "embedding": np.array(info["embedding"], dtype=np.float32),
                    "created_at": info.get("created_at", ""),
                    "image_path": info.get("image_path", "")
                }
        except Exception as e:
            print(f"[Warning] Failed to load database from {self.database_file}: {e}")
            self.users = {}

    def save_database(self) -> None:
        """Saves enrolled users and embeddings to the JSON file."""
        serializable = {}
        for user_id, info in self.users.items():
            serializable[user_id] = {
                "name": info["name"],
                "embedding": info["embedding"].tolist(),
                "created_at": info.get("created_at", ""),
                "image_path": info.get("image_path", "")
            }
        
        with open(self.database_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def enroll_user(self, user_id: str, name: str, image_bgr: np.ndarray,
                     check_liveness: bool = True) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Enrolls a new user into the biometric database.
        Returns:
            (success: bool, message: str, details: dict)
        """
        user_id = user_id.strip()
        name = name.strip()
        if not user_id or not name:
            return False, "User ID and Name cannot be empty.", None

        # 1. Detect and crop face
        try:
            face_rgb, bbox = self.detector.extract_face(image_bgr)
        except Exception as e:
            return False, f"Enrollment failed: {str(e)}", None

        # 2. Check liveness (anti-spoofing)
        liveness_info = None
        if check_liveness and self.liveness_detector is not None:
            is_live, score, diagnostics = self.liveness_detector.evaluate_liveness(face_rgb)
            liveness_info = diagnostics
            if not is_live:
                return False, f"Enrollment rejected: Potential presentation attack / photo spoof detected (Sharpness score: {score:.2f}).", liveness_info

        # 3. Extract deep CNN embedding vector
        try:
            embedding = self.extractor.extract_embedding(face_rgb)
        except Exception as e:
            return False, f"Feature extraction failed: {str(e)}", None

        # 4. Save face crop image
        image_filename = f"{user_id}.jpg"
        image_save_path = self.faces_dir / image_filename
        # Convert RGB back to BGR for saving
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_save_path), face_bgr)

        # 5. Store user record
        self.users[user_id] = {
            "user_id": user_id,
            "name": name,
            "embedding": embedding,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": str(image_save_path)
        }
        self.save_database()

        details = {
            "user_id": user_id,
            "name": name,
            "embedding_dim": len(embedding),
            "bbox": bbox,
            "face_rgb": face_rgb,
            "liveness": liveness_info
        }

        return True, f"Successfully enrolled {name} (ID: {user_id}).", details

    def verify_user(self, claimed_user_id: str, test_image_bgr: np.ndarray,
                    threshold: Optional[float] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        1:1 Verification: Verifies whether the input image belongs to the claimed user ID.
        """
        if claimed_user_id not in self.users:
            return False, f"User '{claimed_user_id}' is not enrolled in the database.", {}

        thresh = threshold if threshold is not None else self.threshold

        # 1. Detect and extract face
        try:
            face_rgb, bbox = self.detector.extract_face(test_image_bgr)
        except Exception as e:
            return False, f"Face detection failed: {str(e)}", {}

        # 2. Liveness check
        liveness_info = {}
        if self.liveness_detector is not None:
            is_live, _, liveness_info = self.liveness_detector.evaluate_liveness(face_rgb)
            if not is_live:
                return False, "Verification failed: Presentation attack detected (Liveness check failed).", {
                    "bbox": bbox, "face_rgb": face_rgb, "liveness": liveness_info, "status": "REJECTED"
                }

        # 3. Extract test embedding
        test_embedding = self.extractor.extract_embedding(face_rgb)

        # 4. Compute similarity
        enrolled_embedding = self.users[claimed_user_id]["embedding"]
        similarity = self.extractor.compute_similarity(test_embedding, enrolled_embedding, metric=config.SIMILARITY_METRIC)

        is_match = similarity >= thresh
        user_name = self.users[claimed_user_id]["name"]
        
        status = "ACCEPTED" if is_match else "REJECTED"
        msg = f"{status}: {user_name} (Similarity: {similarity:.1%}, Threshold: {thresh:.1%})"

        result = {
            "status": status,
            "is_match": is_match,
            "claimed_user_id": claimed_user_id,
            "user_name": user_name,
            "similarity": similarity,
            "threshold": thresh,
            "bbox": bbox,
            "face_rgb": face_rgb,
            "liveness": liveness_info
        }

        return is_match, msg, result

    def identify_user(self, test_image_bgr: np.ndarray,
                      threshold: Optional[float] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        1:N Identification: Finds the closest matching identity from the enrolled database.
        Returns:
            (is_accepted: bool, message: str, result_dict: dict)
        """
        if not self.users:
            return False, "No users currently enrolled in the database.", {}

        thresh = threshold if threshold is not None else self.threshold

        # 1. Detect and crop face
        try:
            face_rgb, bbox = self.detector.extract_face(test_image_bgr)
        except Exception as e:
            return False, f"Face detection failed: {str(e)}", {}

        # 2. Liveness check
        liveness_info = {}
        if self.liveness_detector is not None:
            is_live, _, liveness_info = self.liveness_detector.evaluate_liveness(face_rgb)
            if not is_live:
                return False, "REJECTED: Face spoof detected by liveness analyzer.", {
                    "bbox": bbox, "face_rgb": face_rgb, "liveness": liveness_info, "status": "REJECTED"
                }

        # 3. Extract test embedding
        test_embedding = self.extractor.extract_embedding(face_rgb)

        # 4. Compare against all enrolled users
        best_user_id = None
        best_name = "Unknown"
        best_similarity = -1.0
        all_scores = {}

        for user_id, user_data in self.users.items():
            enrolled_emb = user_data["embedding"]
            sim = self.extractor.compute_similarity(test_embedding, enrolled_emb, metric=config.SIMILARITY_METRIC)
            all_scores[user_id] = {
                "name": user_data["name"],
                "similarity": sim
            }
            if sim > best_similarity:
                best_similarity = sim
                best_user_id = user_id
                best_name = user_data["name"]

        # 5. Apply Decision Logic (FR8 & FR9 in PRD)
        is_accepted = best_similarity >= thresh
        status = "ACCEPTED" if is_accepted else "REJECTED"

        if is_accepted:
            msg = f"ACCEPTED: Matched '{best_name}' (Score: {best_similarity:.1%}, Threshold: {thresh:.1%})"
        else:
            msg = f"REJECTED: Unknown person or confidence too low (Best score: {best_similarity:.1%}, Required: {thresh:.1%})"

        result = {
            "status": status,
            "is_accepted": is_accepted,
            "matched_user_id": best_user_id if is_accepted else "UNKNOWN",
            "matched_name": best_name if is_accepted else "Unknown Individual",
            "similarity": best_similarity,
            "threshold": thresh,
            "all_scores": all_scores,
            "bbox": bbox,
            "face_rgb": face_rgb,
            "liveness": liveness_info
        }

        return is_accepted, msg, result

    def delete_user(self, user_id: str) -> bool:
        """Deletes a user from database and discards enrolled image."""
        if user_id in self.users:
            img_path = self.users[user_id].get("image_path")
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
            del self.users[user_id]
            self.save_database()
            return True
        return False

    def list_users(self) -> List[Dict[str, Any]]:
        """Returns metadata list of all enrolled users."""
        return [
            {
                "user_id": u["user_id"],
                "name": u["name"],
                "created_at": u.get("created_at", "N/A"),
                "image_path": u.get("image_path", "")
            }
            for u in self.users.values()
        ]

