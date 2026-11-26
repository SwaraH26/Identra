"""
Automated Benchmarking and Evaluation Script.
Runs comprehensive testing across Genuine and Impostor verification pairs.
Computes FAR, FRR, EER, Accuracy, and produces visual evaluation plots.
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

import config
from src.face_detector import FaceDetector
from src.feature_extractor import FaceFeatureExtractor
from src.evaluation import BiometricEvaluator
from generate_sample_data import generate_benchmark_dataset

def run_backbone_benchmark(backbone_name: str, sample_dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates a specific CNN backbone on genuine and impostor face pairs.
    """
    print(f"\n--- Evaluating Backbone: {backbone_name.upper()} ---")
    extractor = FaceFeatureExtractor(backbone=backbone_name, embedding_dim=config.EMBEDDING_DIM)

    # 1. Compute enrolled embeddings for each identity
    enrolled_embeddings = {}
    for uid, data in sample_dataset.items():
        enroll_path = data["enrollment_image"]
        img_bgr = cv2.imread(enroll_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(img_rgb, config.FACE_INPUT_SIZE)
        emb = extractor.extract_embedding(resized_rgb)
        enrolled_embeddings[uid] = emb

    y_true = []
    y_scores = []
    pair_logs = []

    # 2. Genuine Pairs (Test images of user X vs Enrolled image of user X)
    for uid, data in sample_dataset.items():
        ref_emb = enrolled_embeddings[uid]
        for test_path in data["test_images"]:
            img_bgr = cv2.imread(test_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(img_rgb, config.FACE_INPUT_SIZE)
            test_emb = extractor.extract_embedding(resized_rgb)
            sim = extractor.compute_similarity(test_emb, ref_emb, metric="cosine")
            
            y_true.append(1)
            y_scores.append(sim)
            pair_logs.append({
                "backbone": backbone_name,
                "pair_type": "Genuine",
                "claimed_id": uid,
                "true_id": uid,
                "similarity": sim
            })

    # 3. Impostor Pairs (Test images of user X vs Enrolled image of user Y, where X != Y)
    uids = list(sample_dataset.keys())
    for i, true_uid in enumerate(uids):
        data = sample_dataset[true_uid]
        for test_path in data["test_images"]:
            img_bgr = cv2.imread(test_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(img_rgb, config.FACE_INPUT_SIZE)
            test_emb = extractor.extract_embedding(resized_rgb)

            for j, claimed_uid in enumerate(uids):
                if true_uid != claimed_uid:
                    ref_emb = enrolled_embeddings[claimed_uid]
                    sim = extractor.compute_similarity(test_emb, ref_emb, metric="cosine")
                    
                    y_true.append(0)
                    y_scores.append(sim)
                    pair_logs.append({
                        "backbone": backbone_name,
                        "pair_type": "Impostor",
                        "claimed_id": claimed_uid,
                        "true_id": true_uid,
                        "similarity": sim
                    })

    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores, dtype=float)

    evaluator = BiometricEvaluator(output_dir=config.EVALUATION_DIR)
    eer, opt_th, _, _, _ = evaluator.find_eer(y_true, y_scores)
    metrics = evaluator.compute_metrics(y_true, y_scores, opt_th)

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    summary = {
        "backbone": backbone_name,
        "total_pairs": len(y_true),
        "genuine_pairs": int(np.sum(y_true == 1)),
        "impostor_pairs": int(np.sum(y_true == 0)),
        "eer": eer,
        "optimal_threshold": opt_th,
        "accuracy_at_eer": metrics["accuracy"],
        "far_at_eer": metrics["far"],
        "frr_at_eer": metrics["frr"],
        "roc_auc": roc_auc
    }

    return summary, y_true, y_scores


def main():
    print("=" * 65)
    print(" ADVANCED FACIAL RECOGNITION SYSTEM - BENCHMARK & EVALUATION")
    print("=" * 65)

    # 1. Ensure sample data exists
    dataset = generate_benchmark_dataset(config.SAMPLE_DATA_DIR, num_identities=6)

    # 2. Run benchmark across supported CNN architectures
    backbones = ["mobilenet_v2", "resnet18", "custom_cnn"]
    results = []
    best_summary = None
    best_y_true = None
    best_y_scores = None

    for bb in backbones:
        summary, y_true, y_scores = run_backbone_benchmark(bb, dataset)
        results.append(summary)
        if bb == config.DEFAULT_BACKBONE:
            best_summary = summary
            best_y_true = y_true
            best_y_scores = y_scores

    # 3. Generate visual 4-panel benchmark chart for primary backbone
    evaluator = BiometricEvaluator(output_dir=config.EVALUATION_DIR)
    evaluator.generate_evaluation_report(
        best_y_true, best_y_scores, 
        system_name=f"MobileNetV2 CNN Face Recognition"
    )

    # 4. Display Comparison Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print(" ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 65)
    for res in results:
        print(f"Backbone: {res['backbone']:<15} | Accuracy: {res['accuracy_at_eer']:.2%} | EER: {res['eer']:.2%} | AUC: {res['roc_auc']:.3f} | Opt Thresh: {res['optimal_threshold']:.2f}")

    # Save summary table to CSV
    csv_path = config.EVALUATION_DIR / "architecture_comparison.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n[Saved] Evaluation metrics saved to '{csv_path}'")
    print(f"[Saved] Benchmark plots saved to '{config.EVALUATION_DIR / 'biometric_evaluation_benchmark.png'}'")
    print("=" * 65)

if __name__ == "__main__":
    main()

