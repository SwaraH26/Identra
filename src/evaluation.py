"""
Academic Performance Evaluation & Benchmarking Module.
Computes biometric verification metrics: Accuracy, FAR, FRR, EER, Precision, Recall, F1-score,
and generates publication-quality plots (PRD Section 11 & FR10).
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
import config

class BiometricEvaluator:
    """
    Evaluates verification performance on genuine and impostor score pairs.
    """
    def __init__(self, output_dir: Path = config.EVALUATION_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, float]:
        """
        Computes biometric performance metrics at a specific decision threshold.
        y_true: 1 for Genuine attempt, 0 for Impostor attempt.
        y_scores: similarity scores [0.0, 1.0].
        """
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)
        y_pred = (y_scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        total_genuine = tp + fn
        total_impostor = tn + fp
        total = len(y_true)

        # False Acceptance Rate: Impostors accepted / Total impostor attempts
        far = (fp / total_impostor) if total_impostor > 0 else 0.0

        # False Rejection Rate: Genuine rejected / Total genuine attempts
        frr = (fn / total_genuine) if total_genuine > 0 else 0.0

        # Overall Accuracy
        accuracy = ((tp + tn) / total) if total > 0 else 0.0

        # Precision, Recall, F1
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "far": float(far),
            "frr": float(frr),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

    @staticmethod
    def find_eer(y_true: np.ndarray, y_scores: np.ndarray, num_points: int = 1000) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the Equal Error Rate (EER) and the corresponding optimal threshold.
        Returns:
            (eer, optimal_threshold, thresholds, far_curve, frr_curve)
        """
        thresholds = np.linspace(0.0, 1.0, num_points)
        far_list = []
        frr_list = []

        for th in thresholds:
            m = BiometricEvaluator.compute_metrics(y_true, y_scores, th)
            far_list.append(m["far"])
            frr_list.append(m["frr"])

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)

        # EER is where |FAR - FRR| is minimized
        diff = np.abs(far_arr - frr_arr)
        idx = int(np.argmin(diff))

        eer = float((far_arr[idx] + frr_arr[idx]) / 2.0)
        optimal_th = float(thresholds[idx])

        return eer, optimal_th, thresholds, far_arr, frr_arr

    def generate_evaluation_report(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                   system_name: str = "Deep Learning Face Recognition") -> Dict[str, Any]:
        """
        Runs comprehensive evaluation, generates all plots, and saves figures.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)

        # 1. Compute EER
        eer, opt_th, thresholds, far_curve, frr_curve = self.find_eer(y_true, y_scores)
        metrics_at_eer = self.compute_metrics(y_true, y_scores, opt_th)

        # 2. Compute ROC & AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # 3. Create 4-panel evaluation figure
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axs = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(f"{system_name} - Biometric Performance Benchmark", fontsize=16, fontweight="bold", y=0.98)

        # Panel 1: Score Distribution
        genuine_scores = y_scores[y_true == 1]
        impostor_scores = y_scores[y_true == 0]

        axs[0, 0].hist(impostor_scores, bins=25, alpha=0.6, color="salmon", label=f"Impostor Pairs (N={len(impostor_scores)})", density=True)
        axs[0, 0].hist(genuine_scores, bins=25, alpha=0.6, color="royalblue", label=f"Genuine Pairs (N={len(genuine_scores)})", density=True)
        axs[0, 0].axvline(opt_th, color="black", linestyle="--", linewidth=1.8, label=f"Optimal Threshold = {opt_th:.2f}")
        axs[0, 0].set_title("1. Similarity Score Distribution", fontsize=12, fontweight="bold")
        axs[0, 0].set_xlabel("Cosine Similarity Score")
        axs[0, 0].set_ylabel("Density")
        axs[0, 0].legend(loc="upper left")

        # Panel 2: FAR vs FRR Curve
        axs[0, 1].plot(thresholds, far_curve, label="FAR (False Acceptance Rate)", color="crimson", linewidth=2)
        axs[0, 1].plot(thresholds, frr_curve, label="FRR (False Rejection Rate)", color="navy", linewidth=2)
        axs[0, 1].scatter([opt_th], [eer], color="forestgreen", s=90, zorder=5, label=f"EER = {eer:.2%} @ Th={opt_th:.2f}")
        axs[0, 1].set_title("2. FAR vs. FRR as a Function of Threshold", fontsize=12, fontweight="bold")
        axs[0, 1].set_xlabel("Decision Threshold")
        axs[0, 1].set_ylabel("Error Rate")
        axs[0, 1].set_ylim(-0.02, 1.02)
        axs[0, 1].legend(loc="center right")

        # Panel 3: ROC Curve
        axs[1, 0].plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        axs[1, 0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
        axs[1, 0].set_title("3. Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
        axs[1, 0].set_xlabel("False Positive Rate (FAR)")
        axs[1, 0].set_ylabel("True Positive Rate (1 - FRR)")
        axs[1, 0].legend(loc="lower right")

        # Panel 4: Confusion Matrix at Optimal Threshold
        cm = np.array([[metrics_at_eer["tn"], metrics_at_eer["fp"]],
                       [metrics_at_eer["fn"], metrics_at_eer["tp"]]])
        im = axs[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[1, 1].set_title(f"4. Confusion Matrix (Threshold = {opt_th:.2f})", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
        
        classes = ["Impostor", "Genuine"]
        tick_marks = np.arange(len(classes))
        axs[1, 1].set_xticks(tick_marks)
        axs[1, 1].set_xticklabels(classes)
        axs[1, 1].set_yticks(tick_marks)
        axs[1, 1].set_yticklabels(classes)

        thresh_val = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axs[1, 1].text(j, i, f"{cm[i, j]}",
                               horizontalalignment="center",
                               color="white" if cm[i, j] > thresh_val else "black",
                               fontweight="bold", fontsize=13)

        axs[1, 1].set_ylabel("True Identity")
        axs[1, 1].set_xlabel("Predicted Status")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save plot
        plot_path = self.output_dir / "biometric_evaluation_benchmark.png"
        plt.savefig(str(plot_path), dpi=200)
        plt.close(fig)

        summary = {
            "system_name": system_name,
            "eer": eer,
            "optimal_threshold": opt_th,
            "roc_auc": roc_auc,
            "metrics_at_optimal_threshold": metrics_at_eer,
            "plot_saved_at": str(plot_path)
        }

        return summary

