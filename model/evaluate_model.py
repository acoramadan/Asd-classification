from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np
from .train_baseline import BaselineTrainer

class ModelEvaluator:
    def __init__(self, model_type="logreg", n_splits=5, random_state=42, pos_label="ASD"):
        self.model_type = model_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.pos_label = pos_label

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(classification_report(y_test, y_pred, digits=3))
        print("Confusion Matrix:\n", cm)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=self.pos_label),
            "recall": recall_score(y_test, y_pred, pos_label=self.pos_label),
            "f1": f1_score(y_test, y_pred, pos_label=self.pos_label),
            "confusion_matrix": cm
        }

    def cross_validate(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            baseline = BaselineTrainer(model_type=self.model_type)
            model = baseline.train(X_train, y_train)
            metrics = self.evaluate(model, X_test, y_test)
            results.append(metrics)

        print("\nMean Metrics:")
        for key in results[0].keys():
            avg = np.mean([m[key] for m in results])
            print(f"{key.capitalize():<9}: {avg:.4f}")

        return results
 
    def cross_validate_with_confusionmatrix(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        results = []
        best_fold = {"fold": -1, "f1": 0.0}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n Fold {fold}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            baseline = BaselineTrainer(model_type=self.model_type)
            model = baseline.train(X_train, y_train)
            metrics = self.evaluate(model, X_test, y_test)

            print(f"Confusion Matrix (Fold {fold}):\n{metrics['confusion_matrix']}")
            results.append(metrics)

            if metrics["f1"] > best_fold["f1"]:
                best_fold = {"fold": fold, "f1": metrics["f1"]}

        print("\nMean Metrics:")
        for key in ["accuracy", "precision", "recall", "f1"]:
            avg = np.mean([m[key] for m in results])
            print(f"{key.capitalize():<9}: {avg:.4f}")

        print(f"\nBest Fold: Fold {best_fold['fold']} with F1-score = {best_fold['f1']:.4f}")
        return results