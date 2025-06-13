from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from .bert_trainer import BertFCTrainer
class BertFCEvaluator:
    def __init__(self, model_class, dataset, device='cpu', n_splits=5, pos_label='ASD'):
        self.model_class = model_class
        self.dataset = dataset
        self.device = device
        self.n_splits = n_splits
        self.pos_label = pos_label
    
    def evalute_kfold(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        y = np.array([1 if d['labels'] == 1 else 0 for d in self.dataset])
        results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)),y),1):
            print(f"\nFold{fold}")

            train_subset = torch.utils.data.Subset(self.dataset, train_idx)
            test_subset = torch.utils.data.Subset(self.dataset, test_idx)

            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size=16)

            model = self.model_class().to(self.device)
            trainer = BertFCTrainer(model, device=self.device)
            trainer.train(train_loader)

            metrics = self.evaluate_single(model, test_loader)
            results.append(metrics)
        
        print("\nAverage Metrics Across Folds:")
        for key in results[0]:
            avg = np.mean([m[key] for m in results])
            print(f"{key.capitalize():<10}: {avg:.4f}")
        return results
    
    def evaluate_single(self, model, dataloader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs).squeeze()
                predicted = (preds >= 0.5).int().cpu().numpy()
                true_labels = labels.cpu().numpy()

                all_preds.extend(predicted)
                all_labels.extend(true_labels)
        y_true = ['ASD' if l == 1 else 'NON ASD' for l in all_labels]
        y_pred = ['ASD' if p == 1 else 'NON ASD' for p in all_preds]

        print("Classification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=self.pos_label),
            'recall': recall_score(y_true, y_pred, pos_label=self.pos_label),
            'f1': f1_score(y_true, y_pred, pos_label=self.pos_label),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }