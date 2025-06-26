import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class TrainerFusionBert:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=2e-5, batch_size=16, epochs=10, patience=3):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    def train(self, train_dataset, val_dataset):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                linguistic = batch['linguistic'].to(self.device)
                labels = batch['labels'].float().unsqueeze(1).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, linguistic)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss  : {val_loss:.4f}")
            print(f"  Val Acc   : {val_metrics['accuracy']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        if best_model:
            self.model.load_state_dict(best_model)
        return self.model

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                linguistic = batch['linguistic'].to(self.device)
                labels = batch['labels'].float().unsqueeze(1).to(self.device)

                outputs = self.model(input_ids, attention_mask, linguistic)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = (outputs >= 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        y_true = [int(x[0]) for x in all_labels]
        y_pred = [int(x[0]) for x in all_preds]

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        return total_loss / len(dataloader), metrics