import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np

class BertFCTrainer:
    def __init__(self, model, lr=1e-4, batch_size=32, epochs=20, patience=3, device=None):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, X_train, y_train, X_val, y_val):
        print("sudah di update")
        self.model.to(self.device)
        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None  

        train_loader = self._create_dataloader(X_train, y_train)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = self.evaluate_loss(val_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if best_model is not None:
            self.model.load_state_dict(best_model)

    def evaluate_loss(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.unsqueeze(1))
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        test_loader = self._create_dataloader(X_test, y_test, shuffle=False)
        y_preds = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
                y_preds.extend(predictions)

        print(classification_report(y_test, y_preds, target_names=['NON ASD', 'ASD']))

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return (output.cpu().numpy() > 0.5).astype(int).flatten()

    def _create_dataloader(self, X, y, shuffle=True):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
