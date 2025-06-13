import torch
import torch.nn as nn

class BertFCClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.5):
        super(BertFCClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x).squeeze()