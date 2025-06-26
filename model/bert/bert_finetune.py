from transformers import AutoModel
import torch.nn as nn
import torch
class FineTuneBertWithLinguistic(nn.Module):
    def __init__(self, model_name="indobenchmark/indobert-base-p1", ling_dim=10, hidden_dim=128, dropout=0.3):
        super(FineTuneBertWithLinguistic, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + ling_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, linguistic):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.last_hidden_state[:, 0]  
        combined = torch.cat((cls_output, linguistic), dim=1)
        return self.classifier(combined)