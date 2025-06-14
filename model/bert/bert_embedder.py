from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class BertEmbedder:
    def __init__(self, model_name="indobenchmark/indobert-base-p1", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.model.eval()  

    def encode_text(self, text):
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                padding=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def encode_series(self, text_series):
        return np.array([self.encode_text(text) for text in text_series])