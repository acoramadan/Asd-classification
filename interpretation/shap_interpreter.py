import shap
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

class ShapInterpreter:
    def __init__(self, model, tokenizer_name="indobenchmark/indobert-base-p1", max_length=128):
        self.model = model 
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.bert_model = AutoModel.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def bert_embed(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            mean_pooled = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(mean_pooled)
        return np.array(embeddings)

    def explain(self, texts, num_samples=100):
        def embed_predict(texts):
            embeds = self.bert_embed(texts)
            return self.model.predict_proba(embeds)

        explainer = shap.Explainer(embed_predict, shap.maskers.Text(self.tokenizer))
        shap_values = explainer(texts[:num_samples])

        return shap_values

    def visualize(self, shap_values, sample_index=0):
        shap.plots.text(shap_values[sample_index])

    def save_shap_explanations_to_csv(self, shap_values, texts, output_path, true_labels=None):
        records = []
        for i, sv in enumerate(shap_values):
            words = sv.data
            scores = sv.values
            for word, score in zip(words, scores):
                record = {
                    "index": i,
                    "word": word,
                    "shap_value": score,
                    "text": texts[i]
                }
                if true_labels is not None:
                    record["true_label"] = true_labels[i]
                records.append(record)

        df_out = pd.DataFrame(records)
        df_out.to_csv(output_path, index=False)
        print(f"✅ SHAP explanations saved to: {output_path}")