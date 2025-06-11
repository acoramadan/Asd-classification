from lime.lime_text import LimeTextExplainer
import pandas as pd
import numpy as np

class LimeTextInterpreter:
    def __init__(self, model, vectorizer, class_names):
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)
    
    def explain_instance(self, text, num_features=10):
        explanation = self.explainer.explain_instance(
        text_instance=text,
        classifier_fn=self.predict_proba,
        num_features=num_features
    )
        return explanation

    def save_lime_explanation_to_csv(self, df, output_path, num_features=10):
        results = []

        for i, row in df.iterrows():
            text = row['transcription']
            true_label = row['label']
            pred_label = self.model.predict(self.vectorizer.transform([text]))[0]
            explanation = self.explain_instance(text, num_features=num_features)
            for word, weight in explanation.as_list():
                results.append({
                  "index": i,
                  "true_label": true_label,
                  "predicted_label": pred_label,
                  "word": word,
                  "weight": weight,
                  "text": text
                })
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_path, index=False)
        print("Berhasil menyimpan!")
