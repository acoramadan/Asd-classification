from lime.lime_text import LimeTextExplainer
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