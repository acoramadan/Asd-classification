import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

class FeatureExtractor:
    def __init__(self,
                 model_name="indobenchmark/indobert-base-p1",
                 max_length=128,
                 tfidf_max_features=5000,
                 tfidf_ngram_range=(1, 2),
                 tfidf_sublinear_tf=True):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            sublinear_tf=tfidf_sublinear_tf
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

        self.linguistic_cols = [
            'total_words', 'unique_words', 'num_sentences', 'stopwords',
            'num_adjectives', 'num_nouns', 'num_verbs', 'num_adverbs',
            'type_token_ratio', 'avg_words_per_sentence'
        ]

    def fit_transform_tfidf(self, text_series):
        return self.tfidf_vectorizer.fit_transform(text_series)

    def transform_tfidf(self, new_text_series):
        return self.tfidf_vectorizer.transform(new_text_series)

    def get_tfidf_vectorizer(self):
        return self.tfidf_vectorizer

    def encode_text_bert(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def encode_series_bert(self, text_series):
        return np.array([self.encode_text_bert(text) for text in text_series])

    def extract_fused_features_bert(self, df, text_column='clean_text'):
        bert_embeddings = self.encode_series_bert(df[text_column])
        linguistic_features = df[self.linguistic_cols].to_numpy().astype(np.float32)
        fused = np.concatenate([bert_embeddings, linguistic_features], axis=1)
        return fused