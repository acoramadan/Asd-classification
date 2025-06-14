import stanza
import numpy as np
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

class BackTranslationAugmentor:
    def __init__(self):
        self.id2en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
        self.id2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-id-en")
        self.en2id_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
        self.en2id_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-id")

        stanza.download('id', processors='tokenize,pos', verbose=False)
        self.nlp = stanza.Pipeline(lang='id', processors='tokenize,pos', verbose=False)

        self.stopwords = set([ 'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri',
            'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antara', 'antaranya', 'apa', 'apaan',
            'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan', 'atas', 'atau', 'ataukah', 'ataupun',
            'awal', 'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi', 'bagian',
            'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik', 'banyak', 'bapak', 'baru', 'bawah',
            'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu', 'begitukah', 'begitulah', 'begitupun',
            'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah', 'benar', 'benarkah', 'benarlah', 'berada',
            'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah', 'berapapun', 'berarti',
            'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya', 'berjumlah', 'berkali',
            'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan',
            'bermacam', 'bermaksud', 'bermula', 'bersama', 'bersiap', 'bertanya', 'berturut', 'bertutur', 'berupa',
            'besar', 'betul', 'biasa', 'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'buat',
            'bukan', 'bukankah', 'bukannya'])

    def back_translate(self, text: str) -> str:
        id2en = self.id2en_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True, truncation=True)
        en = self.id2en_model.generate(**id2en)
        en_text = self.id2en_tokenizer.decode(en[0], skip_special_tokens=True)

        en2id = self.en2id_tokenizer.prepare_seq2seq_batch([en_text], return_tensors="pt", padding=True, truncation=True)
        back = self.en2id_model.generate(**en2id)
        return self.en2id_tokenizer.decode(back[0], skip_special_tokens=True)

    def extract_features(self, text: str) -> dict:
        doc = self.nlp(text)
        words = [word.text for sent in doc.sentences for word in sent.words]
        pos_tags = [word.upos for sent in doc.sentences for word in sent.words]

        total_words = len(words)
        unique_words = len(set(words))
        num_sentences = len(doc.sentences)
        stopwords_count = sum([1 for word in words if word.lower() in self.stopwords])
        num_adjectives = pos_tags.count("ADJ")
        num_nouns = pos_tags.count("NOUN")
        num_verbs = pos_tags.count("VERB")
        num_adverbs = pos_tags.count("ADV")
        ttr = unique_words / total_words if total_words > 0 else 0
        avg_words_per_sentence = total_words / num_sentences if num_sentences > 0 else 0

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "num_sentences": num_sentences,
            "stopwords": stopwords_count,
            "num_adjectives": num_adjectives,
            "num_nouns": num_nouns,
            "num_verbs": num_verbs,
            "num_adverbs": num_adverbs,
            "type_token_ratio": ttr,
            "avg_words_per_sentence": avg_words_per_sentence
        }

    def augment_dataframe(self, df: pd.DataFrame, label_col='label', text_col='transcription',
                          target_label='NON ASD') -> pd.DataFrame:
        subset = df[df[label_col].str.upper().str.strip() == target_label].copy()
        print(f"Processing {len(subset)} utterances for back translation...")

        augmented_rows = []

        for _, row in tqdm(subset.iterrows(), total=len(subset)):
            original_text = row[text_col]
            bt_text = self.back_translate(original_text)
            features = self.extract_features(bt_text)

            new_row = {
                label_col: row[label_col],
                text_col: bt_text,
                **features
            }
            augmented_rows.append(new_row)

        df_augmented = pd.DataFrame(augmented_rows)
        df_augmented["is_augmented"] = True
        df["is_augmented"] = False

        return pd.concat([df, df_augmented], ignore_index=True)