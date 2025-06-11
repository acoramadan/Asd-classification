import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TextPreprocessor:
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        nltk_stopwords = set([
            'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 'akhiri',
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
            'bukan', 'bukankah', 'bukannya'
        ])
        
        sastrawi_stopwords = set("""
            yang untuk pada ke dari dengan dan di adalah itu dalam tidak bahwa oleh sebagai
            sehingga agar karena jika namun maka telah lebih juga hanya saat sedang akan bisa
            dapat kepada mereka kami kita anda saya aku kamu ia dia nya para seluruh setiap
            tersebut yaitu yakni contoh tersebutlah antara olehnya adanya adanya jadi
            adalahnya merupakan namunlah hingga setelah sebelum tentang pun apapun
        """.split())

        self.stopwords = nltk_stopwords.union(sastrawi_stopwords)
        self.stopwords.update([
            "iya", "ya", "loh", "sih", "kan", "um", "uh", "hmm", "heeh", "gitu", "nih",
            "nggak", "gak", "ndak", "nda", "idak", "ga"
        ])

    def normalize_text(self, text):
        text = re.sub(r'\b(ngg+ak+|g+ak+|ndak+|nda+|idak+|ga+)\b', 'tidak', text)
        text = re.sub(r'\b(iya+|iye+|yaa+|ye+|heeh+|he+|eh+)\b', 'iya', text)
        text = re.sub(r'\b(hmm+|uh+|um+)\b', 'hmm', text)
        return text

    def preprocess(self, text):
        if pd.isnull(text):
            return ""
        text = text.lower()
        text = self.normalize_text(text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = self.stemmer.stem(text)
        tokens = text.strip().split()
        tokens = [word for word in tokens if word not in self.stopwords]
        return ' '.join(tokens)

    def preprocess_series(self, text_series):
        return text_series.apply(self.preprocess)
