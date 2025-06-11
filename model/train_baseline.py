from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class BaselineTrainer:
    def __init__(self, model_type='logreg'):
        self.model_type = model_type
        self.model = None

    def train(self, X_train, y_train):
        if self.model_type == 'logreg':
            self.model = LogisticRegression(max_iter=1000)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        else:
            raise ValueError("Model type harus 'logreg' atau 'svm'")

        self.model.fit(X_train, y_train)
        return self.model
