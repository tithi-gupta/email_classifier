
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SpamClassifier:
    def __init__(self, path):
        self.df = pd.read_csv(path, encoding='latin-1')
        self.df['Category'] = self.df['Category'].map({'ham': 0, 'spam': 1})
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self):
        X = self.vectorizer.fit_transform(self.df['Message'])
        y = self.df['Category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

    def predict(self, new_text):
        X_new = self.vectorizer.transform([new_text])
        prediction = self.model.predict(X_new)
        return "Spam" if prediction[0] else "Not Spam"
