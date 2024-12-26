from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def classify_text(abstracts, labels):
    """Classify texts using TF-IDF and a Naive Bayes classifier."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(abstracts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
    return classifier, vectorizer