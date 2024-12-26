import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(abstracts):
    """Preprocess a list of abstracts by removing stopwords and special characters."""
    stop_words = set(stopwords.words('english'))
    processed_texts = [
        [word for word in word_tokenize(re.sub(r'[^\w\s]', '', abstract.lower())) if word not in stop_words]
        for abstract in abstracts
    ]
    return processed_texts