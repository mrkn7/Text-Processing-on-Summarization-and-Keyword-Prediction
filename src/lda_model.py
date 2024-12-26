from gensim import corpora
from gensim.models import LdaModel

def create_lda_model(processed_texts, num_topics=10, passes=15, random_state=42):
    """Create an LDA model from preprocessed texts."""
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=random_state)
    return lda_model, dictionary, corpus

def display_topics(lda_model, num_words=6):
    """Display topics from the trained LDA model."""
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)