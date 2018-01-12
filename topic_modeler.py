import os
import pandas as pd

from collections import Counter
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
                                            HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline

from tokenizer import StemTokenizer


def topic_name(fnames, feats):
    """
    utility function for mapping a topic vector to a human-readable group of
    words
    """
    total = sum(feats)
    return  ', '.join('(%.2f) %s' % (feats[i] / total, fnames[i])
                  for i in feats.argsort()[:-6:-1])


class TopicModeler(object):
    TFIDF = 'tfidf'     # term frequency-inverse document frequency
    TF = 'tf'           # plain old term frequency
    HASH = 'hash'       # terms are hashed into a smaller space (e.g. 1000)
    HASH_IDF = 'hash-idf'
    NMF = 'nmf'         # Non-Negative Matrix Factorization
    LDA = 'lda'         # Latent Dirichlet Allocation

    def __init__(self, vector_type=TFIDF, model_type=NMF, n_features=1000,
                 n_topics=20):
        """
        Holds state for text processing and topic modeling.
        Vector type choices: 'tfidf', 'tf', 'hash', 'hash-idf'
        Model type choices: 'lda', 'nmf'
        """
        self.vector_type = vector_type
        self.model_type = model_type
        self.n_features = n_features
        self.n_topics = n_topics

    def vectorize(self, docs):
        """
        Fit a document vectorizer to a set of documents and transform them into
        vectors.
        """
        print 'vectorizing', len(docs), 'documents of total size', \
            sum([len(d) for d in docs])/1000, 'KB'

        # generate hashing vectors
        if self.vector_type == self.HASH_IDF:
            hasher = HashingVectorizer(n_features=self.n_features,
                                       tokenizer=StemTokenizer(),
                                       stop_words='english',
                                       non_negative=True, norm=None,
                                       binary=False)
            self.vectorizer = make_pipeline(hasher, TfidfTransformer())

        elif self.vector_type == self.HASH:
            self.vectorizer = HashingVectorizer(n_features=self.n_features,
                                                tokenizer=StemTokenizer(),
                                                stop_words='english',
                                                non_negative=False, norm='l2',
                                                binary=False)

        else:
            # generate term-frequency, inverse-document-frequency vectors
            if self.vector_type == self.TFIDF:
                Vectorizer = TfidfVectorizer
            # generate plain term-frequency vector
            elif self.vector_type == self.TF:
                Vectorizer = CountVectorizer

            self.vectorizer = Vectorizer(max_df=0.8, min_df=2,
                                         max_features=self.n_features,
                                         tokenizer=StemTokenizer(),
                                         stop_words='english')

        return self.vectorizer.fit_transform(docs)

    def fit_topic_model(self, vectors, verbose=False):
        if self.model_type == self.NMF:
            self.model = NMF(n_components=self.n_topics, random_state=1,
                             alpha=.1, l1_ratio=.5)
        elif self.model_type == self.LDA:
            self.model = LatentDirichletAllocation(n_topics=self.n_topics,
                                                   max_iter=5,
                                                   learning_method='online',
                                                   learning_offset=50.,
                                                   random_state=0)

        print 'fitting model of type', self.model_type, 'to', vectors.shape[0],\
            'with', self.n_topics, 'topics'

        res = self.model.fit_transform(vectors)
        self.baseline_topics = sum(res) / len(res)

        word_names = self.vectorizer.get_feature_names()
        self.topics = []
        for group in self.model.components_:
            topic = topic_name(word_names, group)
            self.topics.append(topic)

        return self.topics

    def fit(self, docs):
        """
        Train a topic modeler on the entire text corpus.
        """
        print 'building vectors...'
        vectors = self.vectorize(docs)

        print 'fitting model...'
        self.fit_topic_model(vectors)

    def transform(self, docs):
        vecs = self.vectorizer.transform(docs)   # docs needs to be a list!
        res = self.model.transform(vecs)
        return pd.DataFrame(columns=self.topics, data=res)

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)


def print_top_topics(tm, df):
    """
    Count the number of documents for which each topic is one of the top 3
    factors, and print the most common topics.
    """
    best_topics = [sorted(r.index, key=lambda i: r[i], reverse=True)[:3]
                   for _, r in df.iterrows()]
    topics = Counter(sum(best_topics, []))
    print 'Top topics:'
    for topic, n in sorted(topics.items(), key=lambda i: i[1], reverse=True):
        print '%d:' % n, topic


def print_topic_shares(tm, df):
    topics = {t: sum(df[t]) for t in df.columns}
    print 'Top topics:'
    for topic, n in sorted(topics.items(), key=lambda i: i[1], reverse=True):
        print '%.2f:' % n, topic
