import os
import pandas as pd

from collections import Counter
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
                                            HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import stop_words

# include contraction parts in stop words
STOPWORDS = list(stop_words.ENGLISH_STOP_WORDS) + ['don', 'haven', 've']


class TopicModeler(object):
    TFIDF = 'tfidf'     # term frequency-inverse document frequency
    TF = 'tf'           # plain old term frequency
    HASH = 'hash'       # terms are hashed into a smaller space (e.g. 1000)
    HASH_IDF = 'hash-idf'
    NMF = 'nmf'         # Non-Negative Matrix Factorization
    LDA = 'lda'         # Latent Dirichlet Allocation

    def __init__(self, vector_type=TFIDF, model_type=NMF, n_features=1000,
                 n_topics=20, max_df=0.6):
        """
        Holds state for text processing and topic modeling.
        Vector type choices: 'tfidf', 'tf', 'hash', 'hash-idf'
        Model type choices: 'lda', 'nmf'
        """
        self.vector_type = vector_type
        self.model_type = model_type
        self.n_features = n_features
        self.n_topics = n_topics
        self.max_df = max_df

    def _get_topic_names(self):
        """ create human-readable names for each of the components (topics) """
        # feat_names are the names of the tokens which comprise the topics
        feat_names = self.vectorizer.get_feature_names()
        self.topics = []
        for comps in self.model.components_:
            # rank the components of this topic by the proportion of the topic
            # they represent
            total = sum(comps)
            comp_locs = (-comps).argsort()

            # name the topic using the top 5 most significant component tokens
            topic = ', '.join('(%.2f) %s' % (comps[i] / total, feat_names[i])
                              for i in comp_locs[:5])
            self.topics.append(topic)

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
                                       stop_words=STOPWORDS,
                                       non_negative=True,
                                       norm=None, binary=False)
            self.vectorizer = make_pipeline(hasher, TfidfTransformer())

        elif self.vector_type == self.HASH:
            self.vectorizer = HashingVectorizer(n_features=self.n_features,
                                                stop_words=STOPWORDS,
                                                non_negative=False, norm='l2',
                                                binary=False)

        else:
            # generate term-frequency, inverse-document-frequency vectors
            if self.vector_type == self.TFIDF:
                Vectorizer = TfidfVectorizer
            # generate plain term-frequency vector
            elif self.vector_type == self.TF:
                Vectorizer = CountVectorizer

            self.vectorizer = Vectorizer(max_df=self.max_df, min_df=2,
                                         max_features=self.n_features,
                                         stop_words=STOPWORDS)

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
            'documents with', self.n_topics, 'topics'

        self.model.fit(vectors)

        self._get_topic_names()
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

    def print_top_topics(self, docs, n_top=3):
        """
        Count the number of documents for which each topic is one of the top
        n_top factors, and print the most common topics.
        """
        print 'transforming', len(docs), 'documents...'
        df = self.transform(docs)
        print 'done.'

        print 'counting significant topics...'
        best_topics = []
        for _, r in df.iterrows():
            # filter out all topics that are tied with the least significant one
            significant_topics = [i for i in r.index if r[i] > min(r)]
            best_n = sorted(significant_topics, key=lambda i: r[i],
                            reverse=True)[:n_top]
            best_topics.append(best_n)

        topics = Counter(sum(best_topics, []))
        print 'done.'

        print 'Top topics:'
        for topic, n in sorted(topics.items(), key=lambda i: i[1], reverse=True):
            print '%d:' % n, topic

    def print_topic_shares(self, docs):
        """
        Rank the topics by how much each contributes to the corpus as a whole
        """
        print 'transforming', len(docs), 'documents...'
        df = self.transform(docs)
        print 'done.'

        print 'Top topics:'
        topics = {t: sum(df[t]) for t in df.columns}
        for topic, n in sorted(topics.items(), key=lambda i: i[1], reverse=True):
            print '%.2f:' % n, topic
