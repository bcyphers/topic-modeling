import re
import nltk

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import url_parse


class StemTokenizer(object):
    def __init__(self, stem=False):
        self.stem = stem
        if stem:
            self.stemmer = SnowballStemmer('english')

        # hyphenations count as one token, but contractions don't
        self.tokenizer = RegexpTokenizer('\w+(?:[-/]\w+)*')
        self.stopwords = stopwords.words('english')

    def __call__(self, text):
        out = []
        # this line is necessary because links surrounded by lots of periods or
        # commas (......google.com,,,,,,,,,,,,,) break the url regex. Any
        # combination of two or more periods or commas is shortened.
        text = re.sub('\.[\.]+', '.', text)
        text = re.sub(',[,]+', ',', text)

        # replace unicode non-breaking spaces with normal spaces
        text = re.sub(u'\xa0', u' ', text)

        # replace all urls with __link__
        text = re.sub(url_parse.WEB_URL_REGEX, '__link__', text)

        sentences = nltk.tokenize.sent_tokenize(text)

        for sent in sentences:
            for token in self.tokenizer.tokenize(sent):
                # optional: do stemming
                if self.stem:
                    token = self.stemmer.stem(t)
                token = token.lower()
                if len(token) >= 2 and token not in self.stopwords:
                    out.append(token)

        return out
