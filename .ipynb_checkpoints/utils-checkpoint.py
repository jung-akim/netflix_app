# Run in python console
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords

# Run in terminal or command prompt
# !python3 -m spacy download en
import re
import pandas as pd
import string

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.option_context('display.max_colwidth', 500);


def create_dictionary(texts):
    id2word = corpora.Dictionary(texts)
    return id2word


def lemmatize(texts, allowed_postags=['NOUN']):
    print(f"Lemmatizing...")
    """https://spacy.io/api/annotation"""
    nlp = spacy.load('en', disable=['parser',
                                    'ner'])  # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def make_trigrams(lists_of_words_no_stops, m1=5, t1=.2, s1='npmi', m2=5, t2=.2, s2='npmi'):
    bigram = gensim.models.Phrases(lists_of_words_no_stops, min_count=m1, threshold=t1, scoring=s1)
    trigram = gensim.models.Phrases(bigram[lists_of_words_no_stops], min_count=m2, threshold=t2, scoring=s2)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in lists_of_words_no_stops]


def make_bigrams(lists_of_words_no_stops, min_count=5, threshold=.2, scoring='npmi'):
    print(f"Making bigrams...")
    # Build the bigram models
    bigram = gensim.models.Phrases(lists_of_words_no_stops, min_count=min_count, threshold=threshold,
                                   scoring=scoring)  # higher threshold fewer phrases(bigrams).
    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in lists_of_words_no_stops]


def remove_stopwords(lists_of_words):
    stop_words = stopwords.words('english')
    stop_words.extend(['five_star', 'five', 'star', 'stars', 'netflix'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in lists_of_words]


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def remove_things(text):
    """
    Lowercase, remove punctuation, and remove repeats of more than 2.
    """
    remove_digits_lower = lambda x: re.sub('\w*\d\w*', ' ', x.lower())
    remove_punc = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    remove_repeats = lambda x: re.sub(r'(.)\1+', r'\1\1', x)
    return text.map(remove_digits_lower).map(remove_punc).map(remove_repeats)


class NLPpipe:
    def __init__(self, bigram=True, sent_to_words=sent_to_words, remove_stop_words=remove_stopwords,
                 make_bigrams=make_bigrams, make_trigrams=make_trigrams, lemmatize=lemmatize,
                 create_dict=create_dictionary):
        self.bigram = bigram
        self.remove_things = remove_things
        self.sent_to_words = sent_to_words
        self.remove_stopwords = remove_stopwords
        self.make_bigrams = make_bigrams
        self.make_trigrams = make_trigrams
        self.lemmatize = lemmatize
        self.clean_text = None
        self.create_dictionary = create_dictionary
        self.dictionary = None
        self.min_count = 5
        self.threshold = .2
        self.scoring = 'npmi'

    def preprocess(self, documents):
        """
        Break sentences into words, remove punctuations and stopmake words, make bigrams, and lemmatize the documents
        """
        cleaned_docs = remove_things(documents)
        lists_of_words = list(self.sent_to_words(cleaned_docs))
        lists_of_words_no_stops = self.remove_stopwords(lists_of_words)
        if self.bigram:
            ngrams = self.make_bigrams(lists_of_words_no_stops, self.min_count, self.threshold, self.scoring)
        else:
            ngrams = self.make_trigrams(lists_of_words_no_stops, self.threshold, self.scoring)  # Need to fix parameters
        data_lemmatized = self.lemmatize(ngrams, allowed_postags=['NOUN'])
        return data_lemmatized

    def fit(self, text, min_count=None, threshold=None, scoring=None):
        """
        Create a dictionary after preprocessing.
        """
        if min_count is not None: self.min_count = min_count
        if threshold is not None: self.threshold = threshold
        if scoring is not None: self.scoring = scoring

        self.clean_text = self.preprocess(text)
        self.dictionary = corpora.Dictionary(self.clean_text)

    def transform(self, text, tf_idf=False):
        """
        Return a term-doc matrix using the fit dictionary
        """
        clean_text = self.preprocess(text)
        term_doc = [self.dictionary.doc2bow(text) for text in clean_text]
        if tf_idf:
            return models.TfidfModel(term_doc, smartirs='ntc')[term_doc]
        else:
            return term_doc

    def fit_transform(self, text, tf_idf=False, min_count=None, threshold=None, scoring=None):
        """
        Create a dictionary after preprocessing and return a term-doc matrix using the dictionary.
        """
        if min_count is not None: self.min_count = min_count
        if threshold is not None: self.threshold = threshold
        if scoring is not None: self.scoring = scoring

        self.clean_text = self.preprocess(text)
        self.dictionary = corpora.Dictionary(self.clean_text)
        term_doc = [self.dictionary.doc2bow(text) for text in self.clean_text]
        if tf_idf:
            return models.TfidfModel(term_doc, smartirs='ntc')[term_doc]
        else:
            return term_doc