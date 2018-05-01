import argparse
import datetime
import os
import pickle 
import uuid

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def build_lda_model(documents, n_topics):

    documents = [document.split() for document in documents]

    dictionary = Dictionary(documents)

    corpus = [dictionary.doc2bow(doc) for doc in documents]

    lda_model = LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=3)

    return dictionary, corpus, lda_model
