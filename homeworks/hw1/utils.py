import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bokeh.models as bm, bokeh.plotting as pl
from collections import Counter
from gensim.models import KeyedVectors


def tokenize(text: str):
    """
    :returns: list of tokenized words
    """
    # your code here
    pass


def read_corpus(lang: str):
    """
    read corpus of texts with specified language.
    :args:
        lang (string): ru or be
    :returns:
        list of lists, with tokenized words from the corpus
    """
    assert lang in ['ru', 'be']
    texts = pd.read_csv(f'{lang}.csv')['text']

    # your code here
    pass


def get_distinct_words(corpus, min_count=10):
    """ 
    collect a list of distinct words for the corpus.
    :args:
        corpus (list of list of strings): corpus of texts
        min_count (int): ignores all words with total frequency lower than this
    :returns:
        words (list of strings): sorted list of distinct words across the corpus
        word_counter (collections.Counter()): dict that contains for every word in "words" an anount of times the word appears
    """
    words = []
    word_counter = Counter()

    # your code here

    return words, word_counter


def plot_embeddings(reduced_matrix, token=None, radius=10, alpha=0.25, show=True, color='blue'):
    """ 
    :args:
        reduced_matrix (np.ndarray [n_words, 2]): matrix of 2-dimensioal word embeddings
        token (list): list of tokens that contains captions for each embedding
    """

    if isinstance(color, str):
        color = [color] * len(reduced_matrix)
    data_source = bm.ColumnDataSource({'x': reduced_matrix[:, 0], 'y' : reduced_matrix[:, 1], 'color': color, token=token})

    fig = pl.figure(active_scroll='wheel_zoom', width=600, height=400)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig


def eval_simlex(model: KeyedVectors):
    """
    calculates Spearman's correlation between humans' and model's similarity predictions
    """
    simlex = pd.read_csv('simlex911.csv')
    sims = []
    for row in simlex.iterrows():
        embed1 = model.get_vector(row[1]['word1'])
        embed2 = model.get_vector(row[1]['word2'])

        sims.append(1 - cosine(embed1, embed2))

    corr = spearmanr(np.array(sims), simlex['similarity'])
    return corr.correlation
